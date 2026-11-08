import argparse
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

try:
    from .library import EDITIONS, resolve_metadata
except ImportError:
    from library import EDITIONS, resolve_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_SELECTOR = 'p, li, pre, blockquote, div.poem, div.verse'


def clean_text(element):
    # Preserve inline punctuation while separating lines and nested text blocks.
    def render(node):
        if isinstance(node, Comment):
            return ''
        if isinstance(node, NavigableString):
            return str(node)
        text = ''.join(render(child) for child in node.children)
        if node.name in {'p', 'div', 'li', 'blockquote', 'pre', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            return ' ' + text + ' '
        return text
    return ' '.join(render(element).split())


def classify_paragraph(paragraph, commentary_classes):
    nodes = [paragraph, *paragraph.parents]
    classes = {c for node in nodes if isinstance(node, Tag)
               for c in node.get('class', [])}
    if classes & set(commentary_classes):
        return 'commentary'
    if classes & {'footnote', 'footnotes'} or paragraph.get('role') == 'doc-footnote':
        return 'footnote'
    if classes & {'poem', 'verse', 'stanza'}:
        return 'verse'
    return 'body'


def find_anchor(container, heading):
    for node in (heading, heading.find(id=True), heading.find('a', attrs={'name': True}), container):
        if node is not None:
            anchor = node.get('id') or node.get('name')
            if anchor:
                return anchor
    return None


def extract_chapters(soup, book_id, commentary_classes=(), *,
                     container_selector=None, heading_selector='h1, h2, h3',
                     paragraph_selector=BLOCK_SELECTOR, title_pattern=None,
                     exclude_selectors=(), diagnostics=None, strict_anchors=True):
    """Extract HTML sections in document order; selectors define edition policy.

    Without an explicit container, prefer Gutenberg chapter containers, then
    main/article/body. No author, language or chapter-title pattern is assumed.
    """
    # Work on a copy: callers can still inspect the untouched original document.
    soup = BeautifulSoup(str(soup), 'html.parser')
    for selector in ('script', 'style', 'nav', 'header', 'footer', '#pg-header',
                     '#pg-footer', '.pg-boilerplate', *exclude_selectors):
        for node in soup.select(selector):
            node.decompose()
    if container_selector:
        roots = soup.select(container_selector)
        if not roots:
            raise ValueError(f'Container selector matched nothing: {container_selector}')
    else:
        roots = soup.select('div.chapter')
        if not roots:
            roots = [soup.select_one('main') or soup.select_one('article') or soup.body or soup]
    # Nested selected roots must not duplicate text.
    root_ids = {id(root) for root in roots}
    roots = [root for root in roots if not any(id(p) in root_ids for p in root.parents)]
    pattern = re.compile(title_pattern, re.IGNORECASE) if title_pattern else None
    chapters, used_ids = [], set()
    skipped_sections = 0

    def new_section(title, anchor):
        position = len(chapters) + 1
        chapter_id = anchor or f'section{position:04d}'
        if chapter_id in used_ids:
            if strict_anchors:
                raise ValueError(f'Duplicate section anchor: {chapter_id}')
            chapter_id = f'{chapter_id}-section{position:04d}'
            while chapter_id in used_ids:
                chapter_id += '-x'
        used_ids.add(chapter_id)
        section = {'chapter_id': chapter_id, 'position': position,
                   'title': title, 'source_anchor': anchor, 'paragraphs': []}
        chapters.append(section)
        return section

    for root in roots:
        headings = {id(node) for node in root.select(heading_selector)}
        blocks = root.select(paragraph_selector)
        block_ids = {id(node) for node in blocks}
        # Keep outermost selected blocks so nested p/li/blockquote text appears once.
        blocks = {id(node) for node in blocks
                  if not any(id(parent) in block_ids for parent in node.parents)}
        current = None
        accepted = pattern is None
        for node in root.descendants:
            if not isinstance(node, Tag):
                continue
            if id(node) in headings:
                title = clean_text(node)
                accepted = bool(title) and (pattern is None or bool(pattern.search(title)))
                current = None
                if accepted:
                    anchor = find_anchor(root if root is not soup and root.get('class') and 'chapter' in root.get('class', []) else node, node)
                    current = new_section(title, anchor)
                else:
                    skipped_sections += 1
            elif id(node) in blocks:
                # Headings embedded inside a selected text block are not paragraphs.
                if not accepted or not clean_text(node) or any(id(p) in headings for p in node.parents):
                    continue
                if current is None:
                    current = new_section('Untitled section', None)
                position = len(current['paragraphs']) + 1
                current['paragraphs'].append({
                    'paragraph_id': f"{book_id}:{current['chapter_id']}:p{position:04d}",
                    'position': position,
                    'content_type': classify_paragraph(node, commentary_classes),
                    'html_classes': node.get('class', []),
                    'source_anchor': node.get('id') or node.get('name'),
                    'text': clean_text(node),
                })
    empty = [c['title'] for c in chapters if not c['paragraphs']]
    chapters = [c for c in chapters if c['paragraphs']]
    for position, chapter in enumerate(chapters, 1):
        chapter['position'] = position
    if not chapters:
        raise ValueError('No text extracted. Check container, heading, paragraph and title selectors.')
    if diagnostics is not None:
        diagnostics.update({'empty_sections_omitted': empty,
                            'sections_excluded_by_title': skipped_sections,
                            'section_count': len(chapters),
                            'paragraph_count': sum(len(c['paragraphs']) for c in chapters)})
    return chapters


def validate_book(book, expected_chapters=None):
    chapters = book.get('chapters', [])
    if not chapters:
        raise ValueError('Book has no sections')
    if expected_chapters is not None and len(chapters) != expected_chapters:
        raise ValueError(f'Expected {expected_chapters} sections, extracted {len(chapters)}')
    chapter_ids, paragraph_ids = set(), set()
    for index, chapter in enumerate(chapters, 1):
        if chapter['chapter_id'] in chapter_ids or chapter['position'] != index:
            raise ValueError('Duplicate section ID or invalid section order')
        chapter_ids.add(chapter['chapter_id'])
        if not chapter['paragraphs']:
            raise ValueError(f"Empty section: {chapter['chapter_id']}")
        for position, paragraph in enumerate(chapter['paragraphs'], 1):
            if paragraph['paragraph_id'] in paragraph_ids or paragraph['position'] != position:
                raise ValueError('Duplicate paragraph ID or invalid paragraph order')
            if not paragraph['text'].strip():
                raise ValueError('Empty paragraph')
            paragraph_ids.add(paragraph['paragraph_id'])
    return len(chapters), len(paragraph_ids)


def main():
    try:
        from .gutenberg_import import extract_document
    except ImportError:
        from gutenberg_import import extract_document
    parser = argparse.ArgumentParser(description='Extract Gutenberg HTML, TXT, EPUB or ZIP with automatic metadata.')
    parser.add_argument('input_path', type=Path)
    for name in ('book-id', 'title', 'author', 'translator', 'source-url'):
        parser.add_argument('--'+name)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-chapters', type=int)
    parser.add_argument('--encoding')
    args = parser.parse_args()
    try:
        book = extract_document(args.input_path.read_bytes(), args.input_path.name,
                                source_url=args.source_url, overrides=vars(args), encoding=args.encoding)
        validate_book(book, args.expected_chapters)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output = args.output or PROJECT_ROOT / 'data' / 'processed' / f'{args.input_path.stem}.json'
    if output.resolve() == args.input_path.resolve():
        parser.error('Output must not overwrite the original source')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(book, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(book['extraction_diagnostics'], ensure_ascii=True))
    print(f'Saved to: {output}')


if __name__ == '__main__':
    main()
