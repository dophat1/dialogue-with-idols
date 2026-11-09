"""Dynamic local library, source identity, and legacy edition hints."""
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (PROJECT_ROOT / 'demo_library' if os.getenv('DIALOGUE_DEMO_MODE') == '1'
            else PROJECT_ROOT / 'data' / 'processed')
DEFAULT_BOOK_ID = 'pg1232'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EDITIONS = {
    'pg1232': {'stem': 'the_prince', 'title': 'The Prince', 'chapters': 26, 'commentary_classes': []},
    'pg132': {'stem': 'art_of_war', 'title': 'The Art of War', 'chapters': 13, 'commentary_classes': ['p1']},
}
SEARCHABLE_TYPES = frozenset({'body', 'verse'})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def gutenberg_id(url):
    parsed = urlsplit(url or '')
    if parsed.hostname not in {'www.gutenberg.org', 'gutenberg.org'}:
        return None
    match = re.search(r'/(?:cache/epub|ebooks|files)/(\d+)(?:/|$|\.)', parsed.path)
    return f'pg{match[1]}' if match else None


def resolve_metadata(soup, overrides=None):
    overrides = overrides or {}
    def meta(name):
        nodes = soup.find_all('meta', attrs={'name': re.compile('^'+re.escape(name)+'$', re.I)})
        nodes += soup.find_all('meta', attrs={'property': name})
        return '; '.join(n.get('content', '').strip() for n in nodes if n.get('content')) or None
    source_url = overrides.get('source_url') or meta('og:url') or meta('dcterms.source')
    detected_id = gutenberg_id(source_url)
    book_id = overrides.get('book_id') or detected_id
    if detected_id and overrides.get('book_id') and overrides['book_id'] != detected_id:
        raise ValueError('Book ID override conflicts with Gutenberg source metadata')
    title = meta('dc.title') or meta('og:title')
    if not title and soup.title:
        title = soup.title.get_text(' ', strip=True)
        title = re.sub(r'\s*\|\s*Project Gutenberg.*$', '', title, flags=re.I)
    heading = soup.find('h1')
    title = title or (heading.get_text(' ', strip=True) if heading else None)
    metadata = {'book_id': book_id, 'title': title, 'author': meta('dc.creator'),
                'translator': meta('marcrel.trl'), 'language': meta('dc.language'),
                'source_url': source_url}
    for key in metadata:
        if overrides.get(key) is not None:
            metadata[key] = overrides[key]
    return metadata


def book_paths(book_id=DEFAULT_BOOK_ID, data_dir=DATA_DIR):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', book_id):
        raise ValueError('Invalid book ID')
    data_dir = Path(data_dir)
    bundle = data_dir / book_id
    if bundle.is_dir():
        return bundle / 'book.json', bundle / 'passages.json', bundle / 'vectors.npz'
    stem = EDITIONS.get(book_id, {}).get('stem', book_id)
    return tuple(data_dir / f'{stem}{suffix}' for suffix in ('.json', '_passages.json', '_vectors.npz'))


def catalog(data_dir=DATA_DIR):
    """Discover only complete indexes; unpublished staging directories are ignored."""
    data_dir = Path(data_dir)
    candidates = list(data_dir.glob('*.json'))
    candidates += [p for p in data_dir.glob('*/book.json') if not p.parent.name.startswith('.')]
    books = {}
    for path in candidates:
        if path.stem.endswith('_passages'):
            continue
        try:
            book = json.loads(path.read_text(encoding='utf-8'))
            book_id = book['book_id']
            paths = book_paths(book_id, data_dir)
            if path.resolve() != paths[0].resolve() or not all(p.is_file() for p in paths):
                continue
            books[book_id] = {key: book.get(key) for key in
                ('book_id', 'title', 'author', 'translator', 'language', 'format')}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return books


def source_link(book, chapter=None, paragraph=None):
    url = book.get('source_url')
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        return None
    if book.get('format') in {'epub', 'txt', 'zip'}:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ''))
    anchor = (paragraph or {}).get('source_anchor') or (chapter or {}).get('source_anchor')
    # Legacy inspected records used chapter_id for genuine source anchors.
    if anchor is None and chapter and 'source_anchor' not in chapter:
        anchor = chapter.get('chapter_id')
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query,
                       quote(anchor, safe='') if anchor else parsed.fragment))


def run_book_id(run):
    if run.get('book_id'):
        return run['book_id']
    ids = [item.get('source_id', '') for item in run.get('evidence', [])]
    # Old runs are identified from stored source IDs, never from current UI state.
    if ids and all(source_id.startswith('pg1232:') for source_id in ids):
        return 'pg1232'
    return None


def saved_source_link(run, source):
    if 'source_url' in source:
        return source_link({'source_url': source['source_url']})
    if run_book_id(run) == 'pg1232':
        chapter_id = source['source_id'].split(':')[1]
        return source_link({'source_url': 'https://www.gutenberg.org/cache/epub/1232/pg1232-images.html'},
                           {'source_anchor': chapter_id})
    return None
