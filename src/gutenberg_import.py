"""Gutenberg downloads and format adapters; no per-book configuration required."""
import hashlib
import io
import posixpath
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from charset_normalizer import from_bytes

try:
    from .extract_book import BLOCK_SELECTOR, clean_text, extract_chapters, validate_book
    from .library import EDITIONS, gutenberg_id, resolve_metadata
except ImportError:
    from extract_book import BLOCK_SELECTOR, clean_text, extract_chapters, validate_book
    from library import EDITIONS, gutenberg_id, resolve_metadata

MAX_SOURCE_BYTES = 50 * 1024 * 1024
MAX_UNPACKED_BYTES = 100 * 1024 * 1024
START = re.compile(r'\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\r\n]*?\*{3}', re.I)
END = re.compile(r'\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\r\n]*?\*{3}', re.I)
HEADING = re.compile(r'^(?:(?:chapter|book|part|act|scene|volume|section|canto)\s+\S+|[IVXLCDM]+[.\s]|第.{1,20}[章回部]).*$', re.I)
PARSER_VERSION = 'gutenberg-auto-v1'


def validate_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in {'www.gutenberg.org', 'gutenberg.org'}
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('Use an https://www.gutenberg.org book or download URL.')
    return url


class GutenbergRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url):
    validate_url(url)
    request = Request(url, headers={'User-Agent': 'DialogueWithIdols/1.0 (personal ebook reader)'})
    with build_opener(GutenbergRedirect()).open(request, timeout=30) as response:
        validate_url(response.url)
        size = response.headers.get('Content-Length')
        if size and int(size) > MAX_SOURCE_BYTES:
            raise ValueError('Book exceeds the 50 MB import limit.')
        content = response.read(MAX_SOURCE_BYTES + 1)
        if len(content) > MAX_SOURCE_BYTES:
            raise ValueError('Book exceeds the 50 MB import limit.')
        return content, response.url


def readable_download(content, url, expected_id):
    if gutenberg_id(url) != expected_id:
        raise ValueError('Download redirected away from the requested Gutenberg book.')
    if zipfile.is_zipfile(io.BytesIO(content)):
        return
    text = decode_text(content)
    if START.search(text) or re.search(r'\[e-?book\s*#\s*'+expected_id[2:]+r'\]', text, re.I):
        return
    soup = BeautifulSoup(content, 'html.parser')
    metadata = resolve_metadata(soup)
    if metadata.get('book_id') == expected_id and metadata.get('title'):
        return
    raise ValueError('The server returned a site/error page instead of readable book content.')


def fetch_source(value):
    """Accept a book number, book page, or direct text/HTML/EPUB/ZIP URL."""
    value = value.strip()
    if re.fullmatch(r'(?:pg)?\d+', value, re.I):
        value = 'https://www.gutenberg.org/ebooks/' + re.sub(r'\D', '', value)
    validate_url(value)
    expected_id = gutenberg_id(value)
    if not expected_id:
        raise ValueError('Use a Gutenberg ebook page or a download URL containing its ebook number.')
    parsed = urlsplit(value)
    is_page = bool(re.fullmatch(r'/ebooks/\d+/?', parsed.path))
    unsupported = bool(re.search(r'\.(?:mobi|azw3?|pdf|kindle)(?:[./]|$)', parsed.path, re.I))
    if not is_page and not unsupported:
        content, url = download(value)
        readable_download(content, url, expected_id)
        return content, Path(urlsplit(url).path).name, url
    page_url = f'https://www.gutenberg.org/ebooks/{expected_id[2:]}'
    candidates = []
    failures = []
    try:
        page, final_url = download(page_url)
        if gutenberg_id(final_url) != expected_id:
            raise ValueError('Book page redirected away from the requested ebook.')
        soup = BeautifulSoup(page, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = urljoin(final_url, link['href'])
            try:
                validate_url(href)
            except ValueError:
                continue
            # Site navigation/help pages must never become candidate book files.
            if gutenberg_id(href) != expected_id:
                continue
            kind = link.get('type', '').lower()
            path = urlsplit(href).path.lower()
            label = link.get_text(' ', strip=True).lower()
            score = None
            if kind.startswith('text/html') or path.endswith(('.html', '.htm')) or 'read online' in label:
                score = 0
            elif kind.startswith('text/plain') or '.txt' in path:
                score = 1
            elif 'epub' in kind or '.epub' in path:
                score = 2
            if score is not None and not re.search(r'/ebooks/\d+/?$', path):
                candidates.append((score, href))
    except (OSError, ValueError) as error:
        failures.append(type(error).__name__)
    # Gutenberg's standard cache paths also work when page discovery is unavailable.
    number = expected_id[2:]
    candidates += [(3, f'https://www.gutenberg.org/cache/epub/{number}/pg{number}-images.html'),
                   (4, f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt')]
    attempted = set()
    for _, url in sorted(set(candidates))[:8]:
        if url in attempted:
            continue
        attempted.add(url)
        try:
            content, final = download(url)
            readable_download(content, final, expected_id)
            return content, Path(urlsplit(final).path).name, final
        except (OSError, ValueError) as error:
            failures.append(type(error).__name__)
    raise ValueError('No readable HTML, plain-text, or EPUB edition could be downloaded. Try uploading a downloaded book.'
                     + (f' Download errors: {", ".join(failures)}.' if failures else ''))


def decode_text(content, encoding=None):
    if encoding:
        return content.decode(encoding)
    if content.startswith((b'\xff\xfe', b'\xfe\xff')):
        return content.decode('utf-16')
    try:
        return content.decode('utf-8-sig')
    except UnicodeDecodeError:
        best = from_bytes(content).best()
        if best is None:
            raise ValueError('Text encoding could not be detected; use --encoding.')
        return str(best)


def strip_markers(text):
    start = START.search(text)
    if start:
        text = text[start.end():]
    end = END.search(text)
    if end:
        text = text[:end.start()]
    return text.strip()


def text_metadata(text):
    header = text[:START.search(text).start()] if START.search(text) else text[:12000]
    metadata = {}
    for key, label in [('title', 'Title'), ('author', 'Author'), ('translator', 'Translator'), ('language', 'Language')]:
        match = re.search(r'^'+label+r':\s*([^\r\n]+)((?:\r?\n[ \t]+[^\r\n]+)*)', header, re.I | re.M)
        if match:
            metadata[key] = ' '.join(match.group(1, 2)).strip()
            metadata[key] = ' '.join(metadata[key].split())
    match = re.search(r'(?:e-?book\s*(?:no\.?|number|#)|\[e-?book\s*#)\s*(\d+)', text[:15000], re.I)
    if match:
        metadata['book_id'] = 'pg' + match[1]
    return metadata


def plain_chapters(text, book_id):
    text = strip_markers(text).replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\s*\n', text)
    chapters, current = [], None
    for block in blocks:
        block = ' '.join(block.split())
        if not block:
            continue
        is_heading = len(block) < 180 and bool(HEADING.match(block))
        if current is None or is_heading:
            current = {'chapter_id': f'section{len(chapters)+1:04d}', 'position': len(chapters)+1,
                       'title': block if is_heading else 'Text', 'source_anchor': None, 'paragraphs': []}
            chapters.append(current)
            if is_heading:
                continue
        # Bound exceptionally large paragraphs without dropping text.
        while block:
            stop = len(block)
            if stop > 6000:
                stop = block.rfind(' ', 3000, 6000)
                if stop < 0:
                    stop = 6000
            part, block = block[:stop], block[stop:].lstrip()
            position = len(current['paragraphs'])+1
            current['paragraphs'].append({'paragraph_id': f"{book_id}:{current['chapter_id']}:p{position:04d}",
                'position': position, 'content_type': 'body', 'html_classes': [], 'source_anchor': None, 'text': part})
    chapters = [chapter for chapter in chapters if chapter['paragraphs']]
    for position, chapter in enumerate(chapters, 1):
        chapter['position'] = position
    return chapters


def trim_html(soup):
    for node in soup.select('script, style, nav, header, footer, #pg-header, #pg-footer, .pg-boilerplate, #toc, .toc, [role="doc-toc"]'):
        node.decompose()
    # Gutenberg TOCs are often unlabelled tables of internal chapter links.
    for node in list(soup.select('table, ol, ul')):
        if node.parent is None:
            continue
        links = node.find_all('a', href=True)
        linked = sum(len(link.get_text(' ', strip=True)) for link in links)
        if len(links) >= 3 and linked >= .7 * max(1, len(clean_text(node))):
            node.decompose()
    root = soup.body or soup
    strings = list(root.find_all(string=True))
    starts = [(i, START.search(str(node))) for i, node in enumerate(strings) if START.search(str(node))]
    ends = [(i, END.search(str(node))) for i, node in enumerate(strings) if END.search(str(node))]
    start_index, start = starts[0] if starts else (-1, None)
    end_index, end = next(((i, match) for i, match in ends if i >= start_index), (len(strings), None))
    for index, node in enumerate(strings):
        if index < start_index or index > end_index:
            node.extract()
        elif index == start_index == end_index:
            node.replace_with(str(node)[start.end():end.start()])
        elif index == start_index:
            node.replace_with(str(node)[start.end():])
        elif index == end_index:
            node.replace_with(str(node)[:end.start()])


def html_chapters(soup, book_id, warnings, use_edition=True):
    trim_html(soup)
    edition = EDITIONS.get(book_id) if use_edition else None
    diagnostics = {}
    if edition:
        try:
            chapters = extract_chapters(soup, book_id, edition['commentary_classes'],
                                        heading_selector='h2', title_pattern=r'^chapter\s+', diagnostics=diagnostics)
            if len(chapters) == edition['chapters']:
                return chapters, diagnostics
        except ValueError:
            pass
        warnings.append('Known edition layout changed; extracted all readable sections instead.')
    if soup.body is None:
        wrapped = BeautifulSoup('<html><body></body></html>', 'html.parser')
        for node in list(soup.contents):
            wrapped.body.append(node.extract())
        soup = wrapped
    # Leaf text containers cover Gutenberg layouts using divs instead of paragraphs.
    selector = BLOCK_SELECTOR + ', tr, [data-reader-block]'
    for node in soup.select('div, section, article'):
        if not node.select('p, li, pre, blockquote, h1, h2, h3, h4, h5, h6, div, section, article, tr') and clean_text(node):
            node['data-reader-block'] = 'true'
    text_before = clean_text(soup.body)
    try:
        chapters = extract_chapters(soup, book_id, EDITIONS.get(book_id, {}).get('commentary_classes', []),
            container_selector='body', heading_selector='h1, h2, h3, h4, h5, h6',
            paragraph_selector=selector, strict_anchors=False, diagnostics=diagnostics)
        captured = sum(len(p['text']) for c in chapters for p in c['paragraphs'])
        headings = sum(len(c['title']) for c in chapters)
        if captured + headings < len(text_before) * .85:
            raise ValueError('Substantial text outside structured blocks')
    except ValueError:
        warnings.append('Unusual HTML layout: extracted continuous text; section structure and anchors may be unavailable.')
        chapters = plain_chapters(soup.body.get_text('\n\n'), book_id)
        diagnostics = {'layout_fallback': 'continuous-text'}
    if not START.search(str(soup)):
        # The markers may already have been removed; this does not claim perfect attribution.
        diagnostics['attribution_policy'] = 'known edition classes or explicit HTML footnote/verse markers'
    return chapters, diagnostics


def checked_archive(content):
    archive = zipfile.ZipFile(io.BytesIO(content))
    entries = archive.infolist()
    if len(entries) > 3000 or sum(item.file_size for item in entries) > MAX_UNPACKED_BYTES:
        archive.close()
        raise ValueError('Archive is too large to import (100 MB unpacked / 3000 entries).')
    if any(item.flag_bits & 1 for item in entries):
        archive.close()
        raise ValueError('Encrypted archives are not supported.')
    return archive


def safe_member(base, href):
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        raise ValueError('EPUB content must be inside the archive.')
    path = posixpath.normpath(posixpath.join(base, unquote(parsed.path)))
    if path.startswith(('../', '/')) or path == '..' or '\\' in path:
        raise ValueError('Invalid EPUB member path.')
    return path


def archive_document(content, filename, warnings):
    with checked_archive(content) as archive:
        names = archive.namelist()
        if 'META-INF/container.xml' in names:
            container = ET.fromstring(archive.read('META-INF/container.xml'))
            rootfile = next((node.get('full-path') for node in container.iter() if node.tag.endswith('rootfile')), None)
            if not rootfile:
                raise ValueError('EPUB package document is missing.')
            rootfile = safe_member('', rootfile)
            package = ET.fromstring(archive.read(rootfile))
            metadata, items, spine = {}, {}, []
            translators = []
            authors = []
            roles = {node.get('refines', '').lstrip('#'): ''.join(node.itertext()).strip()
                     for node in package.iter() if node.get('property') == 'role'}
            for node in package.iter():
                tag = node.tag.rsplit('}', 1)[-1]
                text = ' '.join(''.join(node.itertext()).split())
                if tag in ('title', 'language') and text:
                    key = tag
                    metadata[key] = '; '.join(filter(None, (metadata.get(key), text)))
                if tag in ('creator', 'contributor'):
                    role = roles.get(node.get('id')) or next((value for key, value in node.attrib.items() if key.endswith('role')), None)
                    if role == 'trl':
                        translators.append(text)
                    elif tag == 'creator' and role in (None, 'aut'):
                        authors.append(text)
                if tag == 'identifier':
                    match = re.search(r'(?:gutenberg(?:\.org/(?:ebooks/|[^ ]*?))?|ebooks/)(\d+)', text, re.I)
                    if match:
                        metadata['book_id'] = 'pg'+match[1]
                if tag == 'item':
                    items[node.get('id')] = node.attrib
                if tag == 'itemref' and node.get('linear', 'yes') != 'no':
                    spine.append(node.get('idref'))
            metadata['author'] = '; '.join(authors) or None
            metadata['translator'] = '; '.join(translators) or None
            docs = []
            for identity in spine:
                item = items.get(identity)
                if not item:
                    raise ValueError('EPUB spine references a missing manifest item.')
                if 'nav' in item.get('properties', '').split():
                    continue
                if item.get('media-type') in ('application/xhtml+xml', 'text/html'):
                    member = safe_member(posixpath.dirname(rootfile), item['href'])
                    docs.append((member, archive.read(member)))
            if not docs:
                raise ValueError('EPUB has no readable HTML spine. Image-only books need OCR.')
            return metadata, docs, 'epub'
        html = [name for name in names if name.lower().endswith(('.html', '.htm', '.xhtml')) and not name.startswith('__MACOSX/')]
        text = [name for name in names if name.lower().endswith(('.txt', '.txt.utf-8')) and not name.startswith('__MACOSX/')]
        if html:
            # Natural filename order is a best effort for non-EPUB archives.
            html.sort(key=lambda name: [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)])
            if len(html) > 1:
                warnings.append('Multi-file HTML ZIP: using natural filename order; verify the section preview.')
            return {}, [(name, archive.read(name)) for name in html], 'zip'
        if len(text) == 1:
            return {}, [(text[0], archive.read(text[0]))], 'txt'
        raise ValueError('ZIP must contain HTML, a single text book, or an EPUB package.')


def extract_document(content, filename, source_url=None, overrides=None, encoding=None):
    if not content or len(content) > MAX_SOURCE_BYTES:
        raise ValueError('Upload a nonempty book of at most 50 MB.')
    overrides = overrides or {}
    warnings = []
    diagnostics = {}
    docs = []
    fmt = 'html' if re.search(r'\.(?:x?html?|html?\.utf-8)$', filename, re.I) else 'txt'
    try:
        if zipfile.is_zipfile(io.BytesIO(content)):
            metadata, docs, fmt = archive_document(content, filename, warnings)
        elif content.startswith((b'%PDF', b'\x89PNG', b'GIF8', b'\xff\xd8', b'ID3')) or b'BOOKMOBI' in content[:100]:
            raise ValueError('This is not a readable text format. Import its Gutenberg book URL to use the HTML/text edition instead.')
        else:
            text = decode_text(content, encoding)
            if '\x00' in text or sum(ord(ch) < 9 for ch in text[:20000]) > 10:
                raise ValueError('Binary content cannot be indexed. Use HTML, TXT, EPUB or a text/HTML ZIP.')
            if re.search(r'<(?:!doctype\s+html|html|head|body|p|h[1-6])\b', text[:20000], re.I):
                fmt = 'html'
            if fmt == 'html':
                docs = [(filename, content)]
                html = BeautifulSoup(content, 'html.parser', from_encoding=encoding)
                metadata = resolve_metadata(html, {'source_url': source_url} if source_url else {})
                header_metadata = text_metadata(html.get_text('\n', strip=True))
                for key, value in header_metadata.items():
                    if value and (not metadata.get(key) or (key == 'title' and 'gutenberg' in metadata[key].casefold())):
                        metadata[key] = value
            else:
                metadata = text_metadata(text)
                docs = [(filename, content)]
        if docs and fmt in ('epub', 'zip'):
            for _, document in docs:
                candidate = resolve_metadata(BeautifulSoup(document, 'html.parser'))
                for key, value in candidate.items():
                    if value and not metadata.get(key):
                        metadata[key] = value
        if fmt == 'txt' and docs:
            metadata.update(text_metadata(decode_text(docs[0][1], encoding)))
        url_id = gutenberg_id(source_url)
        base_id = overrides.get('book_id') or metadata.get('book_id') or url_id
        if url_id and base_id and url_id != base_id:
            raise ValueError('Downloaded content identifies a different Gutenberg book.')
        base_id = base_id or 'local'+hashlib.sha256(content).hexdigest()[:16]
        if not re.fullmatch(r'[A-Za-z0-9_-]+', base_id):
            raise ValueError('Invalid book ID.')
        book = dict(metadata)
        book['book_id'] = base_id
        for key in ('title', 'author', 'translator', 'language', 'source_url'):
            if overrides.get(key) is not None:
                book[key] = overrides[key]
        book['source_url'] = source_url or book.get('source_url')
        if not book.get('source_url') and base_id.startswith('pg') and base_id[2:].isdigit():
            book['source_url'] = f'https://www.gutenberg.org/ebooks/{base_id[2:]}'
        if not book.get('title'):
            book['title'] = Path(filename).stem
            warnings.append('Title not found in metadata; using the filename.')
        if not book.get('author'):
            book['author'] = 'Unknown author'
            warnings.append('Author not found in metadata; attribution requires inspection.')
        chapters = []
        if fmt == 'txt':
            chapters = plain_chapters(decode_text(docs[0][1], encoding), base_id)
        else:
            for member, document in docs:
                soup = BeautifulSoup(document, 'html.parser', from_encoding=encoding)
                # Empty EPUB cover and license-only documents do not fail the entire book.
                try:
                    sections, detail = html_chapters(soup, base_id, warnings, use_edition=len(docs) == 1)
                except ValueError:
                    warnings.append(f'No readable text in archive section: {member}')
                    continue
                diagnostics[member] = detail
                for section in sections:
                    if len(docs) > 1 or fmt == 'epub':
                        original_anchor = section.get('source_anchor')
                        section['source_locator'] = member + ('#'+original_anchor if original_anchor else '')
                        section['chapter_id'] = f'section{len(chapters)+1:04d}'
                        section['source_anchor'] = None
                        for paragraph in section['paragraphs']:
                            paragraph['source_anchor'] = None
                    section['position'] = len(chapters)+1
                    for index, paragraph in enumerate(section['paragraphs'], 1):
                        paragraph['paragraph_id'] = f"{base_id}:{section['chapter_id']}:p{index:04d}"
                    chapters.append(section)
        book.update({'chapters': chapters, 'format': fmt, 'source_filename': Path(filename).name,
                     'source_sha256': hashlib.sha256(content).hexdigest(), 'ingestion_version': PARSER_VERSION})
        validate_book(book)
        total = sum(len(p['text']) for c in chapters for p in c['paragraphs'])
        if total < 100:
            raise ValueError('Too little readable text was extracted. This may be an image-only or damaged book.')
        if fmt in ('txt', 'epub', 'zip') and book.get('source_url') and base_id.startswith('pg') and base_id[2:].isdigit():
            book['source_url'] = f'https://www.gutenberg.org/ebooks/{base_id[2:]}'
        book['extraction_diagnostics'] = {'section_count': len(chapters),
            'paragraph_count': sum(len(c['paragraphs']) for c in chapters),
            'character_count': total, 'warnings': list(dict.fromkeys(warnings)), 'sections': diagnostics}
        return book
    except (zipfile.BadZipFile, KeyError, ET.ParseError, UnicodeError) as error:
        raise ValueError(f'Book could not be decoded or its archive is incomplete: {type(error).__name__}') from error
