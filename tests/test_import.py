import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from bs4 import BeautifulSoup

from src.gutenberg_import import (extract_document, validate_url, safe_member, fetch_source,
                                  MAX_SOURCE_BYTES, strip_markers, readable_download)
from src.ingest import import_bytes
from src.library import catalog, book_paths
from src.semantic_search import load_search_data
from src.retrieval import RetrievalModel, retrieve, evidence_for_results

TEXT = ('The silver key opens the hidden garden behind the old house. ' * 5)
HTML = f'''<html><head><meta name="dc.title" content="A New Book"><meta name="dc.creator" content="New Author"><meta name="dc.language" content="en"><meta property="og:url" content="https://www.gutenberg.org/cache/epub/99999/pg99999-images.html"></head><body><div id="pg-header"><p>License header</p></div><h1 id="first">Beginning</h1><p>{TEXT}</p><h2 id="next">Another section</h2><p>Some other events occur here. {TEXT}</p><div id="pg-footer"><p>License footer</p></div></body></html>'''
PLAIN = f'''The Project Gutenberg eBook\nTitle: Test Book\nAuthor: Test Writer\nLanguage: English\n[eBook #99999]\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\nCHAPTER I. THE GARDEN\n\n{TEXT}\n\nCHAPTER II. THE HOUSE\n\n{TEXT}\n\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicense footer'''


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        import re
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}

    def encode(self, text, **kwargs):
        return list(range(len(text.split())+2))


class FakeModel:
    tokenizer = FakeTokenizer()
    max_seq_length = 128

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(4, dtype=np.float32)/2
        return np.ones((len(texts), 4), dtype=np.float32)/2


def zipped(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return buffer.getvalue()


def epub():
    return zipped({
        'META-INF/container.xml': '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        'OPS/book.opf': '''<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>EPUB Novel</dc:title><dc:creator>A Writer</dc:creator><dc:language>en</dc:language><dc:identifier>https://www.gutenberg.org/ebooks/99999</dc:identifier></metadata><manifest><item id="two" href="two.xhtml" media-type="application/xhtml+xml"/><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="one"/><itemref idref="two"/></spine></package>''',
        'OPS/one.xhtml': f'<html><body><h2 id="same">First</h2><p>{TEXT}</p></body></html>',
        'OPS/two.xhtml': f'<html><body><h2 id="same">Second</h2><p>{TEXT}</p></body></html>',
    })


class ImportTests(unittest.TestCase):
    def test_unregistered_html_and_metadata(self):
        book = extract_document(HTML.encode(), 'new.html')
        self.assertEqual(book['book_id'], 'pg99999')
        self.assertEqual(book['author'], 'New Author')
        self.assertEqual(len(book['chapters']), 2)
        all_text = ' '.join(p['text'] for c in book['chapters'] for p in c['paragraphs'])
        self.assertNotIn('License', all_text)
        self.assertIn('silver key', all_text)
        self.assertEqual(book['chapters'][0]['source_anchor'], 'first')

    def test_legacy_html_header_supplies_missing_metadata(self):
        legacy = f'<html><head><title>The Project Gutenberg eBook</title></head><body><pre>Title: Old Book\nAuthor: Old Writer\n[eBook #99998]\n*** START OF THE PROJECT GUTENBERG EBOOK OLD ***</pre><p>{TEXT}</p></body></html>'
        book = extract_document(legacy.encode(), 'legacy.htm')
        self.assertEqual(book['title'], 'Old Book')
        self.assertEqual(book['author'], 'Old Writer')
        self.assertEqual(book['book_id'], 'pg99998')

    def test_plain_text_encodings_and_boilerplate(self):
        for content in (PLAIN.encode('utf-8'), PLAIN.encode('utf-16')):
            book = extract_document(content, 'book.txt')
            self.assertEqual(book['book_id'], 'pg99999')
            self.assertEqual(book['title'], 'Test Book')
            self.assertEqual(len(book['chapters']), 2)
            self.assertNotIn('License', json.dumps(book['chapters']))
        book = extract_document(PLAIN.replace('Test Writer', 'André').encode('cp1252'), 'book.txt', encoding='cp1252')
        self.assertEqual(book['author'], 'André')

    def test_epub_spine_order_and_unique_citations(self):
        book = extract_document(epub(), 'book.epub')
        self.assertEqual(book['title'], 'EPUB Novel')
        self.assertEqual([c['title'] for c in book['chapters']], ['First', 'Second'])
        self.assertEqual(book['book_id'], 'pg99999')
        self.assertEqual(book['chapters'][0]['source_locator'], 'OPS/one.xhtml#same')
        ids = [p['paragraph_id'] for c in book['chapters'] for p in c['paragraphs']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_zip_html_and_plain_text(self):
        for name, content in [('book.html', HTML), ('book.txt', PLAIN)]:
            book = extract_document(zipped({name: content}), 'book.zip')
            self.assertTrue(book['chapters'])
            self.assertEqual(book['title'], 'A New Book' if name.endswith('html') else 'Test Book')

    def test_unusual_html_fallback_preserves_text(self):
        book = extract_document(f'<html><body>{TEXT}<h2>Second</h2>{TEXT}</body></html>'.encode(), 'unknown.html')
        self.assertTrue(book['extraction_diagnostics']['warnings'])
        self.assertIn('silver key', str(book['chapters']))

    def test_inline_punctuation_and_comments(self):
        body = f'<html><body><h1>Title</h1><p>Hello <em>world</em>.<!--secret comment--><br>Next line. {TEXT}</p></body></html>'
        book = extract_document(body.encode(), 'x.html')
        text = book['chapters'][0]['paragraphs'][0]['text']
        self.assertIn('Hello world. Next line.', text)
        self.assertNotIn('secret comment', text)

    def test_failures_do_not_publish(self):
        for content, name in [(b'%PDF binary', 'book.pdf'), (b'', 'empty.txt'), (b'tiny', 'tiny.txt'),
                              (zipped({'META-INF/container.xml': 'broken'}), 'bad.epub')]:
            with self.assertRaises(ValueError):
                extract_document(content, name)
        class BrokenModel(FakeModel):
            def encode(self, texts, **kwargs):
                raise RuntimeError('simulated embedding failure')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                import_bytes(HTML.encode(), 'book.html', data_dir=tmp, embedder=BrokenModel())
            self.assertEqual(catalog(tmp), {})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_download_validation_and_book_page_resolution(self):
        for url in ('https://example.com/11', 'http://www.gutenberg.org/ebooks/11',
                    'https://www.gutenberg.org:123/11', 'https://user:pass@www.gutenberg.org/11'):
            with self.assertRaises(ValueError):
                validate_url(url)
        page = b'<a type="application/epub+zip" href="/ebooks/99999.epub3.images">EPUB</a><a type="text/html" href="/cache/epub/99999/pg99999-images.html">Read online</a>'
        with patch('src.gutenberg_import.download', side_effect=[(page, 'https://www.gutenberg.org/ebooks/99999'), (HTML.encode(), 'https://www.gutenberg.org/cache/epub/99999/pg99999-images.html')]) as get:
            content, filename, url = fetch_source('99999')
            self.assertEqual(content, HTML.encode())
            self.assertEqual(filename, 'pg99999-images.html')
            self.assertIn('/cache/epub/', get.call_args.args[0])
        with self.assertRaises(ValueError):
            safe_member('OPS', '../../outside')

    def test_site_pages_and_foreign_book_downloads_are_rejected(self):
        with self.assertRaises(ValueError):
            readable_download(b'<html><title>Project Gutenberg</title><p>Contact information</p></html>',
                              'https://www.gutenberg.org/cache/epub/11/pg11-images.html', 'pg11')
        with self.assertRaises(ValueError):
            readable_download(HTML.encode(), 'https://www.gutenberg.org/about/contact_information.html', 'pg99999')
        page = b'<a href="/about/contact_information.html">Contact</a><a type="text/html" href="/cache/epub/99999/pg99999-images.html">Read online</a>'
        with patch('src.gutenberg_import.download', side_effect=[(page, 'https://www.gutenberg.org/ebooks/99999'), (HTML.encode(), 'https://www.gutenberg.org/cache/epub/99999/pg99999-images.html')]) as download:
            fetch_source('99999')
            self.assertNotIn('contact_information', download.call_args.args[0])

    def test_import_publishes_discoverable_loadable_index_and_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = import_bytes(HTML.encode(), 'book.html', data_dir=tmp, embedder=FakeModel())
            self.assertIn(book['book_id'], catalog(tmp))
            loaded = load_search_data(book['book_id'], tmp)
            self.assertEqual(loaded.vectors.shape[1], 4)
            self.assertTrue(all(p.startswith(book['book_id']+':') for p in loaded.passage_ids))
            with patch('src.ingest.load_embedder', side_effect=AssertionError('must reuse')):
                self.assertEqual(import_bytes(HTML.encode(), 'book.html', data_dir=tmp)['book_id'], book['book_id'])
            changed = import_bytes(HTML.replace('silver key', 'golden key').encode(), 'book.html', data_dir=tmp, embedder=FakeModel())
            self.assertNotEqual(changed['book_id'], book['book_id'])
            self.assertEqual(len(catalog(tmp)), 2)

    def test_reranker_promotes_answer_candidate_and_rejects_invalid_scores(self):
        body = f'<html><body><h2>Story</h2><p>{TEXT}</p><p>Correct evidence is here. {"Different events. "*20}</p></body></html>'
        with tempfile.TemporaryDirectory() as tmp:
            book = import_bytes(body.encode(), 'rerank.html', data_dir=tmp, embedder=FakeModel())
            data = load_search_data(book['book_id'], tmp)
            ranker = SimpleNamespace(predict=lambda pairs, **kwargs: np.array([10 if 'Correct evidence' in passage.split('Matched paragraph:', 1)[1].split('Next paragraph:')[0] else 0 for _, passage in pairs]))
            model = RetrievalModel(FakeModel(), ranker)
            result = retrieve(data, 'something', model, limit=1)
            self.assertIn('Correct evidence', data.passages[result[0]['passage_id']]['text'])
            self.assertEqual(result[0]['rerank_score'], 10)
            model.reranker = SimpleNamespace(predict=lambda pairs, **kwargs: np.array([np.nan]*len(pairs)))
            with self.assertRaises(ValueError):
                retrieve(data, 'something', model)

    def test_precise_phrase_evidence_survives_a_poor_neural_ranking(self):
        target = 'The ideal victory defeats an opponent without fighting.'
        decoys = ''.join(f'<p>A general can win a battle by preparing troops. This is another military situation number {i}.</p>' for i in range(7))
        body = f'<html><body><h2>Strategy</h2>{decoys}<p>{target}</p></body></html>'
        with tempfile.TemporaryDirectory() as tmp:
            book = import_bytes(body.encode(), 'strategy.html', data_dir=tmp, embedder=FakeModel())
            data = load_search_data(book['book_id'], tmp)
            ranker = SimpleNamespace(predict=lambda pairs, **kwargs: np.array([-10 if 'without fighting' in context.split('Matched paragraph:', 1)[1].split('Next paragraph:')[0] else 10 for _, context in pairs]))
            results = retrieve(data, 'How can a general win without fighting a battle?', RetrievalModel(FakeModel(), ranker))
            self.assertTrue(any('without fighting' in source['text'] for source in evidence_for_results(data, results)))

    def test_hybrid_retrieval_and_long_source_evidence(self):
        body = '<html><body><h2>Story</h2><p>'+('Unrelated background. '*500)+TEXT+'</p><p>Another event happened in a forest. '+('Different words. '*30)+'</p></body></html>'
        with tempfile.TemporaryDirectory() as tmp:
            book = import_bytes(body.encode(), 'long.html', data_dir=tmp, embedder=FakeModel())
            data = load_search_data(book['book_id'], tmp)
            result = retrieve(data, 'silver key garden', FakeModel(), limit=1)
            self.assertIn('silver key', data.passages[result[0]['passage_id']]['text'])
            source = evidence_for_results(data, result)[0]
            self.assertIn('silver key', source['text'])
            self.assertTrue(source['excerpted'])
            self.assertLessEqual(len(source['text']), 3000)


if __name__ == '__main__':
    unittest.main()
