import copy
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
from bs4 import BeautifulSoup

from src.library import (resolve_metadata, fingerprint, source_link, saved_source_link,
                         run_book_id, book_paths, catalog)
from src.create_passages import create_passages
from src.semantic_search import load_search_data
from src.answer_question import generate_answer, Answer, Claim


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i+1) for i in range(len(text))]}

    def encode(self, text, **kwargs):
        return list(range(len(text)))


class LibraryTests(unittest.TestCase):
    def test_packaged_prince_is_available_without_local_imports(self):
        with tempfile.TemporaryDirectory() as directory, patch('src.library.DATA_DIR', Path(directory)):
            books = catalog()
            self.assertIn('pg1232', books)
            self.assertEqual(books['pg1232']['title'], 'The Prince')
            self.assertEqual(book_paths('pg1232')[0].name, 'the_prince.json')
            self.assertEqual(load_search_data('pg1232').book['title'], 'The Prince')

    def test_metadata_detection_overrides_and_unknown_edition(self):
        html = '<meta name="dc.title" content="A"><meta name="dc.creator" content="B"><meta name="marcrel.trl" content="C"><meta property="og:url" content="https://www.gutenberg.org/cache/epub/132/pg132-images.html">'
        soup = BeautifulSoup(html, 'html.parser')
        metadata = resolve_metadata(soup)
        self.assertEqual(metadata['book_id'], 'pg132')
        self.assertEqual(metadata['translator'], 'C')
        self.assertEqual(resolve_metadata(soup, {'title': 'Override'})['title'], 'Override')
        self.assertEqual(resolve_metadata(BeautifulSoup(html.replace('/132/', '/999/'), 'html.parser'))['book_id'], 'pg999')
        self.assertIsNone(resolve_metadata(BeautifulSoup(html.replace('www.gutenberg.org', 'example.com'), 'html.parser'))['book_id'])
        with self.assertRaises(ValueError):
            resolve_metadata(soup, {'book_id': 'pg1232'})

    def test_source_links_and_legacy_history(self):
        book = {'source_url': 'https://example.org/book'}
        self.assertEqual(source_link(book, {'source_anchor': None, 'chapter_id': 'section0001'}), 'https://example.org/book')
        self.assertEqual(source_link(book, {'source_anchor': 'a b'}), 'https://example.org/book#a%20b')
        self.assertIsNone(source_link({'source_url': 'javascript:alert(1)'}))
        legacy = {'evidence': [{'source_id': 'pg1232:chap01:p0001'}]}
        self.assertEqual(run_book_id(legacy), 'pg1232')
        self.assertTrue(saved_source_link(legacy, legacy['evidence'][0]).endswith('#chap01'))
        self.assertEqual(saved_source_link({'book_id': 'pg132'}, {'source_url': 'https://example.org/132#chap09'}), 'https://example.org/132#chap09')
        self.assertIsNone(saved_source_link({'book_id': 'pg132'}, {'source_id': 'pg132:chap09:p0001'}))

    def test_chunks_exclude_commentary_and_preserve_offsets(self):
        book = {'book_id': 'pg132', 'chapters': [{'chapter_id': 'chap09', 'title': 'I', 'paragraphs': [
            {'paragraph_id': f'pg132:chap09:p{i}', 'content_type': kind, 'text': 'x'*length}
            for i, (kind, length) in enumerate([('body', 300), ('commentary', 20), ('verse', 30), ('footnote', 10)])]}]}
        data = create_passages(book, Tokenizer(), 256)
        self.assertEqual(len(data['passages']), 3)
        self.assertEqual({p['content_type'] for p in data['passages']}, {'body', 'verse'})
        self.assertEqual(data['passages'][1]['char_start'], 180)
        for p in data['passages']:
            parent = next(x for x in book['chapters'][0]['paragraphs'] if x['paragraph_id'] == p['paragraph_id'])
            self.assertEqual(parent['text'][p['char_start']:p['char_end']], p['text'])

    def fixture(self, directory, book_id='pg132'):
        book = {'book_id': book_id, 'title': 'Fixture', 'author': 'Author',
                'source_url': 'https://example.org/book', 'chapters': [
            {'chapter_id': 'chap09', 'title': 'I', 'source_anchor': 'chap09', 'paragraphs': [
                {'paragraph_id': f'{book_id}:chap09:p1', 'content_type': 'body', 'text': 'abc'}]}]}
        data = create_passages(book, Tokenizer(), 256)
        paths = book_paths(book_id, directory)
        paths[0].write_text(json.dumps(book), encoding='utf-8')
        paths[1].write_text(json.dumps(data), encoding='utf-8')
        self.write_vectors(paths[2], data)
        return book, data, paths

    def write_vectors(self, path, data, **changes):
        fields = dict(vectors=np.ones((len(data['passages']), 3)),
                      passage_ids=np.array([p['passage_id'] for p in data['passages']]),
                      embedding_model=np.array(data['embedding_model']),
                      book_id=np.array(data['book_id']), book_sha256=np.array(data['book_sha256']),
                      passages_sha256=np.array(fingerprint(data)))
        fields.update(changes)
        np.savez_compressed(path, **fields)

    def test_selected_sources_and_qualified_chapter_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            for book_id in ('pg132', 'pg1232'):
                self.fixture(tmp, book_id)
            for book_id in ('pg132', 'pg1232'):
                result = load_search_data(book_id, tmp)
                self.assertEqual(set(result.chapters), {f'{book_id}:chap09'})
                self.assertTrue(all(x.startswith(book_id+':') for x in result.passage_ids))
                self.assertEqual(len(tuple(result)), 5)

    def test_stale_text_and_embedding_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            book, data, paths = self.fixture(tmp)
            book['chapters'][0]['paragraphs'][0]['text'] = 'xyz'
            paths[0].write_text(json.dumps(book), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Book changed'):
                load_search_data('pg132', tmp)
            book, data, paths = self.fixture(tmp)
            data['passages'][0]['text'] = 'xyz'
            paths[1].write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Stale'):
                load_search_data('pg132', tmp)
            book, data, paths = self.fixture(tmp)
            self.write_vectors(paths[2], data, vectors=np.array([[np.nan, 1, 1]]))
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                load_search_data('pg132', tmp)

    def test_foreign_index_and_commentary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            book, data, paths = self.fixture(tmp)
            data['book_id'] = 'pg1232'
            paths[1].write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'selected book'):
                load_search_data('pg132', tmp)
            book, data, paths = self.fixture(tmp)
            data['passages'][0]['content_type'] = 'commentary'
            paths[1].write_text(json.dumps(data), encoding='utf-8')
            self.write_vectors(paths[2], data)
            with self.assertRaisesRegex(ValueError, 'Non-author'):
                load_search_data('pg132', tmp)

    def test_generation_metadata_and_saved_citations_for_each_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            for book_id in ('pg132', 'pg1232'):
                self.fixture(tmp, book_id)
                search_data = load_search_data(book_id, tmp)
                embedder = SimpleNamespace(tokenizer=Tokenizer(), max_seq_length=256,
                                           encode=lambda *a, **k: np.ones(3))
                source_id = f'{book_id}:chap09:p1'
                answer = Answer(evidence_status='sufficient', explanation='', claims=[
                    Claim(text='Supported test claim', evidence_type='paraphrase', source_ids=[source_id])], follow_up_question='Why?')
                response = SimpleNamespace(status='completed', output_parsed=answer, model='mock-model', usage=None)
                with ExitStack() as stack:
                    stack.enter_context(patch('src.answer_question.load_dotenv'))
                    stack.enter_context(patch('src.answer_question.os.getenv', side_effect=lambda key: 'mock-value'))
                    client = stack.enter_context(patch('src.answer_question.OpenAI'))
                    client.return_value.responses.parse.return_value = response
                    stack.enter_context(patch('src.answer_question.PROJECT_ROOT', Path(tmp)))
                    stack.enter_context(redirect_stdout(StringIO()))
                    run = generate_answer('Question', '', embedder, search_data)
                    payload = json.loads(client.return_value.responses.parse.call_args.kwargs['input'])
                    self.assertEqual(payload['document_metadata']['book_id'], book_id)
                    self.assertEqual({e['book_id'] for e in payload['evidence']}, {book_id})
                    self.assertEqual(run['book_id'], book_id)
                    self.assertTrue(run['evidence'][0]['source_url'].endswith('#chap09'))
                    self.assertEqual(json.loads((Path(tmp)/'data'/'runs'/f"{run['run_id']}.json").read_text())['book_id'], book_id)
                    with self.assertRaisesRegex(ValueError, 'same book'):
                        generate_answer('Next', 'topic', embedder, search_data,
                                        parent_run={'book_id': 'pg1232' if book_id == 'pg132' else 'pg132'})


if __name__ == '__main__':
    unittest.main()
