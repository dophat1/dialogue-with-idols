"""UI smoke tests use prepared local indexes and stub model/generation calls."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from library import book_paths


class FakeEmbedder:
    max_seq_length = 256
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1]*len(text.split()))

    def encode(self, *args, **kwargs):
        return np.ones(384, dtype=np.float32) / np.sqrt(384)


@unittest.skipUnless(all(path.exists() for key in ('pg132', 'pg1232') for path in book_paths(key)), 'Prepare both local indexes first')
class AppTests(unittest.TestCase):
    def test_discussion_suggestions_prefill_without_generating(self):
        prince_questions = (
            "Is it better for a leader to be loved or feared?",
            "What does Machiavelli say about choosing advisers?",
            "How can a leader encourage people to speak honestly?",
        )
        war_questions = (
            "Why is knowing yourself as important as knowing your opponent?",
            "What does Sun Tzu say about preparing before taking action?",
            "How should a leader adapt when circumstances change?",
        )
        with patch('library_ui.load_embedder', return_value=FakeEmbedder()), patch('library_ui.load_reranker', return_value=None), \
                patch('answer_question.generate_answer') as generate:
            app = AppTest.from_file(str(ROOT/'src'/'discussion_app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            initial_suggestion = prince_questions[0] if app.selectbox[0].value == 'pg1232' else war_questions[0]
            self.assertEqual(app.session_state['question'], initial_suggestion)

            # Switching to The Prince initializes its first prompt.
            app.selectbox[0].set_value('pg1232').run()
            self.assertEqual(app.session_state['question'], prince_questions[0])
            self.assertTrue(all(any(b.label == question for b in app.button) for question in prince_questions))

            # Clicking a suggestion fills the box, but does not submit/generate.
            next(b for b in app.button if b.label == prince_questions[2]).click().run()
            self.assertEqual(app.text_area[0].value, prince_questions[2])
            generate.assert_not_called()

            # A custom prompt survives a normal rerun, then switching books resets it.
            app.text_area[0].set_value('My own question about the book').run()
            self.assertEqual(app.text_area[0].value, 'My own question about the book')
            app.selectbox[0].set_value('pg132').run()
            self.assertEqual(app.session_state['question'], war_questions[0])
            self.assertTrue(all(any(b.label == question for b in app.button) for question in war_questions))
            next(b for b in app.button if b.label == war_questions[2]).click().run()
            self.assertEqual(app.text_area[0].value, war_questions[2])
            generate.assert_not_called()

            app.selectbox[0].set_value('pg1232').run()
            self.assertEqual(app.session_state['question'], prince_questions[0])

    def test_search_book_switch_clears_results(self):
        with patch('library_ui.load_embedder', return_value=FakeEmbedder()), patch('library_ui.load_reranker', return_value=None):
            app = AppTest.from_file(str(ROOT/'src'/'search_app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.selectbox[0].set_value('pg1232').run()
            next(w for w in app.text_input if w.label == 'What would you like to explore?').set_value('leadership')
            next(w for w in app.button if w.label == 'Search').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['search_results']['results'])
            self.assertTrue(all(x['passage_id'].startswith('pg1232:') for x in app.session_state['search_results']['results']))
            app.selectbox[0].set_value('pg132').run()
            self.assertFalse(app.exception)
            self.assertNotIn('search_results', app.session_state)
            next(w for w in app.text_input if w.label == 'What would you like to explore?').set_value('planning')
            next(w for w in app.button if w.label == 'Search').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(all(x['passage_id'].startswith('pg132:') for x in app.session_state['search_results']['results']))

    def test_discussion_switch_and_saved_run_render(self):
        with patch('library_ui.load_embedder', return_value=FakeEmbedder()), patch('library_ui.load_reranker', return_value=None):
            app = AppTest.from_file(str(ROOT/'src'/'discussion_app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.selectbox[0].set_value('pg132').run()
            self.assertFalse(app.exception)
            fake_run = {
                'run_id': 'ui-test-not-written', 'created_at': '2026-11-09', 'book_id': 'pg132',
                'question': 'Question', 'research_query': 'topic', 'generation_model_returned': 'mock',
                'usage': None, 'evidence': [{'source_id': 'pg132:chap09:p0001', 'chapter': 'Chapter I',
                    'content_type': 'body', 'text': 'Evidence',
                    'source_url': 'https://www.gutenberg.org/cache/epub/132/pg132-images.html#chap09'}],
                'answer': {'evidence_status': 'sufficient', 'explanation': '',
                    'claims': [{'text': 'Claim', 'evidence_type': 'paraphrase', 'source_ids': ['pg132:chap09:p0001']}],
                    'follow_up_question': 'Why?'},
            }
            with patch('answer_question.generate_answer', return_value=fake_run) as generate:
                app.text_area[0].set_value('Question')
                next(b for b in app.button if b.label == 'Generate cited answer').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(generate.call_args.args[3].book['book_id'], 'pg132')
                self.assertEqual(app.session_state['discussion_run']['book_id'], 'pg132')
                self.assertTrue(any('132/pg132-images.html#chap09' in b.url for b in app.get('link_button')))
            app.selectbox[0].set_value('pg1232').run()
            self.assertFalse(app.exception)
            self.assertNotIn('discussion_run', app.session_state)


class ImportAppTests(unittest.TestCase):
    def test_empty_library_url_import_preview_and_question(self):
        import tempfile
        import streamlit as st
        from src.ingest import import_bytes
        from src.library import catalog as read_catalog
        from src.semantic_search import load_search_data as read_index
        from src.retrieval import retrieve, evidence_for_results
        from test_import import HTML, FakeModel
        st.cache_resource.clear()
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel()
            def import_fixture(value, **kwargs):
                return import_bytes(HTML.encode(), 'uploaded.html', data_dir=tmp, embedder=model)
            def generate_fixture(question, research_query, embedder, data, parent_run=None):
                evidence = evidence_for_results(data, retrieve(data, question, embedder))
                return {'run_id': 'mock-ui-import', 'book_id': data.book['book_id'],
                        'created_at': '2026-11-09', 'question': question, 'research_query': research_query,
                        'generation_model_returned': 'mock', 'usage': None, 'evidence': evidence,
                        'answer': {'evidence_status': 'sufficient', 'explanation': '', 'claims': [
                            {'text': 'The key opens the garden.', 'evidence_type': 'paraphrase',
                             'source_ids': [evidence[0]['source_id']]}], 'follow_up_question': 'What next?'}}
            with patch('library_ui.catalog', side_effect=lambda: read_catalog(tmp)), \
                 patch('library_ui.import_source', side_effect=import_fixture), \
                 patch('library_ui.cached_embedder', return_value=model), \
                 patch('semantic_search.load_search_data', side_effect=lambda identity: read_index(identity, tmp)), \
                 patch('answer_question.generate_answer', side_effect=generate_fixture):
                app = AppTest.from_file(str(ROOT/'src'/'discussion_app.py'), default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.selectbox), 0)
                next(w for w in app.text_input if w.label == 'Gutenberg URL or book number').set_value('99999')
                next(w for w in app.button if w.label == 'Import and index').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.selectbox), 1)
                self.assertTrue(app.selectbox[0].value.startswith('pg99999_'))
                self.assertFalse(any('Not sure where to start?' in item.value for item in app.markdown))
                self.assertTrue(app.text_area)
                app.text_area[0].set_value('What opens the hidden garden?')
                next(w for w in app.button if w.label == 'Find passages').click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.session_state['retrieved_preview']['results'])
                next(w for w in app.button if w.label == 'Generate cited answer').click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.session_state['discussion_run']['book_id'].startswith('pg99999_'))
        st.cache_resource.clear()


if __name__ == '__main__':
    unittest.main()
