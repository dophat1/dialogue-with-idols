"""Shared library/import controls for the two existing Streamlit entry points."""
import os
import streamlit as st

from ingest import import_bytes, import_source, load_embedder
from library import DEFAULT_EMBEDDING_MODEL, catalog
from retrieval import RetrievalModel, load_reranker


@st.cache_resource
def cached_embedding_model(model_name):
    return load_embedder(model_name)


@st.cache_resource
def cached_embedder(model_name=DEFAULT_EMBEDDING_MODEL, language=None):
    language = language or ('en' if model_name.endswith('/all-MiniLM-L6-v2') else None)
    return RetrievalModel(cached_embedding_model(model_name), load_reranker(language))


def library_controls():
    demo_mode = os.getenv('DIALOGUE_DEMO_MODE') == '1'
    if demo_mode:
        with st.sidebar:
            st.info('Portfolio demo: choose a book below. Imports are disabled for this public preview.')
    else:
        with st.sidebar:
            st.header('Add a Gutenberg book')
            with st.form('import_book'):
                address = st.text_input('Gutenberg URL or book number', placeholder='https://www.gutenberg.org/ebooks/11')
                uploaded = st.file_uploader('Or upload a downloaded book', type=['html', 'htm', 'xhtml', 'txt', 'utf-8', 'epub', 'zip'])
                st.caption('HTML, plain text, EPUB, or text/HTML ZIP · Up to 50 MB. Importing builds a local search index.')
                submitted = st.form_submit_button('Import and index')
            if submitted:
                if not address.strip() and uploaded is None:
                    st.warning('Paste a Gutenberg URL/book number or choose a file.')
                else:
                    try:
                        with st.status('Preparing book...', expanded=True) as status:
                            callback = lambda message: status.update(label=message)
                            # Reuse one cached multilingual model for imports and questions.
                            model = cached_embedder()
                            if uploaded is not None:
                                book = import_bytes(uploaded.getvalue(), uploaded.name, embedder=model, progress=callback)
                            else:
                                book = import_source(address.strip(), embedder=model, progress=callback)
                            status.update(label=f"Ready: {book['title']}", state='complete', expanded=False)
                        st.session_state['source_book'] = book['book_id']
                        st.session_state['import_message'] = f"Imported {book['title']}. You can now search or ask a question."
                        st.rerun()
                    except (OSError, ValueError, RuntimeError) as error:
                        st.error(f'Import could not finish: {error}')
            st.caption('Books and indexes stay on this computer. Only selected evidence is sent to OpenAI when you generate an answer.')
    books = catalog()
    if not books:
        st.info('The demo books are unavailable. Please check the deployment logs.' if demo_mode
                else 'Start by importing a Gutenberg book in the sidebar. You can use a book number, URL, or downloaded file.')
        st.stop()
    if st.session_state.get('source_book') not in books:
        st.session_state['source_book'] = next(iter(books))
    selected = st.selectbox('Source book', options=list(books), key='source_book',
        format_func=lambda key: f"{books[key]['title']} · {books[key]['author']}" +
            (f" · {key[-6:]}" if '_' in key else ''))
    if 'import_message' in st.session_state:
        st.success(st.session_state.pop('import_message'))
    return selected


def book_details(book):
    with st.expander('Book and extraction details'):
        st.write(f"Title: {book['title']}")
        st.write(f"Author: {book['author']}")
        if book.get('translator'):
            st.write(f"Translator: {book['translator']}")
        st.write(f"Language: {book.get('language') or 'not recorded'}")
        st.write(f"Sections: {len(book['chapters']):,} · Paragraphs: {sum(len(c['paragraphs']) for c in book['chapters']):,}")
        st.caption('Author/editor attribution is automatic. Unknown editions may contain unmarked editorial material; inspect excerpts before relying on attribution.')
        for warning in book.get('extraction_diagnostics', {}).get('warnings', []):
            st.warning(warning)


def render_passages(search_data, results):
    for number, result in enumerate(results, 1):
        passage = search_data.passages[result['passage_id']]
        chapter = search_data.chapters[passage['chapter_key']]
        with st.expander(f"{number}. {chapter['title']}", expanded=number == 1):
            st.text(passage['text'])
            st.caption(f"{passage['content_type']} · {passage['passage_id']}")
            if chapter.get('source_locator'):
                st.caption(f"In the downloaded book: {chapter['source_locator']}")
            if passage.get('source_url'):
                st.link_button('Open original source', passage['source_url'], key=f"preview-source-{number}-{passage['passage_id']}")
            paragraphs = chapter['paragraphs']
            index = next(i for i, p in enumerate(paragraphs) if p['paragraph_id'] == passage['paragraph_id'])
            st.markdown('**Surrounding context**')
            for position in range(max(0, index-1), min(len(paragraphs), index+2)):
                paragraph = paragraphs[position]
                st.caption(('Matched paragraph' if position == index else 'Neighbor')+' · '+paragraph['content_type'])
                st.text(paragraph['text'][:6000])
                if len(paragraph['text']) > 6000:
                    st.caption('Preview shortened; open the original source for the complete paragraph.')
            st.caption(f"Semantic similarity: {result.get('semantic_score', result['score']):.3f} · Lexical score: {result.get('lexical_score', 0):.2f}. Ranking is not proof of support.")
