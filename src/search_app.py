import streamlit as st

from semantic_search import load_search_data
from library_ui import library_controls, book_details, cached_embedder, render_passages
from retrieval import retrieve

st.set_page_config(page_title='Dialogue with Idols · Search', layout='wide')
st.title('Dialogue with Idols')
st.caption('Search your Gutenberg library. This page does not make OpenAI requests.')
selected_book = library_controls()
if st.session_state.get('search_book') != selected_book:
    st.session_state.pop('search_results', None)
    st.session_state['search_book'] = selected_book
try:
    search_data = load_search_data(selected_book)
    model = cached_embedder(search_data.model_name, search_data.book.get('language'))
    book_details(search_data.book)
except (OSError, ValueError, KeyError) as error:
    st.error(f'Could not load search data: {error}')
    st.stop()

with st.form('search'):
    query = st.text_input('What would you like to explore?', placeholder='Ask a question or search for a phrase.')
    submitted = st.form_submit_button('Search')
if submitted:
    try:
        with st.spinner('Searching the book...'):
            results = retrieve(search_data, query, model)
        st.session_state['search_results'] = {'query': query, 'results': results}
    except (ValueError, RuntimeError) as error:
        st.session_state.pop('search_results', None)
        st.warning(str(error))
search_results = st.session_state.get('search_results')
if search_results:
    st.subheader('Search results')
    st.write(f"For: {search_results['query']}")
    render_passages(search_data, search_results['results'])
else:
    st.info('Enter a question to find relevant passages.')
