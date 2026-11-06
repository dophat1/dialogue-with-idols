import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from semantic_search import load_search_data


st.set_page_config(
    page_title="Dialogue with Idols",
    layout="wide",
)


@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name, device="cpu")


st.title("Dialogue with Idols")
st.caption("Explore passages from The Prince — Niccolò Machiavelli")
st.caption("English translation by W. K. Marriott")

# Read the saved files again on each rerun.
# The model itself is cached because loading it is expensive.
try:
    passages, chapters, vectors, passage_ids, model_name = (
        load_search_data()
    )
    model = load_model(model_name)
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load search data: {error}")
    st.stop()


with st.form("search"):
    query = st.text_input(
        "What would you like to explore?",
        placeholder="How can a ruler use fear without becoming hated?",
    )
    submitted = st.form_submit_button("Search")


if submitted:
    query = query.strip()

    if not query:
        st.warning("Enter a question or a few search words.")
        st.stop()

    query_length = len(model.tokenizer.encode(
        query,
        truncation=False,
    ))

    if query_length > model.max_seq_length:
        st.warning("Please shorten your question.")
        st.stop()

    with st.spinner("Searching the book..."):
        query_vector = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = vectors @ query_vector
        ranked_indices = np.argsort(-scores)

        results = []
        seen_paragraphs = set()

        for index in ranked_indices:
            passage = passages[passage_ids[index]]
            paragraph_id = passage["paragraph_id"]

            if paragraph_id in seen_paragraphs:
                continue

            results.append({
                "passage_id": passage["passage_id"],
                "score": float(scores[index]),
            })
            seen_paragraphs.add(paragraph_id)

            if len(results) == 3:
                break

    st.session_state["search_results"] = {
        "query": query,
        "results": results,
    }


search_results = st.session_state.get("search_results")

if search_results:
    st.subheader("Search results")
    st.write(f"For: {search_results['query']}")

    for rank, result in enumerate(search_results["results"], start=1):
        passage = passages.get(result["passage_id"])

        if passage is None:
            st.warning("Search data changed. Please search again.")
            continue

        st.divider()
        st.subheader(f"{rank}. {passage['chapter_title']}")
        st.text(passage["text"])

        source_url = (
            "https://www.gutenberg.org/cache/epub/1232/"
            f"pg1232-images.html#{passage['chapter_id']}"
        )
        st.link_button("Open source chapter", source_url)

        with st.expander("Read surrounding context"):
            chapter = chapters[passage["chapter_id"]]
            paragraphs = chapter["paragraphs"]

            index = next(
                index
                for index, paragraph in enumerate(paragraphs)
                if paragraph["paragraph_id"] == passage["paragraph_id"]
            )

            start = max(0, index - 1)
            stop = min(len(paragraphs), index + 2)

            for position in range(start, stop):
                paragraph = paragraphs[position]
                label = (
                    "Parent paragraph"
                    if position == index
                    else "Neighboring paragraph"
                )

                st.caption(
                    f"{label} · {paragraph['content_type']}"
                )
                st.text(paragraph["text"])

        with st.expander("Search details"):
            st.write(f"Similarity: {result['score']:.3f}")
            st.code(passage["passage_id"])
            st.caption(
                "Similarity measures the model's text matching. "
                "It does not establish factual support."
            )
else:
    st.info("Enter a question to find relevant passages.")