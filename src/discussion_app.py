import streamlit as st
from openai import OpenAIError
from sentence_transformers import SentenceTransformer

from answer_question import generate_answer
from semantic_search import load_search_data
from library import run_book_id, saved_source_link
from library_ui import library_controls, book_details, cached_embedder, render_passages
from retrieval import retrieve

import json
import os
from pathlib import Path



st.set_page_config(
    page_title="Dialogue with Idols",
    layout="wide",
)
DEMO_MODE = os.getenv('DIALOGUE_DEMO_MODE') == '1'
DEMO_ANSWER_LIMIT = 3


@st.cache_resource
def load_embedder(model_name, language=None):
    return cached_embedder(model_name, language)


st.title("Dialogue with Idols")
selected_book = library_controls()
if st.session_state.get('active_book') != selected_book:
    st.session_state.pop('discussion_run', None)
    st.session_state.pop('retrieved_preview', None)
    st.session_state['active_book'] = selected_book
st.caption(
    "Your request and selected source paragraphs are sent to OpenAI "
    "when you generate an answer."
)

try:
    search_data = load_search_data(selected_book)
    embedder = load_embedder(search_data[-1], search_data.book.get('language'))
    book = search_data.book
    st.caption(f"{book['title']} · {book['author']} · Translation: {book.get('translator') or 'not specified'}")
    book_details(book)
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the book: {error}")
    st.stop()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / ("demo_runs" if DEMO_MODE else "data/runs")

saved_runs = {}
skipped_files = 0

for path in RUN_DIR.glob("*.json"):
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))

        # Basic compatibility checks for our saved-run format.
        required_fields = {
            "run_id",
            "created_at",
            "question",
            "research_query",
            "generation_model_returned",
            "answer",
            "evidence",
            "usage",
        }

        if not required_fields.issubset(saved):
            raise ValueError("Incomplete saved run")

        if run_book_id(saved) == selected_book:
            saved_runs[path.stem] = saved

    except (OSError, ValueError, TypeError):
        skipped_files += 1


with st.sidebar:
    st.header("This session" if DEMO_MODE else "Saved answers")

    if skipped_files:
        st.caption(
            f"Skipped {skipped_files} unreadable or incompatible files."
        )

    if saved_runs:
        ordered_ids = sorted(
            saved_runs,
            key=lambda run_id: saved_runs[run_id]["created_at"],
            reverse=True,
        )

        def history_label(run_id):
            saved = saved_runs[run_id]
            question_preview = " ".join(saved["question"].split())

            return (
                f"{saved['created_at'][:10]} · "
                f"{question_preview[:65]} · {run_id[:6]}"
            )

        selected_id = st.selectbox(
            "Choose an answer",
            options=ordered_ids,
            format_func=history_label,
        )

        if st.button("Open saved answer"):
            st.session_state["discussion_run"] = saved_runs[selected_id]

    else:
        st.caption("Your current answer stays in this browser session." if DEMO_MODE else "Generated answers will appear here.")

    if DEMO_MODE:
        st.caption(f"Demo answers remaining this session: {max(0, DEMO_ANSWER_LIMIT - st.session_state.get('demo_answer_count', 0))}")
    else:
        st.caption(
            "Opening a saved answer reads the stored result. "
            "It does not generate another answer."
        )

with st.form("discussion"):
    current_run = st.session_state.get("discussion_run")

    if current_run:
        st.caption(
            "Available context: "
            + current_run["question"][:150]
        )
    question = st.text_area(
        "Your question or opinion",
        placeholder=(
            "What trade-offs does this book describe? "
            "Challenge my position."
        ),
        max_chars=8000,
    )

    research_query = st.text_input(
        "Search the book for",
        placeholder="Name the subject you want to investigate.",
        help=(
            "Optional. Describe the subject to research, "
            "without instructions about how to answer."
        ),
    )

    follow_up = st.checkbox(
        "Continue from the currently open answer",
        value=False,
        disabled=current_run is None,
    )

    st.caption(
        "For follow-ups, the previous research topic is included automatically. "
        "You can enter a specific research query to focus the search."
    )

    preview_submitted = st.form_submit_button("Find passages")
    submitted = st.form_submit_button("Generate cited answer")


if preview_submitted:
    try:
        query = research_query.strip() or question.strip()
        if follow_up and current_run and not research_query.strip():
            query = current_run.get('research_query', current_run['question']) + ' ' + query
        with st.spinner('Finding relevant passages...'):
            results = retrieve(search_data, query, embedder)
        st.session_state['retrieved_preview'] = {'book_id': selected_book, 'query': query, 'results': results}
    except (ValueError, RuntimeError) as error:
        st.session_state.pop('retrieved_preview', None)
        st.warning(str(error))

preview = st.session_state.get('retrieved_preview')
if preview and preview['book_id'] == selected_book:
    st.subheader('Retrieved passages')
    st.caption(f"Search: {preview['query']}")
    render_passages(search_data, preview['results'])


if submitted:
    # Clear the previous answer so a failure cannot leave stale output.
    st.session_state.pop("discussion_run", None)

    if not question.strip():
        st.warning("Enter a question or opinion.")
        st.stop()

    if DEMO_MODE and st.session_state.get('demo_answer_count', 0) >= DEMO_ANSWER_LIMIT:
        st.warning('This demo session has used its three generated answers. You can still find passages.')
        st.stop()

    try:
        with st.spinner("Finding evidence and drafting an answer..."):
            run = generate_answer(
                question,
                research_query,
                embedder,
                search_data,
                parent_run=current_run if follow_up else None,
                **({'save_run': False} if DEMO_MODE else {}),
            )

        st.session_state["discussion_run"] = run
        if DEMO_MODE:
            st.session_state['demo_answer_count'] = st.session_state.get('demo_answer_count', 0) + 1

    except OpenAIError:
        st.error(
            "The model request failed. Check your connection, "
            "API configuration, and account limits."
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"Could not complete the discussion: {error}")

def get_conversation_chain(latest_run, available_runs):
    chain = []
    visited = set()
    current = latest_run
    warning = None

    while current is not None:
        run_id = current["run_id"]

        if run_id in visited:
            warning = "A circular parent link was found."
            break

        visited.add(run_id)
        chain.append(current)

        parent_id = current.get("parent_run_id")

        if not parent_id:
            break

        parent = available_runs.get(parent_id)

        if parent is None:
            warning = (
                "An earlier turn is unavailable. "
                "Showing the turns that could be loaded."
            )
            break

        current = parent

    # We collected newest first; display oldest first.
    chain.reverse()
    return chain, warning


def render_run(run):
    answer = run["answer"]
    evidence = {
        source["source_id"]: source
        for source in run["evidence"]
    }

    st.divider()
    st.caption(f"Saved: {run['created_at']}")
    metadata = run.get('document_metadata', {})
    if metadata:
        st.caption(f"{metadata['title']} · {metadata['author']}")
    st.write("Your request:")
    st.text(run["question"])

    st.caption(
        "Generated answer. Citation IDs were checked; "
        "claim support has not been independently verified."
    )
    st.caption(
        f"Model-assessed evidence status: "
        f"{answer['evidence_status']}"
    )
    st.text(answer["explanation"])

    for number, claim in enumerate(answer["claims"], start=1):
        st.markdown(f"**Argument {number} · {claim['evidence_type']}**")
        st.text(claim["text"])

        for source_id in claim["source_ids"]:
            source = evidence[source_id]

            with st.expander(
                f"Evidence for argument {number}: {source_id}"
            ):
                st.caption(source["chapter"])
                st.caption(f"Content type: {source['content_type']}")
                if source.get('excerpted'):
                    st.caption(f"Excerpt from paragraph characters {source['char_start']}–{source['char_end']}")
                if source.get('source_locator'):
                    st.caption(source['source_locator'])
                st.text(source["text"])

                source_url = saved_source_link(run, source)
                if source_url:
                    st.link_button(
                        f"Open source · {run['run_id'][:6]} · argument {number} · {source_id}",
                        source_url,
                    )

    st.markdown("**A question to consider**")
    st.text(answer["follow_up_question"])

    with st.expander("Run details"):
        st.write(f"Run ID: {run['run_id']}")
        st.write(f"Model: {run['generation_model_returned']}")
        st.write(f"Research query: {run['research_query']}")
        st.json(run["usage"])

active_run = st.session_state.get("discussion_run")

if active_run:
    available_runs = {
        saved["run_id"]: saved
        for saved in saved_runs.values()
    }

    # A newly generated run may not be in the sidebar's file list yet.
    available_runs[active_run["run_id"]] = active_run

    conversation, warning = get_conversation_chain(
        active_run,
        available_runs,
    )

    st.header("Conversation")
    st.caption(f"{len(conversation)} saved turn(s)")

    if warning:
        st.warning(warning)

    for turn_number, turn in enumerate(conversation, start=1):
        st.subheader(f"Turn {turn_number}")
        render_run(turn)
