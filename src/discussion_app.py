import streamlit as st
from openai import OpenAIError
from sentence_transformers import SentenceTransformer

from answer_question import generate_answer
from semantic_search import load_search_data

import json
from pathlib import Path



st.set_page_config(
    page_title="Dialogue with Idols",
    layout="wide",
)


@st.cache_resource
def load_embedder(model_name):
    return SentenceTransformer(model_name, device="cpu")


st.title("Dialogue with Idols")
st.caption("Discuss The Prince · W. K. Marriott translation")
st.caption(
    "Your request and selected source paragraphs are sent to OpenAI "
    "when you generate an answer."
)

try:
    search_data = load_search_data()
    embedder = load_embedder(search_data[-1])
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the book: {error}")
    st.stop()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "data" / "runs"

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

        saved_runs[path.stem] = saved

    except (OSError, ValueError, TypeError):
        skipped_files += 1


with st.sidebar:
    st.header("Saved answers")

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
        st.caption("Generated answers will appear here.")

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
            "A ruler should always prioritize being loved. "
            "Challenge my position."
        ),
        max_chars=8000,
    )

    research_query = st.text_input(
        "Search the book for",
        placeholder="A ruler should always prioritize being loved.",
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
        "For follow-ups, use a self-contained research query "
        "that names the subject you want to investigate."
    )

    submitted = st.form_submit_button("Generate cited answer")


if submitted:
    # Clear the previous answer so a failure cannot leave stale output.
    st.session_state.pop("discussion_run", None)

    if not question.strip():
        st.warning("Enter a question or opinion.")
        st.stop()

    try:
        with st.spinner("Finding evidence and drafting an answer..."):
            run = generate_answer(
                question,
                research_query,
                embedder,
                search_data,
                parent_run=current_run if follow_up else None
            )

        st.session_state["discussion_run"] = run

    except OpenAIError:
        st.error(
            "The model request failed. Check your connection, "
            "API configuration, and account limits."
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"Could not complete the discussion: {error}")


run = st.session_state.get("discussion_run")

if run:
    answer = run["answer"]
    evidence = {
        source["source_id"]: source
        for source in run["evidence"]
    }

    st.divider()
    st.subheader("Discussion")
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
                st.text(source["text"])

                chapter_id = source_id.split(":")[1]
                source_url = (
                    "https://www.gutenberg.org/cache/epub/1232/"
                    f"pg1232-images.html#{chapter_id}"
                )
                st.link_button("Open original chapter", source_url)

    st.markdown("**A question to consider**")
    st.text(answer["follow_up_question"])

    with st.expander("Run details"):
        st.write(f"Run ID: {run['run_id']}")
        st.write(f"Model: {run['generation_model_returned']}")
        st.write(f"Research query: {run['research_query']}")
        st.json(run["usage"])