"""Shared visual styling for the reading and discussion pages."""
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERO_IMAGE = PROJECT_ROOT / "assets" / "reading-room.png"


def apply_theme():
    st.markdown(
        """
        <style>
        :root {
          --ink: #30241d;
          --muted-ink: #71604f;
          --plum: #684c78;
          --plum-dark: #4f3b60;
          --paper: #fff8ec;
          --line: #e4d2b9;
          --sage: #e8e4d5;
        }
        .stApp,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"] {
          font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
        }
        .stApp {
          background:
            radial-gradient(ellipse at 85% 2%, rgba(255, 247, 228, .86), transparent 34rem),
            repeating-linear-gradient(88deg, rgba(102, 67, 39, .025) 0 1px, transparent 1px 15px),
            linear-gradient(135deg, #e5cfb0 0%, #f1e2ca 34%, #e8d5b8 68%, #ddc39f 100%);
          background-attachment: fixed;
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stAppViewContainer"] .main .block-container {
          max-width: 1180px;
          padding-top: 2.4rem;
          padding-bottom: 5rem;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
        h1 { font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif; font-weight: 500; }
        h2, h3 { font-weight: 600; }
        [data-testid="stCaptionContainer"] { color: var(--muted-ink); }
        .eyebrow {
          color: var(--plum);
          font-size: .72rem;
          font-weight: 700;
          letter-spacing: .18em;
          text-transform: uppercase;
          margin: .2rem 0 .45rem;
        }
        .hero-copy {
          color: var(--muted-ink);
          font-size: 1.06rem;
          max-width: 44rem;
          line-height: 1.7;
          margin-top: -.5rem;
          margin-bottom: 1.6rem;
        }
        [data-testid="stImage"] img {
          border: 1px solid var(--line);
          border-radius: 20px;
          box-shadow: 0 14px 38px rgba(45, 35, 52, .09);
        }
        [data-testid="stForm"] {
          background: rgba(255, 249, 239, .88);
          border: 1px solid var(--line);
          border-radius: 20px;
          padding: 1.25rem 1.35rem 1rem;
          box-shadow: 0 12px 36px rgba(75, 48, 26, .09);
        }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
          background: #fffdf7;
          border-color: #d8c5a9;
          border-radius: 12px;
          font-family: "Segoe UI", system-ui, sans-serif;
        }
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stTextInput"] input:focus {
          border-color: var(--plum);
          box-shadow: 0 0 0 .15rem rgba(104, 76, 120, .14);
        }
        .stButton button, [data-testid="stFormSubmitButton"] button {
          border-radius: 999px;
          min-height: 2.75rem;
          height: auto;
          font-weight: 650;
          padding-inline: 1.15rem;
          transition: transform .15s ease, box-shadow .15s ease;
          font-family: "Segoe UI", system-ui, sans-serif;
        }
        .stButton button, .stButton button p { white-space: normal; }
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        .stButton button[kind="primary"] {
          background: var(--plum);
          border-color: var(--plum);
          color: white;
        }
        [data-testid="stFormSubmitButton"] button:hover,
        .stButton button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(53, 40, 65, .12); }
        [data-testid="stExpander"] {
          background: rgba(255, 249, 239, .82);
          border: 1px solid var(--line);
          border-radius: 14px;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #d9c09f, #ead9bd 58%, #e2cfb1);
          border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted-ink); }
        [data-testid="stAlert"] { border-radius: 14px; }
        hr { border-color: var(--line); margin: 1.6rem 0; }
        @media (max-width: 700px) {
          [data-testid="stAppViewContainer"] .main .block-container { padding: 1.5rem 1rem 3rem; }
          h1 { font-size: 2.35rem; }
          [data-testid="stForm"] { padding: 1rem; border-radius: 16px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(kicker, title, description):
    copy, visual = st.columns([0.84, 1.36], gap="large", vertical_alignment="center")
    with copy:
        st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
        st.title(title)
        st.markdown(f'<div class="hero-copy">{description}</div>', unsafe_allow_html=True)
    with visual:
        st.image(
            str(HERO_IMAGE),
            caption="Machiavelli and Sun Tzu meet across an open book.",
            width="stretch",
        )
