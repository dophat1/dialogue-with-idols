"""Shared visual styling for the reading and discussion pages."""
import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        :root {
          --ink: #272137;
          --muted-ink: #696276;
          --plum: #684c78;
          --plum-dark: #4f3b60;
          --paper: #fbf9f5;
          --line: #e9e3db;
          --sage: #e9eee8;
        }
        .stApp {
          background:
            radial-gradient(ellipse at 84% 0%, rgba(223, 215, 231, .48), transparent 34rem),
            linear-gradient(180deg, #f7f4ef 0%, #fbf9f5 52%, #f7f4ef 100%);
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stAppViewContainer"] .main .block-container {
          max-width: 1080px;
          padding-top: 2.4rem;
          padding-bottom: 5rem;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
        h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 500; }
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
        [data-testid="stForm"] {
          background: rgba(255, 255, 255, .78);
          border: 1px solid var(--line);
          border-radius: 20px;
          padding: 1.25rem 1.35rem 1rem;
          box-shadow: 0 12px 36px rgba(45, 35, 52, .055);
        }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
          background: #fffefa;
          border-color: #ded6e4;
          border-radius: 12px;
        }
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stTextInput"] input:focus {
          border-color: var(--plum);
          box-shadow: 0 0 0 .15rem rgba(104, 76, 120, .14);
        }
        .stButton button, [data-testid="stFormSubmitButton"] button {
          border-radius: 999px;
          min-height: 2.75rem;
          font-weight: 650;
          padding-inline: 1.15rem;
          transition: transform .15s ease, box-shadow .15s ease;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        .stButton button[kind="primary"] {
          background: var(--plum);
          border-color: var(--plum);
          color: white;
        }
        [data-testid="stFormSubmitButton"] button:hover,
        .stButton button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(53, 40, 65, .12); }
        [data-testid="stExpander"] {
          background: rgba(255,255,255,.7);
          border: 1px solid var(--line);
          border-radius: 14px;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #f0ece8, #f5f2ed);
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
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="hero-copy">{description}</div>', unsafe_allow_html=True)
