import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.index.vector_store import ensure_index_built
from src.config import TOP_K
from src.pipeline import answer_question

ensure_index_built()

st.set_page_config(page_title="RAG Documents Assistant", page_icon="📚", layout="wide")
SUGGESTED_QUESTIONS = (
    "What is AI risk management?",
    "What are the security risks of large language models?",
    "How does the DPDP Act protect personal data?",
    "What are the four functions of the NIST AI Risk Management Framework?",
    "How can organizations make AI systems more trustworthy?",
)

KNOWLEDGE_BASE = (
    (
        "Digital Personal Data Protection Act",
        "Digital-Personal-Data-Protection-Act-2023.pdf",
        "India's framework for protecting digital personal data and defining responsibilities for data fiduciaries.",
        "⚖️",
    ),
    (
        "NIST AI Risk Management Framework",
        "NIST.AI.100-1.pdf",
        "A practical framework for managing AI risks and building trustworthy, responsible systems.",
        "🛡️",
    ),
    (
        "NIST Generative AI Profile",
        "NIST.AI.600-1.pdf",
        "Guidance for identifying and managing the distinctive risks of generative AI.",
        "✨",
    ),
    (
        "OWASP Top 10 for LLM Applications",
        "OWASP-Top-10-for-LLMs-v2025.pdf",
        "A security reference covering the most important risks in large language model applications.",
        "🔐",
    ),
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 42%);
    }
    .block-container {
        max-width: 1180px;
        padding: 3rem 2rem 4rem;
    }
    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid #dbeafe;
        border-radius: 24px;
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
        box-shadow: 0 12px 30px rgba(30, 64, 175, 0.08);
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: #2563eb;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero h1 {
        color: #0f172a;
        font-size: clamp(2rem, 5vw, 3.2rem);
        line-height: 1.05;
        margin: 0 0 0.7rem;
    }
    .hero p {
        color: #475569;
        font-size: 1.05rem;
        margin: 0;
        max-width: 720px;
    }
    .section-heading {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 2rem 0 0.3rem;
    }
    .section-caption {
        color: #64748b;
        margin-bottom: 0.9rem;
    }
    .knowledge-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .knowledge-card {
        min-height: 164px;
        padding: 1.1rem;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
    }
    .knowledge-icon {
        font-size: 1.35rem;
        margin-bottom: 0.55rem;
    }
    .knowledge-title {
        color: #1e293b;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.45rem;
    }
    .knowledge-description {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .knowledge-file {
        color: #94a3b8;
        font-size: 0.72rem;
        line-height: 1.35;
        margin-top: 0.65rem;
        overflow-wrap: anywhere;
    }
    div.stButton > button {
        border: 1px solid #dbeafe;
        border-radius: 12px;
        color: #1e3a8a;
        background: #f8fbff;
        min-height: 3rem;
        text-align: left;
        transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
    }
    div.stButton > button:hover {
        border-color: #60a5fa;
        box-shadow: 0 5px 14px rgba(37, 99, 235, 0.12);
        transform: translateY(-1px);
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1.5rem 1rem 3rem;
        }
        .hero {
            padding: 1.35rem;
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Grounded AI knowledge assistant</div>
        <h1>Explore your AI governance library.</h1>
        <p>Ask questions across trusted policy, risk-management, and security documents. Every answer is grounded in the indexed source pages.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info("Educational project only. For legal documents, this is not legal advice.")

st.markdown('<div class="section-heading">Try asking</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Choose a prompt to send it through the same assistant workflow.</div>',
    unsafe_allow_html=True,
)

selected_question = None
question_columns = st.columns(2)
for index, suggested_question in enumerate(SUGGESTED_QUESTIONS):
    with question_columns[index % 2]:
        if st.button(suggested_question, key=f"suggested-{index}", use_container_width=True):
            selected_question = suggested_question

if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if selected_question:
    st.session_state.question_input = selected_question

st.markdown('<div class="section-heading">Knowledge Base</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Four indexed sources available for grounded answers.</div>',
    unsafe_allow_html=True,
)
knowledge_cards = "".join(
    f"""
    <div class="knowledge-card">
        <div class="knowledge-icon">{icon}</div>
        <div class="knowledge-title">{title}</div>
        <div class="knowledge-description">{description}</div>
        <div class="knowledge-file">{filename}</div>
    </div>
    """
    for title, filename, description, icon in KNOWLEDGE_BASE
)
st.markdown(f'<div class="knowledge-grid">{knowledge_cards}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Ask the assistant</div>', unsafe_allow_html=True)
question = st.text_area(
    "Your question",
    key="question_input",
    placeholder="Example: What rights does this document describe?",
)
top_k = st.slider("Number of passages to retrieve", min_value=1, max_value=10, value=TOP_K)
ask_clicked = st.button("Ask", type="primary", disabled=not question.strip(), use_container_width=True)

if ask_clicked or selected_question:
    question_started = time.perf_counter()
    with st.spinner("Searching documents and generating an answer..."):
        try:
            result = answer_question(question, top_k=top_k)
            elapsed_ms = (time.perf_counter() - question_started) * 1000
            print(f"[PERF] Streamlit request total       {elapsed_ms:8.1f} ms passages={result['retrieved_count']}")
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Retrieved sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.write(f"- **{source['source']}**, page **{source['page']}**")
            else:
                st.write("No source pages were retrieved.")
            with st.expander("How this works"):
                st.write(
                    f"The system retrieved {result['retrieved_count']} passage(s), "
                    "then passed them to the language model."
                )
        except Exception as exc:
            st.error(str(exc))
            st.caption("Check that the index has been built and your selected LLM is configured.")
