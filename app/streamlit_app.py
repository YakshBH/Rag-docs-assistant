import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.pipeline import answer_question

st.set_page_config(page_title="RAG Documents Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Documents Assistant")
st.caption("Ask questions about your indexed PDFs. Answers should be grounded in the retrieved passages.")
st.info("Educational project only. For legal documents, this is not legal advice.")

question = st.text_area("Your question", placeholder="Example: What rights does this document describe?")
top_k = st.slider("Number of passages to retrieve", min_value=1, max_value=10, value=5)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching documents and generating an answer..."):
        try:
            result = answer_question(question, top_k=top_k)
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Retrieved sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.write(f"- **{source['source']}**, page **{source['page']}**")
            else:
                st.write("No source pages were retrieved.")
            with st.expander("How this works"):
                st.write(f"The system retrieved {result['retrieved_count']} passage(s), then passed them to the language model.")
        except Exception as exc:
            st.error(str(exc))
            st.caption("Check that the index has been built and your selected LLM is configured.")
