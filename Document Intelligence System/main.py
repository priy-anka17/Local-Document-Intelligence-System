import streamlit as st
import os
import tempfile
from utils.rag_pipeline import RAGPipeline

MODEL_PATH = "models/mistral.gguf"

st.set_page_config(page_title="🧠 Local Document Q&A", layout="wide")
st.title("📄 Document Intelligence (Local RAG + OCR)")

uploaded = st.file_uploader("Upload a scanned PDF document", type=["pdf"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded.read())
        tmp_path = tmp_file.name

    st.info("⏳ Extracting text from scanned PDF using OCR...")

    rag = RAGPipeline(model_path=MODEL_PATH)
    chunks, full_text = rag.ocr_pdf(tmp_path, return_full_text=True)
    st.success(f"✅ OCR completed. Total chunks: {len(chunks)}")

    # 📄 Display full text
    st.markdown("### 📖 Extracted Text:")
    st.text_area("Scanned Text (Searchable)", full_text, height=400)

    # 💾 Allow download
    st.download_button(
        label="📥 Download Extracted Text",
        data=full_text,
        file_name="extracted_text.txt",
        mime="text/plain"
    )

    st.info("🔍 Generating embeddings and building FAISS index...")
    index, chunk_store, _ = rag.build_faiss_index(chunks)
    st.success("✅ Index built successfully.")

    question = st.text_input("💬 Ask a question about the document:")
    if question:
        answer, citation = rag.query(question, index, chunk_store)

        st.markdown("### 🧠 Answer")
        st.markdown(f"> {answer}")

        st.markdown("### 🔍 Source (Citation)")
        st.code(citation, language="text")
