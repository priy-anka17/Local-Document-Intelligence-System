import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from llama_cpp import Llama
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class RAGPipeline:
    def __init__(self, model_path):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = Llama(model_path=model_path, n_ctx=4096, n_threads=4)

    def ocr_pdf(self, pdf_path, return_full_text=False):
        images = convert_from_path(pdf_path)
        chunks = []
        full_text = ""

        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            full_text += text + "\n"
            chunks.extend(self._split_text(text, i + 1))

        if return_full_text:
            return chunks, full_text.strip()
        return chunks

    def _split_text(self, text, page_num, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk,
                "page": page_num
            })
        return chunks

    def build_faiss_index(self, chunks):
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        return index, chunks, embeddings

    def query(self, question, index, chunks, k=3):
        q_embed = self.embedding_model.encode([question])
        _, I = index.search(np.array(q_embed).astype("float32"), k)

        context = [chunks[i] for i in I[0]]
        ctx_text = "\n".join([f"(Page {c['page']}): {c['text']}" for c in context])

        prompt = (
            f"You are an expert assistant. Use the following context to answer the question accurately.\n\n"
            f"Context:\n{ctx_text}\n\nQuestion: {question}\nAnswer:"
        )

        output = self.llm(prompt, max_tokens=512, stop=["\n\n"])["choices"][0]["text"].strip()
        return output, ctx_text
