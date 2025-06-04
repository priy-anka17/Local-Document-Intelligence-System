# 🧠 Local Document Intelligence System (OCR + RAG + Citation)

This project is a fully local, open-source **document intelligence system** that lets you upload scanned PDFs, extract text using **OCR (Tesseract)**, make them **searchable and downloadable**, and ask **questions with relevant answers and citations** using a **local RAG (Retrieval-Augmented Generation)** pipeline.

---

## ✨ Features

- 🧾 **OCR PDF Support** – Uses Tesseract to extract text from scanned PDFs.
- 🧠 **RAG-Based Question Answering** – Uses a local LLM with FAISS and SentenceTransformers to answer questions from document content.
- 🔍 **Contextual Citations** – Highlights exact source chunks used in each answer.
- 🖥️ **Fully Local Execution** – No cloud dependencies or API calls.
- 📥 **Downloadable Extracted Text** – Extracted full text is shown and can be downloaded.

---

## 📦 Tech Stack

- `Python 3.10+`
- `Streamlit` – Web UI
- `PyTesseract` – OCR
- `pdf2image` – PDF to image conversion
- `SentenceTransformers` – Embeddings for FAISS
- `FAISS` – Vector index for retrieval
- `llama-cpp-python` – Local LLM inference
- `GGUF Model` – e.g., [Mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

---

## 📸 Output Screenshot

Below is an example of the application interface and output:

![Document Intelligence Output](assets/output.jpg)

--- 
## 📝 Usage
- `Upload a scanned PDF document.`

- `The system extracts text using OCR.`

- `The full extracted text is displayed and can be downloaded.`

- `Ask a question about the document.`

- `Receive an answer with contextual citation.`
---

## ✅ Example GGUF Model Links

You can find GGUF models here:

- [TheBloke GGUF Models](https://huggingface.co/TheBloke)

---

## 🔒 Privacy & Security

This app runs fully locally:

- No cloud uploads  
- No external API calls  
- Your documents never leave your machine

---

## 📄 License

This project is licensed under the MIT License.

