import os
import pickle

import streamlit as st
import faiss
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain + Google‑GenAI imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL  = "gemini-2.0-flash"
INDEX_DIR   = "faiss_index"
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.pkl")
FAISS_FILE  = os.path.join(INDEX_DIR, "index.faiss")

#
# ─── PDF → CHUNKS → EMBEDDINGS → FAISS ───────────────────────────────────────────
#
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(txt):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(txt)

def build_faiss_index(text_chunks):
    # 1) initialize embeddings with your API key and task type
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # :contentReference[oaicite:0]{index=0}

    # 2) embed all chunks
    vectors = embeddings.embed_documents(text_chunks)
    X = np.array(vectors, dtype="float32")

    # 3) create FAISS index
    dim   = X.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(X)

    # 4) persist index + chunks
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(text_chunks, f)

#
# ─── QUERY → RETRIEVE → CHAT ─────────────────────────────────────────────────────
#
def load_index_and_chunks():
    index  = faiss.read_index(FAISS_FILE)
    chunks = pickle.load(open(CHUNKS_FILE, "rb"))
    return index, chunks

def retrieve_chunks(question, top_k=3):
    # re‑use the same embedding setup for your query
    embedder = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL
)


    q_vec = np.array(embedder.embed_query(question), dtype="float32").reshape(1, -1)
    index, chunks = load_index_and_chunks()
    _, ids       = index.search(q_vec, top_k)
    return [chunks[i] for i in ids[0]]

def answer_from_chunks(chunks, question):
    context = "\n\n".join(chunks)
    prompt  = f"""
Answer the question as detailed as possible from the provided context. 
If the answer is not in the provided context, say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    # direct REST‑only chat with your API key
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_name=CHAT_MODEL)
    response = model.generate_content(prompt)
    return response.text
#
# ─── STREAMLIT UI ────────────────────────────────────────────────────────────────
#
def main():
    st.set_page_config(page_title="GemVision Text 💁")
    st.title("📄 Chat with your PDFs")

    with st.sidebar:
        st.header("Upload & Process")
        uploaded = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Build Index"):
            if not uploaded:
                st.error("Upload at least one PDF.")
            else:
                raw    = get_pdf_text(uploaded)
                chunks = chunk_text(raw)
                st.info(f"Splitting into {len(chunks)} chunks…")
                build_faiss_index(chunks)
                st.success("✅ Index built!")

    query = st.text_input("Ask a question about your PDFs:")
    if query:
        with st.spinner("Retrieving…"):
            top_chunks = retrieve_chunks(query)
        with st.spinner("Answering…"):
            ans = answer_from_chunks(top_chunks, query)
        st.subheader("Answer")
        st.write(ans)

if __name__ == "__main__":
    main()
