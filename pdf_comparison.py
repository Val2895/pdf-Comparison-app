import os
import streamlit as st
import pdfplumber
import re
from collections import defaultdict
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import GroqClient

# Disable Streamlit's file-watcher to suppress PyTorch warnings
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable is not set.")
    st.stop()
client = GroqClient(api_key=GROQ_API_KEY)

# Lazy-load embedding model
def get_embedding_model():
    if "embedding_model" not in st.session_state:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            st.error("Missing dependency 'sentence-transformers'. Please add it to requirements.txt.")
            return None
        try:
            st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            return None
    return st.session_state.embedding_model

# Helper functions
def extract_text_with_pdfplumber(file_obj):
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def preprocess_text(text: str) -> str:
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def extract_tables_with_pdfplumber(file_obj):
    tables = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables() or [])
    return tables


def parse_color_quantities(tables):
    color_quantities = defaultdict(int)
    for table in tables:
        for row in table:
            if len(row) >= 6:
                try:
                    color = row[1].strip().upper()
                    qty = int(row[-1].replace(",", "").strip())
                    color_quantities[color] += qty
                except Exception:
                    continue
    return dict(color_quantities)


def find_relevant_chunks(query_embedding, index, chunks, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

class PDFChatBot:
    def __init__(self):
        self.pdf1_chunks = []
        self.pdf2_chunks = []
        self.pdf1_index = None
        self.pdf2_index = None
        self.pdf1_color_quantities = {}
        self.pdf2_color_quantities = {}

    def process_pdfs(self, pdf1_file, pdf2_file):
        model = get_embedding_model()
        if model is None:
            return "Embedding model not available. Check errors above."
        try:
            # 1) Extract text
            raw1 = extract_text_with_pdfplumber(pdf1_file)
            raw2 = extract_text_with_pdfplumber(pdf2_file)
            text1 = preprocess_text(raw1)
            text2 = preprocess_text(raw2)

            if not text1:
                return "PDF 1 contains no extractable text. Please upload a valid PDF."
            if not text2:
                return "PDF 2 contains no extractable text. Please upload a valid PDF."

            # 2) Extract and parse tables
            tbls1 = extract_tables_with_pdfplumber(pdf1_file)
            self.pdf1_color_quantities = parse_color_quantities(tbls1)
            tbls2 = extract_tables_with_pdfplumber(pdf2_file)
            self.pdf2_color_quantities = parse_color_quantities(tbls2)

            # 3) Split into chunks
            self.pdf1_chunks = split_into_chunks(text1)
            self.pdf2_chunks = split_into_chunks(text2)

            # 4) Embeddings & FAISS
            emb1 = model.encode(self.pdf1_chunks)
            emb2 = model.encode(self.pdf2_chunks)
            d = emb1.shape[1]
            self.pdf1_index = faiss.IndexFlatL2(d)
            self.pdf1_index.add(np.array(emb1))
            self.pdf2_index = faiss.IndexFlatL2(d)
            self.pdf2_index.add(np.array(emb2))

            return "PDFs processed successfully!"
        except Exception as e:
            return f"An error occurred while processing the PDFs: {e}"

    def answer_question(self, question: str) -> str:
        if not self.pdf1_index or not self.pdf2_index:
            return "Please process the PDFs first by uploading them and clicking 'Process PDFs'."

        if "compare" in question.lower() and "color" in question.lower():
            if not (self.pdf1_color_quantities or self.pdf2_color_quantities):
                return "No color data found. Check that your PDFs have color/quantity tables."
            lines = []
            for color in sorted(set(self.pdf1_color_quantities) | set(self.pdf2_color_quantities)):
                q1 = self.pdf1_color_quantities.get(color, 0)
                q2 = self.pdf2_color_quantities.get(color, 0)
                lines.append(f"{color}: PDF 1 = {q1}, PDF 2 = {q2}")
            return "\n".join(lines)

        # Semantic search + Groq
        model = get_embedding_model()
        if model is None:
            return "Embedding model not available. Cannot perform semantic search."
        q_emb = model.encode(question)
        rel1 = find_relevant_chunks(q_emb, self.pdf1_index, self.pdf1_chunks)
        rel2 = find_relevant_chunks(q_emb, self.pdf2_index, self.pdf2_chunks)
        context = (
            "Relevant content from PDF 1:\n" + "\n".join(rel1) + "\n\n"
            "Relevant content from PDF 2:\n" + "\n".join(rel2)
        )
        prompt = (
            "You are an expert analyst. Analyze the context to answer the question.\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            "If insufficient, say 'Insufficient information to answer the question.'"
        )
        try:
            response = client.completions.create(
                model="groq-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq API error: {e}"

# Streamlit app
if "pdf_chatbot" not in st.session_state:
    st.session_state.pdf_chatbot = PDFChatBot()
pdf_chatbot = st.session_state.pdf_chatbot

st.title("PDF Chatbot")

pdf1 = st.file_uploader("Upload PDF 1", type=["pdf"])
pdf2 = st.file_uploader("Upload PDF 2", type=["pdf"])

if pdf1 and pdf2 and st.button("Process PDFs"):
    with st.spinner("Processing PDFs..."):
        msg = pdf_chatbot.process_pdfs(pdf1, pdf2)
        if msg.startswith("An error"):
            st.error(msg)
        else:
            st.success(msg)

st.header("Ask Questions About the PDFs")
question = st.text_input("Enter your question:")
if question and st.button("Get Answer"):
    with st.spinner("Generating answer..."):
        st.write(pdf_chatbot.answer_question(question))
