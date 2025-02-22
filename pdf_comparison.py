import os
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import re
from collections import defaultdict

# Suppress PyTorch warnings caused by Streamlit's file watcher
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"

# Configure Gemini API
genai.configure(api_key="Use API KEY HERE")
gemini_model = genai.GenerativeModel('gemini-pro')


# Helper functions
def extract_text_with_pdfplumber(file_path):
    """Extract text from a PDF file using pdfplumber."""
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for i, page in enumerate(pdf.pages):
            extracted_text = page.extract_text()
            if extracted_text:  # Ensure text is not None
                text += extracted_text
            else:
                print(f"Warning: No text extracted from page {i + 1}")
    return text


def preprocess_text(text):
    """Preprocess text by removing headers/footers and cleaning up newlines."""
    if not text:  # Handle None or empty strings
        return ""
    text = re.sub(r"Page \d+", "", text)  # Remove "Page X" patterns
    text = re.sub(r"\n+", " ", text)      # Replace multiple newlines with a space
    return text.strip()


def split_into_chunks(text, chunk_size=500, chunk_overlap=100):
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def generate_embeddings(chunks):
    """Generate embeddings for text chunks using SentenceTransformer."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks)


def store_embeddings_in_faiss(embeddings):
    """Store embeddings in a FAISS index for similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def find_relevant_chunks(query_embedding, index, chunks, top_k=5):
    """Find the most relevant chunks based on query embedding."""
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in indices[0]]


def extract_tables_with_pdfplumber(file_path):
    """Extract tables from a PDF file using pdfplumber."""
    with pdfplumber.open(file_path) as pdf:
        tables = []
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            tables.extend(extracted_tables)
    return tables


def parse_color_quantities(tables):
    """Parse tables to extract color-wise quantities."""
    color_quantities = defaultdict(int)
    for table in tables:
        for row in table:
            # Example row: ['Sample Code', 'Color', 'Size', 'Pattern/Length', 'Size%', 'Order(pcs)']
            if len(row) >= 6:
                try:
                    color = row[1].strip().upper()  # Assuming color is in the second column
                    quantity = int(row[-1].replace(",", "").strip())  # Assuming quantity is in the last column
                    color_quantities[color] += quantity
                except (ValueError, IndexError):
                    continue
    return dict(color_quantities)


class PDFChatBot:
    def __init__(self):
        self.pdf1_chunks = []
        self.pdf2_chunks = []
        self.pdf1_index = None
        self.pdf2_index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pdf1_color_quantities = {}
        self.pdf2_color_quantities = {}

    def process_pdfs(self, pdf1_file, pdf2_file):
        """Process uploaded PDFs: extract text, split into chunks, generate embeddings, and store in FAISS."""
        try:
            # Step 1: Extract text
            pdf1_text = preprocess_text(extract_text_with_pdfplumber(pdf1_file))
            pdf2_text = preprocess_text(extract_text_with_pdfplumber(pdf2_file))

            if not pdf1_text.strip():
                return "PDF 1 contains no extractable text. Please upload a valid PDF."
            if not pdf2_text.strip():
                return "PDF 2 contains no extractable text. Please upload a valid PDF."

            print("Extracted text from PDF 1:", pdf1_text[:100])  # Print first 100 characters
            print("Extracted text from PDF 2:", pdf2_text[:100])

            # Step 2: Split into chunks
            self.pdf1_chunks = split_into_chunks(pdf1_text)
            self.pdf2_chunks = split_into_chunks(pdf2_text)
            print("Chunks from PDF 1:", self.pdf1_chunks[:5])  # Print first 5 chunks
            print("Chunks from PDF 2:", self.pdf2_chunks[:5])

            # Step 3: Generate embeddings
            pdf1_embeddings = generate_embeddings(self.pdf1_chunks)
            pdf2_embeddings = generate_embeddings(self.pdf2_chunks)
            print("Embeddings shape for PDF 1:", pdf1_embeddings.shape)
            print("Embeddings shape for PDF 2:", pdf2_embeddings.shape)

            # Step 4: Store embeddings in FAISS
            self.pdf1_index = store_embeddings_in_faiss(pdf1_embeddings)
            self.pdf2_index = store_embeddings_in_faiss(pdf2_embeddings)
            print("FAISS index for PDF 1 created successfully.")
            print("FAISS index for PDF 2 created successfully.")

            return "PDFs processed successfully!"
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            return f"An error occurred while processing the PDFs: {e}"

    def answer_question(self, question):
        """Answer a question based on the processed PDFs."""
        print("Debug: PDF1 Index:", self.pdf1_index)
        print("Debug: PDF2 Index:", self.pdf2_index)
        if self.pdf1_index is None or self.pdf2_index is None:
            return "Please process the PDFs first by uploading them and clicking 'Process PDFs'."

        # Handle color-wise quantity comparison queries
        if "compare" in question.lower() and "color" in question.lower():
            response = []
            for color in set(self.pdf1_color_quantities.keys()).union(self.pdf2_color_quantities.keys()):
                pdf1_qty = self.pdf1_color_quantities.get(color, 0)
                pdf2_qty = self.pdf2_color_quantities.get(color, 0)
                response.append(f"{color}: PDF 1 = {pdf1_qty}, PDF 2 = {pdf2_qty}")
            return "\n".join(response)

        # Default behavior for other questions
        query_embedding = self.model.encode(question)
        pdf1_relevant_chunks = find_relevant_chunks(query_embedding, self.pdf1_index, self.pdf1_chunks)
        pdf2_relevant_chunks = find_relevant_chunks(query_embedding, self.pdf2_index, self.pdf2_chunks)

        context = (
                f"Relevant content from PDF 1:\n" + "\n".join(pdf1_relevant_chunks) +
                f"\n\nRelevant content from PDF 2:\n" + "\n".join(pdf2_relevant_chunks)
        )

        response = gemini_model.generate_content(
            f"You are an expert analyst. Carefully analyze the following context to answer the question.\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"If the answer cannot be determined from the context, respond with 'Insufficient information to answer the question.'"
        )
        return response.text


# Initialize the chatbot in session_state if it doesn't exist
if "pdf_chatbot" not in st.session_state:
    st.session_state.pdf_chatbot = PDFChatBot()

# Access the chatbot from session_state
pdf_chatbot = st.session_state.pdf_chatbot

# Streamlit app
st.title("PDF Chatbot")

# File uploaders
pdf1_file = st.file_uploader("Upload PDF 1", type=["pdf"])
pdf2_file = st.file_uploader("Upload PDF 2", type=["pdf"])

if pdf1_file and pdf2_file:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            status = pdf_chatbot.process_pdfs(pdf1_file, pdf2_file)
            st.success(status)

# Chat interface
st.header("Ask Questions About the PDFs")
question = st.text_input("Enter your question:")
if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer = pdf_chatbot.answer_question(question)
        st.write("Answer:", answer)
