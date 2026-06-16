import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import faiss
import numpy as np

# Gemini API Key
genai.configure(api_key="API_KEY")

model = genai.GenerativeModel("gemini-2.5-flash")

st.title("📄 AI PDF Chatbot (RAG)")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    st.success(f"{len(chunks)} Chunks Created")

    # Embeddings
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(chunks)

    # FAISS
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        question_embedding = embedding_model.encode(
            [question]
        )

        distances, indices = index.search(
            np.array(question_embedding),
            3
        )

        relevant_chunks = []

        for i in indices[0]:
            relevant_chunks.append(chunks[i])

        context = "\n".join(relevant_chunks)

        prompt = f"""
        Answer using only the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = model.generate_content(prompt)

        st.subheader("AI Answer")

        st.write(response.text)