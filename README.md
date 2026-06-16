# AI-PDF-Chatbot-RAG
AI-powered PDF Chatbot using RAG, LangChain, FAISS, Gemini API, and Streamlit.

## Overview

AI PDF Chatbot is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. The system retrieves relevant information from the uploaded document and uses Google Gemini AI to generate context-aware answers.

## Features

* Upload PDF documents
* Extract text from PDFs
* Document chunking using LangChain
* Embeddings generation using Sentence Transformers
* Vector search using FAISS
* Context-aware question answering using Gemini API
* Interactive Streamlit interface

## Tech Stack

* Python
* Streamlit
* Google Gemini API
* LangChain
* FAISS
* Sentence Transformers
* PyPDF

## Project Workflow

PDF Upload
→ Text Extraction
→ Chunking
→ Embeddings
→ FAISS Vector Store
→ Question Input
→ Context Retrieval
→ Gemini Response

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Skills Demonstrated

* Generative AI
* Retrieval-Augmented Generation (RAG)
* LangChain
* Vector Databases
* Embeddings
* Prompt Engineering
* Python Development
