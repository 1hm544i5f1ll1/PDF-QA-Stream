import streamlit as st
import PyPDF2
import asyncio
import os
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client using updated interface
client = OpenAI(api_key=os.environ.get("OpenApiKey"))

# Globals
pdf_chunks = []
vectorizer = None
chunk_embeddings = None

mcp = FastMCP("pdf_docs")

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@mcp.tool()
async def answer_query(query: str) -> str:
    if not pdf_chunks:
        return "No PDF loaded."
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, chunk_embeddings)
    best_idx = sims.argmax()
    context = pdf_chunks[best_idx]
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = await asyncio.to_thread(lambda: client.responses.create(
        model="gpt-4o",
        instructions="You are a helpful assistant.",
        input=prompt
    ))
    return response.output_text.strip()

st.title("Retrieval QA Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text
    pdf_chunks = split_text(full_text)
    vectorizer = TfidfVectorizer().fit(pdf_chunks)
    chunk_embeddings = vectorizer.transform(pdf_chunks)
    st.success("PDF loaded and indexed.")

user_query = st.text_input("Ask a question about the document:")
if user_query:
    answer = asyncio.run(answer_query(user_query))
    st.write("**Answer:**")
    st.write(answer)
