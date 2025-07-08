# Retrieval QA Chatbot

## Summary  
Streamlit app that uploads a PDF, splits it into TF-IDF chunks, and answers questions via FastMCP + OpenAI.

## Features  
- PDF upload & text chunking  
- TF-IDF vectorization + cosine similarity  
- FastMCP tool for async query  
- GPT-4o answer generation  

## Requirements  
- Python 3.7+  
- streamlit  
- PyPDF2  
- openai  
- mcp-server-fastmcp  
- scikit-learn  

## Installation  
```bash
pip install streamlit PyPDF2 openai mcp-server-fastmcp scikit-learn
