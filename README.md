# 🇹🇭 Thailand FAQ Chatbot

This is a chatbot built using **LangChain**, **Streamlit**, and **RAG (Retrieval-Augmented Generation)**. Ask questions about moving to or living in Thailand, and get answers sourced directly from text files like guides and FAQs.

## 🔧 How it Works

- Loads `.txt` documents from the `thai_docs/` folder
- Uses `langchain` to embed and index the text into a vectorstore
- Uses `OpenAI` GPT model to answer user questions based on the most relevant chunks of data
- Deployed with `Streamlit` as a real-time Q&A interface

## 🚀 How to Run It

```bash
streamlit run app.py
