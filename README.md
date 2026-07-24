# 🧠 Memex AI — Enterprise Hybrid GraphRAG Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Memex AI** is an advanced Knowledge Engine powered by **Hybrid GraphRAG**. It combines **Vector Search (ChromaDB)** with **Dynamic Knowledge Graph Extraction (Google Gemini 2.5 Flash)** to deliver deep, structured, and factual document understanding with interactive graph visualizations.

---

## ✨ Key Features

- 📑 **Multi-Source Document Ingestion**: Process PDFs, plain text files, or scrape web content via URLs.
- 🕸️ **Automated Knowledge Graph Extraction**: Automatically extracts entities, types, and canonical relationships using Gemini 2.5.
- ⚡ **Hybrid GraphRAG Retrieval**: Blends semantic vector search with knowledge graph triplets for high-accuracy answers with source citations.
- 🎨 **Interactive Graph Visualizer**: Interactive network graphs rendered in real-time using PyVis inside the Streamlit UI.
- 🔌 **REST API Ready**: Complete FastAPI REST backend for programmatic access and external integrations.

---

## 🛠️ Tech Stack

- **LLM Engine**: Google Gemini API (`gemini-2.5-flash`)
- **Vector Database**: ChromaDB
- **Frontend / UI**: Streamlit & PyVis
- **REST API Backend**: FastAPI & Uvicorn
- **Graph Processing**: NetworkX
- **Text Processing**: PyMuPDF (`fitz`), Trafilatura

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/cookieshop02/Memex-AI.git](https://github.com/cookieshop02/Memex-AI.git)
cd Memex-AI


