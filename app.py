import streamlit as st
import fitz  # PyMuPDF
import trafilatura
import json
import streamlit.components.v1 as components
from pyvis.network import Network

# Import the shared Core Engine module
from core.engine import MemexCore

# ---------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="Memex - Enterprise Hybrid GraphRAG Engine", layout="wide")
# Custom CSS for Sleek Glassmorphism Look
st.markdown("""
    <style>
    /* Global Background Adjustments */
    .stApp {
        background-color: #0B0F19;
    }
    /* Card Styles */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00B4D8;
    }
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #9CA3AF;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00B4D8;
        border-bottom-color: #00B4D8;
    }
    </style>
""", unsafe_allow_html=True)
st.title("🧠 Memex: Enterprise Hybrid GraphRAG Engine")
st.caption("Powered by Modular Core Engine | Vector Search + Graph Traversal + Source Traceability")

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get key from aistudio.google.com")
    st.markdown("---")
    st.header("📥 Multimodal Data Ingestion")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    web_url = st.text_input("Or paste a Web Article URL:")
    process_btn = st.button("🚀 Process Ingestion via Core Engine", type="primary")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_data" not in st.session_state:
    st.session_state.graph_data = {"nodes": {}, "edges": []}

# Helper Function: Text Extraction from Uploaded Files
def extract_file_chunks(files, url):
    chunks = []
    if files:
        for file in files:
            if file.name.endswith(".pdf"):
                doc = fitz.open(stream=file.read(), filetype="pdf")
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        chunks.append({"source": file.name, "page": page_num + 1, "text": text})
            elif file.name.endswith(".txt"):
                content = file.read().decode("utf-8")
                chunks.append({"source": file.name, "page": 1, "text": content})
    if url:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded) or ""
        if text:
            chunks.append({"source": url, "page": 1, "text": text})
    return chunks

# Helper Function: Render Interactive PyVis Graph
def render_interactive_graph(graph_dict):
    net = Network(height="550px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
    
    # Add Nodes
    for node_name, meta in graph_dict["nodes"].items():
        title = f"Entity: {node_name}\nType: {meta.get('type', 'CONCEPT')}\nSources: {', '.join(meta.get('sources', []))}"
        net.add_node(node_name, label=node_name, title=title, color="#00B4D8", size=22)
        
    # Add Edges with Provenance Metadata
    for edge in graph_dict["edges"]:
        edge_title = f"Relation: {edge['relation']}\nSource Doc: {edge['source_doc']} (Pg {edge['page']})\nSnippet: {edge['snippet']}"
        net.add_edge(edge["source"], edge["target"], title=edge_title, label=edge["relation"], color="#7209B7")
        
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=130)
    net.save_graph("memex_graph.html")
    
    with open("memex_graph.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------
# 2. DATA INGESTION EXECUTION VIA CORE ENGINE
# ---------------------------------------------------------
if process_btn:
    if not api_key:
        st.error("⚠️ Please enter a Google Gemini API Key in the sidebar!")
    else:
        doc_chunks = extract_file_chunks(uploaded_files, web_url)
        
        if not doc_chunks:
            st.error("No valid text found to process. Please upload a file or enter a valid URL.")
        else:
            with st.spinner("⚙️ Initializing Memex Core Engine & Processing Data..."):
                # Instantiating the shared Core Engine
                engine = MemexCore(api_key=api_key)
                
                # Ingest & Process Chunks
                updated_graph = engine.ingest_documents(doc_chunks)
                st.session_state.graph_data = updated_graph
                st.success("✅ Hybrid Indexing Complete (Vector Store + Knowledge Graph Synced)!")

# ---------------------------------------------------------
# 3. INTERFACE TABS & VISUALIZATION
# ---------------------------------------------------------
if st.session_state.graph_data["nodes"]:
    tab1, tab2, tab3 = st.tabs(["🌐 Knowledge Graph", "💬 Hybrid GraphRAG Chat", "🔍 Source Provenance Inspector"])
    
    # Tab 1: Knowledge Graph Visualization
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            html_graph = render_interactive_graph(st.session_state.graph_data)
            components.html(html_graph, height=570)
        with col2:
            st.subheader("📊 Graph Metrics")
            st.metric("Total Nodes", len(st.session_state.graph_data["nodes"]))
            st.metric("Total Relationships", len(st.session_state.graph_data["edges"]))
            if st.button("🗑️ Reset Session Data"):
                st.session_state.graph_data = {"nodes": {}, "edges": []}
                st.session_state.chat_history = []
                st.rerun()

    # Tab 2: GraphRAG Chat System
    with tab2:
        st.subheader("💬 Hybrid Querying (Vector Search + Knowledge Graph)")
        st.caption("Answers synthesized using both ChromaDB embeddings and Graph Traversal via Core Engine.")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        if user_prompt := st.chat_input("Ask: 'What are the candidate's core technical skills?'"):
            if not api_key:
                st.error("Please enter your API Key in the sidebar.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.write(user_prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Executing GraphRAG reasoning..."):
                        engine = MemexCore(api_key=api_key)
                        answer = engine.query(user_prompt)
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Tab 3: Source Provenance Inspector
    with tab3:
        st.subheader("🔍 Metadata Traceability (Edge & Snippet Provenance)")
        st.caption("Inspect exact source documents, page numbers, and text quotes backing each graph connection.")
        
        edges_data = st.session_state.graph_data["edges"]
        if edges_data:
            for edge in edges_data:
                with st.expander(f"Relationship: ({edge['source']}) ──[{edge['relation']}]──► ({edge['target']})"):
                    st.markdown(f"**📄 Source File:** `{edge['source_doc']}` (Page {edge['page']})")
                    st.markdown(f"**💬 Source Quote Snippet:** *\"{edge['snippet']}\"*")
        else:
            st.info("No edge provenance metadata stored yet.")
else:
    st.info("👈 Ingest PDFs or Web URLs from the sidebar to build your Enterprise Hybrid Graph!")