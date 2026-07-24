from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from core.engine import MemexCore

app = FastAPI(
    title="Memex AI API",
    description="REST API for Enterprise Hybrid GraphRAG Engine",
    version="1.0.0"
)

# Global Engine Store to persist state across requests
engines: Dict[str, MemexCore] = {}

def get_engine(api_key: str) -> MemexCore:
    if api_key not in engines:
        engines[api_key] = MemexCore(api_key=api_key)
    return engines[api_key]

# ---------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------
class ChunkModel(BaseModel):
    source: str
    page: int
    text: str

class IngestRequest(BaseModel):
    api_key: str
    chunks: List[ChunkModel]

class QueryRequest(BaseModel):
    api_key: str
    question: str

# ---------------------------------------------------------
# REST API ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "online", "message": "Memex GraphRAG Core Engine API is running!"}

@app.post("/v1/ingest", response_model=Dict[str, Any])
def ingest_documents(req: IngestRequest):
    """
    Ingest document text chunks into ChromaDB Vector Store & Extract Knowledge Graph.
    """
    if not req.api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required.")
    
    try:
        engine = get_engine(req.api_key)
        chunks_data = [chunk.dict() for chunk in req.chunks]
        graph_result = engine.ingest_documents(chunks_data)
        
        # Format nodes dictionary for clean JSON serialization
        formatted_nodes = {}
        for node_name, meta in graph_result["nodes"].items():
            formatted_nodes[node_name] = {
                "type": meta.get("type", "CONCEPT"),
                "sources": list(meta.get("sources", []))
            }
            
        return {
            "status": "success",
            "graph": {
                "nodes": formatted_nodes,
                "edges": graph_result["edges"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/v1/query")
def query_graphrag(req: QueryRequest):
    """
    Query the Knowledge Base using Hybrid GraphRAG (Vector Embeddings + Knowledge Graph).
    """
    if not req.api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required.")
    
    try:
        engine = get_engine(req.api_key)
        answer = engine.query(req.question)
        return {"status": "success", "question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")