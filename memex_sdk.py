import requests
from typing import List, Dict, Any, Optional

class MemexSDK:
    """
    Memex Python SDK for interacting with the Memex GraphRAG API.
    """
    def __init__(self, api_key: str, base_url: str = "http://127.0.0.1:8000"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def ingest(self, doc_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingests document chunks to extract knowledge graphs and create vector embeddings.
        
        doc_chunks format: [{"source": "doc.pdf", "page": 1, "text": "Content..."}]
        """
        url = f"{self.base_url}/v1/ingest"
        payload = {
            "api_key": self.api_key,
            "chunks": doc_chunks
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}

    def query(self, question: str) -> str:
        """
        Queries the Memex Knowledge Base using Hybrid GraphRAG logic.
        """
        url = f"{self.base_url}/v1/query"
        payload = {
            "api_key": self.api_key,
            "question": question
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("answer", "No response returned.")
        except requests.exceptions.RequestException as e:
            return f"SDK Error: {str(e)}"