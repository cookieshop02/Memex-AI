import json
import asyncio
from difflib import SequenceMatcher
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ChromaDB Vector Store Import
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------
# SCHEMAS FOR STRUCTURED EXTRACTION
# ---------------------------------------------------------
class Entity(BaseModel):
    name: str = Field(description="Normalized entity name, e.g., 'Python', 'Aman Sharma'")
    type: str = Field(description="Category like PERSON, TECHNOLOGY, CONCEPT, ORGANIZATION")

class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship label in UPPERCASE, e.g., CREATED, DEPENDS_ON, USED_FOR")
    snippet: str = Field(description="Short text quote supporting this relationship")

class KnowledgeGraphSchema(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]

# ---------------------------------------------------------
# MEMEX CORE ENGINE CLASS
# ---------------------------------------------------------
class MemexCore:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        
        # Initialize Vector DB
        self.chroma_client = chromadb.Client()
        emb_fn = embedding_functions.DefaultEmbeddingFunction()
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="memex_chunks", 
            embedding_function=emb_fn
        )
        
        # In-memory graph structure
        self.graph = {"nodes": {}, "edges": []}

    def _get_canonical_name(self, new_name: str, existing_nodes: set, threshold: float = 0.82) -> str:
        clean_new = new_name.strip()
        for existing in existing_nodes:
            ratio = SequenceMatcher(None, clean_new.lower(), existing.lower()).ratio()
            if ratio >= threshold or (clean_new.lower() in existing.lower() and len(clean_new) > 3):
                return existing
        return clean_new

    async def _async_extract_chunk(self, chunk_info: dict):
        prompt = f"""
        Extract key entities and relationships from the text below.
        
        STRICT RULES:
        1. NEVER use generic nouns like 'Candidate', 'Author', 'Document', 'User', or 'Person'.
        2. Always extract the REAL SPECIFIC NAME (e.g. use actual person's name instead of 'Candidate').
        
        Source File: {chunk_info['source']} | Page: {chunk_info['page']}
        Text:
        {chunk_info['text']}
        """
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=KnowledgeGraphSchema,
                    temperature=0.1,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error extracting chunk: {e}")
            return {"entities": [], "relationships": []}

    def ingest_documents(self, doc_chunks: list[dict]) -> dict:
        """In-gests document chunks into ChromaDB and extracts Knowledge Graph."""
        if not doc_chunks:
            return self.graph

        # Step 1: Add Chunks to ChromaDB Vector Store
        texts = [c["text"] for c in doc_chunks]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in doc_chunks]
        ids = [f"id_{i}_{c['source']}_p{c['page']}" for i, c in enumerate(doc_chunks)]
        
        try:
            self.vector_collection.add(documents=texts, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"Vector DB addition warning: {e}")

        # Step 2: Async Graph Extraction (Top 6 Chunks max for speed)
        target_chunks = doc_chunks[:6]
        
        async def run_batch():
            tasks = [self._async_extract_chunk(c) for c in target_chunks]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_batch())

        # Step 3: Populate and Deduplicate Graph
        existing_nodes = set(self.graph["nodes"].keys())
        
        for idx, res in enumerate(results):
            chunk_meta = target_chunks[idx]
            
            for entity in res.get("entities", []):
                canon_name = self._get_canonical_name(entity["name"], existing_nodes)
                existing_nodes.add(canon_name)
                if canon_name not in self.graph["nodes"]:
                    self.graph["nodes"][canon_name] = {"type": entity.get("type", "CONCEPT"), "sources": set()}
                self.graph["nodes"][canon_name]["sources"].add(chunk_meta["source"])

            for rel in res.get("relationships", []):
                src = self._get_canonical_name(rel["source"], existing_nodes)
                tgt = get_canonical_name = self._get_canonical_name(rel["target"], existing_nodes)
                existing_nodes.add(src)
                existing_nodes.add(tgt)
                
                self.graph["edges"].append({
                    "source": src,
                    "target": tgt,
                    "relation": rel["relation"],
                    "snippet": rel.get("snippet", "No snippet quote available."),
                    "source_doc": chunk_meta["source"],
                    "page": chunk_meta["page"]
                })

        return self.graph

    def query(self, user_query: str) -> str:
        """Executes Hybrid GraphRAG search over both Vector DB & Graph Triplets."""
        # 1. Vector Search Context with Safety Check
        retrieved_chunks = []
        try:
            if self.vector_collection.count() > 0:
                vector_results = self.vector_collection.query(query_texts=[user_query], n_results=3)
                if vector_results and "documents" in vector_results and vector_results["documents"]:
                    for doc_list in vector_results["documents"]:
                        retrieved_chunks.extend(doc_list)
        except Exception as e:
            print(f"Vector search skip/warning: {e}")
            
        vector_context = "\n---\n".join(retrieved_chunks) if retrieved_chunks else "No vector context retrieved."

        # 2. Graph Context
        triplets_summary = []
        for edge in self.graph["edges"]:
            triplets_summary.append(
                f"({edge['source']}) ──[{edge['relation']}]──► ({edge['target']}) | Source: {edge['source_doc']} (Pg {edge['page']})"
            )
        graph_context = "\n".join(triplets_summary) if triplets_summary else "No graph context retrieved."

        # 3. Hybrid Synthesis Prompt
        prompt = f"""
        You are Memex AI Core Engine. Answer the user question using BOTH the Vector Context and Knowledge Graph Context.
        Always cite the Source File/Page if available.

        --- VECTOR CONTEXT ---
        {vector_context}

        --- GRAPH CONTEXT ---
        {graph_context}

        Question: {user_query}
        """
        response = self.client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text