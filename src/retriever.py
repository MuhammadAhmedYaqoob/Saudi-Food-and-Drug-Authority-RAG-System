import pickle
import faiss
import numpy as np
import pathlib
import os
from typing import List, Dict
import openai
import networkx as nx
from .config import OPENAI_API_KEY, OPENAI_EMBED_MODEL, EMBED_DIM, FAISS_DIR, GRAPH_DIR
from .loader import load_multiple_pdfs, is_arabic

# Initialize OpenAI client with API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def _embed_openai(texts: List[str]) -> np.ndarray:
    """Generate embeddings using OpenAI API"""
    if not texts:
        return np.array([])
    
    # Process in batches to avoid API limits
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"[ERROR] Embedding generation failed: {e}")
            # Return zeros for failed batches
            zero_embeddings = [np.zeros(EMBED_DIM) for _ in range(len(batch))]
            all_embeddings.extend(zero_embeddings)
    
    return np.array(all_embeddings)

# ---------- Build / load semantic index -------------------------------------
def build_faiss() -> pathlib.Path:
    """Build FAISS index from document chunks"""
    # Ensure directory exists
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load document chunks
    docs = load_multiple_pdfs()
    
    if not docs:
        print("[ERROR] No documents to index")
        return None
    
    # Create embeddings
    print(f"[INFO] Generating embeddings for {len(docs)} document chunks")
    vecs = _embed_openai([m["text"] for m in docs])
    
    # Create and save FAISS index
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs.astype("float32"))
    
    fp = FAISS_DIR / "semantic.index"
    faiss.write_index(index, str(fp))
    
    # Save document mapping
    doc_map_path = FAISS_DIR / "doc_mapping.pkl"
    with open(doc_map_path, 'wb') as f:
        pickle.dump(docs, f)
    
    print(f"[INFO] FAISS index built and saved to {fp}")
    return fp

def load_graph():
    """Load the knowledge graph from disk"""
    try:
        graph_path = GRAPH_DIR / "sfda_knowledge_graph.pkl"
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load graph: {e}")
    return None

def extract_query_entities(query):
    """Extract potential entities from the query"""
    # Simple extraction based on capitalization and Arabic characters
    query_tokens = query.split()
    entities = []
    
    for token in query_tokens:
        # Clean token
        clean_token = token.strip('.,;:()[]{}"\'"').lower()
        
        # Skip short tokens
        if len(clean_token) < 3:
            continue
            
        # Check for Arabic or capitalized words (potential entities)
        if is_arabic(clean_token) or (clean_token[0].isupper() and clean_token.isalpha()):
            entities.append(clean_token)
    
    return entities

def get_graph_enhanced_results(query, initial_results, k=5):
    """
    Use graph-based heuristics to enhance retrieval results:
    1. Find documents that are connected to initial results via shared entities
    2. Find documents that contain entities mentioned in the query
    3. Use document-document relationships
    """
    G = load_graph()
    if not G:
        return initial_results
        
    enhanced_results = initial_results.copy()
    initial_doc_ids = {doc["id"] for doc in initial_results}
    candidates = {}
    
    # Extract entities from query
    query_entities = extract_query_entities(query)
    query_entity_nodes = [f"ent::{e}" for e in query_entities]
    
    # 1. Find related documents through the graph
    for doc in initial_results:
        doc_node = f"doc::{doc['id']}"
        
        if not G.has_node(doc_node):
            continue
            
        # Get entities in this document
        doc_entities = []
        for _, entity, _ in G.out_edges(doc_node, data=True):
            if entity.startswith("ent::"):
                doc_entities.append(entity)
        
        # Find other documents with these entities
        for entity in doc_entities:
            for node, attrs in G.nodes(data=True):
                if attrs.get("type") == "document" and node != doc_node:
                    # Check if this document contains the entity
                    if G.has_edge(node, entity):
                        # Calculate a score based on shared entities
                        doc_id = node[5:]  # Remove "doc::" prefix
                        if doc_id not in initial_doc_ids:
                            if doc_id not in candidates:
                                candidates[doc_id] = {"score": 0, "shared_entities": set()}
                            candidates[doc_id]["score"] += 1
                            candidates[doc_id]["shared_entities"].add(entity)
    
    # 2. Find documents with query entities
    for entity_node in query_entity_nodes:
        if G.has_node(entity_node):
            for node, attrs in G.nodes(data=True):
                if attrs.get("type") == "document":
                    # Check if document contains this query entity
                    if G.has_edge(node, entity_node):
                        doc_id = node[5:]  # Remove "doc::" prefix
                        if doc_id not in initial_doc_ids:
                            if doc_id not in candidates:
                                candidates[doc_id] = {"score": 0, "shared_entities": set()}
                            # Higher score for direct query entity match
                            candidates[doc_id]["score"] += 2
                            candidates[doc_id]["shared_entities"].add(entity_node)
    
    # 3. Use document-document relationships
    for doc in initial_results:
        doc_node = f"doc::{doc['id']}"
        
        if not G.has_node(doc_node):
            continue
            
        # Check direct document-document relationships
        for _, related_doc, edge_data in G.out_edges(doc_node, data=True):
            if related_doc.startswith("doc::") and edge_data.get("role") == "related_via":
                doc_id = related_doc[5:]  # Remove "doc::" prefix
                if doc_id not in initial_doc_ids:
                    if doc_id not in candidates:
                        candidates[doc_id] = {"score": 0, "shared_entities": set()}
                    candidates[doc_id]["score"] += 1.5
                    # Add the shared entity if available
                    if "entity" in edge_data:
                        candidates[doc_id]["shared_entities"].add(edge_data["entity"])
    
    # Load document mapping to get full documents
    try:
        with open(FAISS_DIR / "doc_mapping.pkl", 'rb') as f:
            all_docs = pickle.load(f)
            all_docs_map = {doc["id"]: doc for doc in all_docs}
    except Exception as e:
        print(f"[WARNING] Could not load document mapping: {e}")
        return initial_results
    
    # Sort candidates by score and add top ones to results
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
    
    for doc_id, data in sorted_candidates:
        if doc_id in all_docs_map:
            # Copy the document and add graph score
            doc = all_docs_map[doc_id].copy()
            doc["graph_score"] = data["score"]
            doc["graph_match_reason"] = f"Shares {len(data['shared_entities'])} entities with initial results"
            enhanced_results.append(doc)
            
            # Stop when we've added enough additional results
            if len(enhanced_results) >= 2*k:
                break
    
    # Re-rank results based on combined semantic + graph scores
    for doc in enhanced_results:
        if "score" not in doc:
            doc["score"] = 0
        
        # Add graph score if available
        if "graph_score" in doc:
            # Normalize to be comparable to semantic score
            doc["combined_score"] = doc["score"] + 0.2 * doc["graph_score"]
        else:
            doc["combined_score"] = doc["score"]
    
    # Sort by combined score
    enhanced_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    
    # Limit to k results
    return enhanced_results[:k]

def query_semantic(q: str, k: int = 5) -> List[Dict]:
    """
    Query using semantic search enhanced with graph-based heuristics:
    1. Perform semantic search via FAISS
    2. Enhance results using graph-based related document discovery
    """
    try:
        # Check query language
        query_lang = "ar" if is_arabic(q) else "en"
        print(f"[INFO] Query detected as {query_lang}")
        
        # Load the FAISS index
        idx = faiss.read_index(str(FAISS_DIR / "semantic.index"))
        
        # Load the document mapping
        doc_map_path = FAISS_DIR / "doc_mapping.pkl"
        with open(doc_map_path, 'rb') as f:
            docs = pickle.load(f)
        
        # Generate query embedding
        qvec = _embed_openai([q])[0].astype("float32")
        
        # Search in the index, get more results than requested for reranking
        D, I = idx.search(qvec.reshape(1, -1), k*2)
        
        # Get the initial matched documents
        initial_results = []
        for i, idx in enumerate(I[0]):
            if idx < len(docs):
                doc = docs[idx].copy()
                doc["score"] = float(D[0][i])
                initial_results.append(doc)
        
        # Apply graph-based enhancements
        enhanced_results = get_graph_enhanced_results(q, initial_results, k)
        
        return enhanced_results
    
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {e}")
        return []

def shortest_entity_path(ent_a: str, ent_b: str, max_len=4):
    """Find shortest path between two entities in the knowledge graph"""
    import networkx as nx
    
    G = load_graph()
    if not G:
        return []
    
    # Handle Arabic/English translation if needed
    ent_a_node = f"ent::{ent_a.lower()}"
    ent_b_node = f"ent::{ent_b.lower()}"
    
    try:
        return nx.shortest_path(G, ent_a_node, ent_b_node, cutoff=max_len)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Try alternative representations (translation pairs)
        alternatives_a = [node for node, attrs in G.nodes(data=True) 
                         if attrs.get("type") == "entity" and 
                            G.has_edge(node, ent_a_node) and
                            G.edges[node, ent_a_node]["role"] == "translates_to"]
        
        alternatives_b = [node for node, attrs in G.nodes(data=True) 
                         if attrs.get("type") == "entity" and 
                            G.has_edge(node, ent_b_node) and
                            G.edges[node, ent_b_node]["role"] == "translates_to"]
        
        # Try all combinations of alternatives
        for alt_a in [ent_a_node] + alternatives_a:
            for alt_b in [ent_b_node] + alternatives_b:
                try:
                    return nx.shortest_path(G, alt_a, alt_b, cutoff=max_len)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                    
        return []