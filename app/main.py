import pickle
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from app.embeddings.embedder import Embedder

DOCUMENT_FILE = "data/processed/documents.pkl"
FAISS_INDEX_FILE = "models/faiss_index.bin"
CLUSTER_MODEL_FILE = "models/gmm_cluster_model.pkl"
PCA_MODEL_FILE = "models/pca_model.pkl"
MEMBERSHIP_FILE = "models/document_cluster_memberships.pkl"

with open(DOCUMENT_FILE, "rb") as f:
    documents = pickle.load(f)
faiss_index = faiss.read_index(FAISS_INDEX_FILE)
with open(CLUSTER_MODEL_FILE, "rb") as f:
    cluster_model = pickle.load(f)
with open(PCA_MODEL_FILE, "rb") as f:
    pca_model = pickle.load(f)
with open(MEMBERSHIP_FILE, "rb") as f:
    memberships = pickle.load(f)
doc_clusters = np.argmax(memberships, axis=1)
embedder = Embedder()
cache = []
hit_count = 0
miss_count = 0
SIMILARITY_THRESHOLD = 0.85
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_system(request: QueryRequest):
    global hit_count, miss_count
    query = request.query
    query_embedding = embedder.encode_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    query_embedding = normalize(query_embedding)
    query_embedding_reduced = pca_model.transform(query_embedding)
    cluster_probs = cluster_model.predict_proba(query_embedding_reduced)
    dominant_cluster = int(np.argmax(cluster_probs))
    cluster_distribution = {
        f"cluster_{i}": float(p)
        for i, p in enumerate(cluster_probs[0])
    }

    cluster_documents = [
        f"Document_{i}"
        for i, c in enumerate(doc_clusters)
        if c == dominant_cluster
    ]
    cluster_documents = cluster_documents[:20]
    best_similarity = 0
    matched_query = None
    cached_result = None

    for entry in cache:
        if entry["cluster"] != dominant_cluster:
            continue
        sim = cosine_similarity(
            query_embedding,
            entry["embedding"].reshape(1, -1)
        )[0][0]

        if sim > best_similarity:
            best_similarity = sim
            matched_query = entry["query"]
            cached_result = entry["result"]

    if best_similarity >= SIMILARITY_THRESHOLD:
        hit_count += 1
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": matched_query,
            "similarity_score": float(best_similarity),
            "result": cached_result,
            "dominant_cluster": dominant_cluster,
            "cluster_distribution": cluster_distribution,
            "documents_in_cluster": cluster_documents
        }

    miss_count += 1
    query_embedding_float = query_embedding.astype("float32")
    D, I = faiss_index.search(query_embedding_float, k=1)
    doc_index = int(I[0][0])
    result = documents[doc_index]
    cache.append({
        "query": query,
        "embedding": query_embedding.flatten(),
        "result": result,
        "cluster": dominant_cluster
    })
    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "result": result,
        "dominant_cluster": dominant_cluster,
        "cluster_distribution": cluster_distribution,
        "documents_in_cluster": cluster_documents
    }

@app.get("/cache/stats")
def cache_stats():
    total = hit_count + miss_count
    hit_rate = hit_count / total if total > 0 else 0
    return {
        "total_entries": len(cache),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": hit_rate
    }

@app.delete("/cache")
def clear_cache():
    global cache, hit_count, miss_count
    cache = []
    hit_count = 0
    miss_count = 0
    return {"message": "Cache cleared"}