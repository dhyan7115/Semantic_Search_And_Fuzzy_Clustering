import pickle
import numpy as np
import faiss

from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app.embeddings.embedder import Embedder


# -----------------------------
# File paths
# -----------------------------

DOCUMENT_FILE = "data/processed/documents.pkl"
FAISS_INDEX_FILE = "models/faiss_index.bin"
CLUSTER_MODEL_FILE = "models/gmm_cluster_model.pkl"
PCA_MODEL_FILE = "models/pca_model.pkl"
MEMBERSHIP_FILE = "models/document_cluster_memberships.pkl"


# -----------------------------
# Load resources at startup
# -----------------------------

print("Loading documents...")
with open(DOCUMENT_FILE, "rb") as f:
    documents = pickle.load(f)

print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_FILE)

print("Loading clustering model...")
with open(CLUSTER_MODEL_FILE, "rb") as f:
    cluster_model = pickle.load(f)

print("Loading PCA model...")
with open(PCA_MODEL_FILE, "rb") as f:
    pca_model = pickle.load(f)

print("Loading cluster memberships...")
with open(MEMBERSHIP_FILE, "rb") as f:
    memberships = pickle.load(f)

# dominant cluster of each document
doc_clusters = np.argmax(memberships, axis=1)

print("Loading embedding model...")
embedder = Embedder()


# -----------------------------
# Semantic cache
# -----------------------------

cache = []
hit_count = 0
miss_count = 0

SIMILARITY_THRESHOLD = 0.85


# -----------------------------
# FastAPI initialization
# -----------------------------

app = FastAPI(title="Cluster-Aware Semantic Search with Fuzzy Clustering")


class QueryRequest(BaseModel):
    query: str


# -----------------------------
# Query endpoint
# -----------------------------

@app.post("/query")
def query_system(request: QueryRequest):

    global hit_count, miss_count

    query = request.query

    # -----------------------------
    # Generate embedding
    # -----------------------------

    query_embedding = embedder.encode_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # normalize embedding
    query_embedding = normalize(query_embedding)

    # PCA transform
    query_embedding_reduced = pca_model.transform(query_embedding)

    # -----------------------------
    # Cluster probabilities
    # -----------------------------

    cluster_probs = cluster_model.predict_proba(query_embedding_reduced)

    dominant_cluster = int(np.argmax(cluster_probs))

    cluster_distribution = {
        f"cluster_{i}": float(p)
        for i, p in enumerate(cluster_probs[0])
    }

    # -----------------------------
    # Find documents in this cluster
    # -----------------------------

    cluster_documents = [
        f"Document_{i}"
        for i, c in enumerate(doc_clusters)
        if c == dominant_cluster
    ]

    # limit number returned
    cluster_documents = cluster_documents[:20]

    # -----------------------------
    # Cluster-aware cache lookup
    # -----------------------------

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

    # -----------------------------
    # Cache hit
    # -----------------------------

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

    # -----------------------------
    # Cache miss → search FAISS
    # -----------------------------

    miss_count += 1

    query_embedding_float = query_embedding.astype("float32")

    D, I = faiss_index.search(query_embedding_float, k=1)

    doc_index = int(I[0][0])

    result = documents[doc_index]

    # store in cache
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
        "similarity_score": 0,
        "result": result,
        "dominant_cluster": dominant_cluster,
        "cluster_distribution": cluster_distribution,
        "documents_in_cluster": cluster_documents
    }


# -----------------------------
# Cache statistics endpoint
# -----------------------------

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


# -----------------------------
# Clear cache endpoint
# -----------------------------

@app.delete("/cache")
def clear_cache():

    global cache, hit_count, miss_count

    cache = []
    hit_count = 0
    miss_count = 0

    return {"message": "Cache cleared"}