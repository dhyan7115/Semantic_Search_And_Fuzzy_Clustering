Cluster-Aware Semantic Search Engine with Fuzzy Clustering and Intelligent Caching
Project Overview

This project implements a semantic search system that retrieves documents based on their meaning rather than simple keyword matching. The system uses sentence embeddings, fuzzy clustering, vector similarity search, and semantic caching to efficiently process user queries.

Traditional search systems rely heavily on exact keywords, which may fail when the user query uses different words for the same concept. This project solves that problem by representing text using vector embeddings and retrieving semantically similar documents.

The system is exposed through a FastAPI backend service that allows users to submit queries and receive relevant documents along with cluster probability distributions.

Dataset

The project uses the 20 Newsgroups dataset, which contains approximately 18,000 documents across 20 different topics such as politics, sports, science, and computer technology.

The dataset is automatically downloaded using the fetch_20newsgroups() function from scikit-learn.

System Architecture

The system pipeline is as follows:

User Query
   ↓
Sentence Embedding Generation
   ↓
Normalization
   ↓
PCA Dimensionality Reduction
   ↓
Fuzzy Clustering (Gaussian Mixture Model)
   ↓
Cluster-Aware Semantic Cache
   ↓
FAISS Vector Similarity Search
   ↓
Return Result + Cluster Probabilities
Key Features

Semantic search based on vector embeddings

Fuzzy clustering with probabilistic topic membership

Fast vector similarity search using FAISS

Semantic caching to reduce repeated computations

Cluster-aware cache lookup for improved efficiency

REST API using FastAPI

Cluster probability distribution for interpretability

Project Structure
TradeMarkia/
│
├── app/
│   ├── embeddings/
│   │   ├── embedder.py
│   │   └── preprocessing.py
│   │
│   ├── clustering/
│   │   └── fuzzy_cluster.py
│   │
│   ├── cache/
│   │   └── semantic_cache.py
│   │
│   ├── services/
│   │   └── query_services.py
│   │
│   ├── vector_store/
│   │   └── faiss_store.py
│   │
│   └── main.py
│
├── scripts/
│   ├── build_embeddings.py
│   ├── build_vector_index.py
│   ├── run_clustering.py
│   └── evaluate_system.py
│
├── models/
├── data/
├── requirements.txt
└── README.md
Major Components
Embedding Generation

Documents are converted into vector embeddings using the SentenceTransformer model:

all-MiniLM-L6-v2

These embeddings represent the semantic meaning of text in a 384-dimensional vector space.

Vector Database

The embeddings are stored in a FAISS vector index, which allows fast nearest-neighbor search.

FAISS enables the system to quickly find documents that are semantically similar to a query.

Fuzzy Clustering

The system performs clustering using a Gaussian Mixture Model.

Unlike traditional clustering methods, fuzzy clustering assigns probabilities across clusters instead of a single cluster label.

Example:

Cluster 2 → 0.12
Cluster 4 → 0.51
Cluster 7 → 0.18

This reflects the fact that documents can belong to multiple topics.

Semantic Cache

A semantic cache stores previously processed queries along with their embeddings and results.

When a new query arrives, the system checks if a similar query already exists in the cache using cosine similarity.

If the similarity exceeds a threshold, the cached result is returned immediately.

Cluster-Aware Cache Optimization

To improve cache efficiency, cached queries are stored along with their cluster information.

When a new query is processed, the system predicts the dominant cluster and only compares it with cached queries belonging to the same cluster.

This reduces the number of comparisons and improves performance for large caches.

API Endpoints

The system exposes the following FastAPI endpoints:

POST /query

Processes user queries and returns the most relevant document.

Example request:

{
  "query": "space shuttle launch"
}

Example response:

{
  "dominant_cluster": 4,
  "cluster_distribution": {...},
  "result": "document text..."
}
GET /cache/stats

Returns statistics about the semantic cache.

Example response:

{
  "total_entries": 10,
  "hit_count": 4,
  "miss_count": 6,
  "hit_rate": 0.4
}
DELETE /cache

Clears the semantic cache.

Installation

Clone the repository:

git clone <repository_url>
cd TradeMarkia

Create a virtual environment:

python -m venv venv

Activate the environment:

Linux / Mac:

source venv/bin/activate

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Project

Run the preprocessing and training scripts:

python -m scripts.build_embeddings
python -m scripts.build_vector_index
python -m scripts.run_clustering

Start the FastAPI server:

uvicorn app.main:app --reload

Open the API interface:

http://127.0.0.1:8000/docs
Evaluation

The system is evaluated using:

Silhouette Score
Measures clustering quality.

Retrieval Accuracy
Measures how often the system retrieves the correct document for a query.

Technologies Used

Python

FastAPI

SentenceTransformers

FAISS

Scikit-learn

NumPy

Uvicorn

Conclusion

This project demonstrates how modern NLP techniques such as embeddings, clustering, vector databases, and intelligent caching can be combined to build an efficient semantic search engine. The system provides both accurate semantic retrieval and improved computational efficiency through cluster-aware caching.
