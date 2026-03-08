import os
import sys
import pickle
import numpy as np
import faiss
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups
# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.embeddings.embedder import Embedder

# Load embeddings
with open("data/processed/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
embeddings = np.array(embeddings)

# same preprocessing used in clustering
embeddings = normalize(embeddings)
with open("models/pca_model.pkl", "rb") as f:
    pca = pickle.load(f)
embeddings_reduced = pca.transform(embeddings)

# Load clustering model
with open("models/gmm_cluster_model.pkl", "rb") as f:
    gmm = pickle.load(f)
labels = gmm.predict(embeddings_reduced)
sil_score = silhouette_score(embeddings_reduced, labels)
print("\nClustering Evaluation")
print("---------------------")
print("Silhouette Score:", sil_score)

# Retrieval Accuracy
index = faiss.read_index("models/faiss_index.bin")
dataset = fetch_20newsgroups(subset="all")
targets = dataset.target
embedder = Embedder()
correct = 0
total = 100
for i in range(total):
    query_text = dataset.data[i]
    true_label = targets[i]
    query_embedding = embedder.encode_query(query_text)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype("float32")
    D, I = index.search(query_embedding, k=1)
    retrieved_index = int(I[0][0])
    retrieved_label = targets[retrieved_index]
    if retrieved_label == true_label:
        correct += 1
accuracy = correct / total
print("\nRetrieval Evaluation")
print("---------------------")
print("Test Queries:", total)
print("Correct Retrievals:", correct)
print("Accuracy:", accuracy)