import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.clustering.fuzzy_cluster import preprocess_embeddings, find_best_gmm

EMBEDDING_FILE = "data/processed/embeddings.pkl"
MODEL_FILE = "models/gmm_cluster_model.pkl"
MEMBERSHIP_FILE = "models/document_cluster_memberships.pkl"
PCA_FILE = "models/pca_model.pkl"

def load_embeddings():
    with open(EMBEDDING_FILE, "rb") as f:
        embeddings = pickle.load(f)
    return np.array(embeddings)

def save_results(model, memberships, pca):
    os.makedirs("models", exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(MEMBERSHIP_FILE, "wb") as f:
        pickle.dump(memberships, f)
    with open(PCA_FILE, "wb") as f:
        pickle.dump(pca, f)

def main():
    embeddings = load_embeddings()
    reduced, pca = preprocess_embeddings(embeddings)
    model, best_k, best_score = find_best_gmm(reduced)
    memberships = model.predict_proba(reduced)
    save_results(model, memberships, pca)

if __name__ == "__main__":
    main()