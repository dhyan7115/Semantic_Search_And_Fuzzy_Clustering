import os
import sys
import pickle
import numpy as np
import faiss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import EMBEDDING_FILE
INDEX_FILE = "models/faiss_index.bin"

def load_embeddings():
    print("Loading embeddings...")
    with open(EMBEDDING_FILE, "rb") as f:
        embeddings = pickle.load(f)
    embeddings = np.array(embeddings).astype("float32")
    print("Embeddings shape:", embeddings.shape)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    print("Creating FAISS index...")
    
    # IndexFlatL2 uses L2 distance for similarity search
    index = faiss.IndexFlatL2(dimension)
    print("Adding vectors to index...")
    index.add(embeddings)
    print("Total vectors in index:", index.ntotal)
    return index

def save_index(index):
    os.makedirs("models", exist_ok=True)
    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_FILE)
    print("Index saved to:", INDEX_FILE)

def main():
    embeddings = load_embeddings()
    index = build_faiss_index(embeddings)
    save_index(index)
    print("Vector index creation complete.")

if __name__ == "__main__":
    main()