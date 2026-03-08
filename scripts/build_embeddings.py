import os
import pickle
from sklearn.datasets import fetch_20newsgroups
from app.embeddings.preprocessing import preprocess_documents
from app.embeddings.embedder import Embedder
from app.config import DOCUMENT_FILE, EMBEDDING_FILE
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_dataset():
    print("Loading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )
    documents = dataset.data
    print("Total documents:", len(documents))
    return documents

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():

    # Load dataset
    documents = load_dataset()
    
    # Preprocess
    print("Preprocessing documents...")
    clean_docs = preprocess_documents(documents)
    
    # Generate embeddings
    print("Generating embeddings...")
    embedder = Embedder()
    embeddings = embedder.encode_documents(clean_docs)
    print("Embedding shape:", embeddings.shape)
    
    # Save processed data
    print("Saving processed documents...")
    save_pickle(clean_docs, DOCUMENT_FILE)
    print("Saving embeddings...")
    save_pickle(embeddings, EMBEDDING_FILE)
    print("Embedding pipeline complete.")

if __name__ == "__main__":
    main()