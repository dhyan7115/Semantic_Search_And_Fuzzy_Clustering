import faiss
import numpy as np

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index

def save_index(index, path):
    faiss.write_index(index, path)


def load_index(path):
    return faiss.read_index(path)