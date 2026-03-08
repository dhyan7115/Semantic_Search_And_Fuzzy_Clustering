import numpy as np

def search_documents(index, embedding, documents, k=1):
    embedding = embedding.astype("float32")
    D, I = index.search(embedding, k)
    doc_index = int(I[0][0])
    return documents[doc_index]