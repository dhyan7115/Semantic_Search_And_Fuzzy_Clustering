from sentence_transformers import SentenceTransformer

class Embedder:

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode_documents(self, docs):
        return self.model.encode(docs)

    def encode_query(self, query):
        return self.model.encode([query])[0]