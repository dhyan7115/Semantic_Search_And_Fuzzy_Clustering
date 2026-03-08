from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def encode_documents(self, docs):
        return self.model.encode(docs, show_progress_bar=True)

    def encode_query(self, query):
        return self.model.encode([query])[0]