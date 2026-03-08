from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticCache:

    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):

        best_similarity = 0
        best_entry = None

        for entry in self.cache:
            sim = cosine_similarity(
                query_embedding,
                entry["embedding"].reshape(1, -1)
            )[0][0]

            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_similarity >= self.threshold:
            self.hit_count += 1
            return True, best_entry, best_similarity

        self.miss_count += 1
        return False, None, best_similarity

    def add(self, query, embedding, result):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0