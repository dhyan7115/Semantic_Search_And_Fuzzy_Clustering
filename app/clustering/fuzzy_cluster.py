import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def preprocess_embeddings(embeddings):
    embeddings = normalize(embeddings)
    pca = PCA(n_components=50, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings, pca


def find_best_gmm(embeddings):
    best_score = -1
    best_model = None
    best_k = None
    for k in range(5, 21):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42
        )
        labels = gmm.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_model = gmm
            best_k = k
    return best_model, best_k, best_score