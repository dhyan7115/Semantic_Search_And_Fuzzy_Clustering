import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def preprocess_embeddings(embeddings, n_components=50, random_state=42):
    """Normalize embeddings and reduce dimensionality with PCA."""
    embeddings = normalize(embeddings)
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca

def find_best_gmm(embeddings, k_min=5, k_max=20, covariance_type="diag", random_state=42):
    """Search best GMM cluster count using silhouette score."""
    best_score = -1
    best_model = None
    best_k = None
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state
        )
        labels = gmm.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"Clusters: {k}  Silhouette Score: {score}")
        if score > best_score:
            best_score = score
            best_model = gmm
            best_k = k
    print("\nBest cluster count:", best_k)
    print("Best Silhouette Score:", best_score)
    return best_model, best_k, best_score