import numpy as np


def dominant_cluster(probabilities):
    """Return dominant cluster index."""
    return int(np.argmax(probabilities))


def get_cluster_distribution(probabilities):
    """Return cluster distribution sorted descending."""
    clusters = list(enumerate(probabilities))
    clusters.sort(key=lambda x: x[1], reverse=True)
    return clusters