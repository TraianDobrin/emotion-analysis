"""Step 3: emotion vectors from story activations, cleaned against neutral text."""

import numpy as np


def neutral_components(neutral, var_frac=0.5):
    """Top principal components of neutral activations [M, D] that together explain
    `var_frac` of their variance. Returns orthonormal rows [k, D]."""
    x = neutral.astype(np.float32)
    x -= x.mean(0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    explained = np.cumsum(s**2) / np.sum(s**2)
    k = int(np.searchsorted(explained, var_frac)) + 1
    return vt[:k]


def compute_emotion_vectors(acts, labels, neutral=None, var_frac=0.5):
    """The paper's recipe, for every layer at once.

    acts [N, L, D]: one mean activation per story; labels: each story's emotion.
      1. average the stories of each emotion
      2. subtract the average over emotions (what all stories share)
      3. remove the top neutral-text directions, which carry topic/style, not emotion

    Returns (vectors [E, L, D], emotion names, number of neutral PCs removed per layer).
    """
    labels = np.asarray(labels)
    emotions = list(dict.fromkeys(labels.tolist()))
    means = np.stack([acts[labels == e].astype(np.float32).mean(0) for e in emotions])
    vectors = means - means.mean(0)
    n_removed = []
    if neutral is not None:
        for layer in range(vectors.shape[1]):
            basis = neutral_components(neutral[:, layer], var_frac)
            vectors[:, layer] -= (vectors[:, layer] @ basis.T) @ basis
            n_removed.append(len(basis))
    return vectors, emotions, n_removed


def classify(x, vectors, center):
    """Predict each row's emotion as the vector with the highest cosine similarity."""
    x = x - center
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return (x @ v.T).argmax(1)


def topic_split(topic_ids, test_frac=0.25, seed=0):
    """Train/test masks that hold out whole topics, so test stories have unseen premises."""
    topic_ids = np.asarray(topic_ids)
    topics = np.unique(topic_ids)
    test_topics = np.random.default_rng(seed).choice(topics, max(1, round(test_frac * len(topics))), replace=False)
    test = np.isin(topic_ids, test_topics)
    return ~test, test
