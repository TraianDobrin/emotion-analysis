"""Building emotion vectors from story activations (difference of means),
denoising them against neutral text, and a logistic-regression probe baseline."""

import numpy as np


def neutral_components(neutral: np.ndarray, var_frac: float = 0.5) -> np.ndarray:
    """Top principal components of neutral activations [M, D] that together explain
    `var_frac` of the variance. Returns an orthonormal [k, D] matrix."""
    x = neutral.astype(np.float32)
    x = x - x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    explained = np.cumsum(s**2) / np.sum(s**2)
    k = int(np.searchsorted(explained, var_frac) + 1)
    return vt[:k]


def project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the span of the orthonormal rows of `basis` [k, D] from v [..., D]."""
    return v - (v @ basis.T) @ basis


def compute_emotion_vectors(
    acts: np.ndarray,
    labels: list[str],
    neutral_acts: np.ndarray | None = None,
    var_frac: float = 0.5,
):
    """Emotion vectors as in the paper.

    acts: [N, L, D] mean story activations; labels: emotion of each story.
    1. Average the activations of each emotion's stories.
    2. Subtract the mean across emotions.
    3. If neutral activations [M, L, D] are given, project out (per layer) the top
       neutral PCs explaining `var_frac` of the variance.

    Returns (vectors [E, L, D] float32, emotions list, n_components per layer).
    """
    labels = np.asarray(labels)
    emotions = list(dict.fromkeys(labels.tolist()))
    means = np.stack([acts[labels == e].astype(np.float32).mean(0) for e in emotions])
    vectors = means - means.mean(0, keepdims=True)

    n_components = []
    if neutral_acts is not None:
        for l in range(vectors.shape[1]):
            basis = neutral_components(neutral_acts[:, l], var_frac)
            vectors[:, l] = project_out(vectors[:, l], basis)
            n_components.append(int(basis.shape[0]))
    return vectors, emotions, n_components


def classify_by_vectors(x: np.ndarray, vectors: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Nearest emotion by cosine similarity. x [N, D], vectors [E, D]. Returns indices."""
    xc = x.astype(np.float32) - center
    xc /= np.linalg.norm(xc, axis=1, keepdims=True) + 1e-8
    vn = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    return (xc @ vn.T).argmax(1)


def topic_split(topic_ids: list, test_frac: float = 0.2, seed: int = 0):
    """Train/test masks that hold out whole topics, so test stories share no premise with training."""
    topic_ids = np.asarray(topic_ids)
    uniq = np.unique(topic_ids)
    rng = np.random.default_rng(seed)
    test_topics = rng.choice(uniq, size=max(1, int(round(test_frac * len(uniq)))), replace=False)
    test = np.isin(topic_ids, test_topics)
    return ~test, test


def logistic_probe(x_train, y_train, x_test, y_test, C: float = 1.0):
    """Multinomial logistic-regression probe baseline. Returns (accuracy, fitted model)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    clf.fit(x_train.astype(np.float32), y_train)
    return float(clf.score(x_test.astype(np.float32), y_test)), clf
