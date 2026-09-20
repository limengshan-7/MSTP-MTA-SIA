"""Probability-only features, Eqs. (1)-(2); never accepts task labels."""

import numpy as np

FEATURE_NAMES = ("self_nll", "one_minus_conf", "delta_nll", "delta_conf",
                 "rank", "delta_rank")


def average_ranks(values):
    """Rank axis 1 of [N,K,T], ascending, one-based, averaging exact ties.

    Averaging ties avoids introducing a preference for low client indices.
    Ranks are not divided by K (a documented choice following Eq. (2)).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 3 or not np.isfinite(v).all():
        raise ValueError("Expected finite [N,K,T] values")
    ranks = np.empty(v.shape, dtype=np.float32)
    for k in range(v.shape[1]):
        below = (v < v[:, k:k + 1, :]).sum(axis=1)
        equal = (v == v[:, k:k + 1, :]).sum(axis=1)
        ranks[:, k, :] = 1.0 + below + 0.5 * (equal - 1)
    return ranks


def features_from_confidence(confidence):
    """Return [N,K,T,6] features from maximum predictive probabilities."""
    c = np.asarray(confidence, dtype=np.float64)
    if c.ndim != 3 or min(c.shape) < 1 or c.shape[1] < 2:
        raise ValueError("Expected [N,K,T], with N,T>=1 and K>=2")
    if not np.isfinite(c).all() or np.any(c <= 0) or np.any(c > 1):
        raise ValueError("Confidence must be finite and in (0,1]")
    loss = -np.log(np.maximum(c, 1e-12))
    ranks = average_ranks(loss)

    def delta(a):
        return np.diff(a, axis=2, prepend=a[:, :, :1])

    # IMPORTANT: Eq. (2) uses delta(c), NOT delta(1-c) from the old script.
    return np.stack((loss, 1.0 - c, delta(loss), delta(c), ranks,
                     delta(ranks)), axis=-1).astype(np.float32)


def features_from_probabilities(probabilities, rounds=None):
    """Read [N,K,T,C] logs, optionally using only the first `rounds` rounds.

    Prefix truncation happens before ANY feature extraction or normalization.
    Validation in chunks also supports memory-mapped probability arrays.
    """
    if probabilities.ndim != 4 or min(probabilities.shape) < 1:
        raise ValueError("Expected nonempty probabilities [N,K,T,C]")
    if probabilities.shape[1] < 2 or probabilities.shape[3] < 2:
        raise ValueError("At least two clients and two task classes are required")
    total = probabilities.shape[2]
    rounds = total if rounds is None else int(rounds)
    if not 1 <= rounds <= total:
        raise ValueError(f"rounds must lie in [1,{total}]")
    c = np.empty((*probabilities.shape[:2], rounds), dtype=np.float32)
    for lo in range(0, len(probabilities), 64):
        p = np.asarray(probabilities[lo:lo + 64, :, :rounds, :])
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError("Invalid predictive probabilities")
        if not np.allclose(p.sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError("Class probabilities must sum to 1")
        c[lo:lo + len(p)] = p.max(axis=-1)
    return features_from_confidence(c)


def fit_normalizer(x_train):
    """Fit ONLY on attack-training queries; pool samples, clients, and time."""
    x = np.asarray(x_train, dtype=np.float64)
    if x.ndim != 4 or x.shape[-1] != 6 or len(x) == 0:
        raise ValueError("Expected nonempty [N,K,T,6] attack-training features")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite training features")
    mean = x.mean(axis=(0, 1, 2))
    std = np.maximum(x.std(axis=(0, 1, 2)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)
