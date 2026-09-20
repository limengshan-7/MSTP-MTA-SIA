"""Sample-identity splits: each row contains ALL K trajectories of a query."""

import numpy as np


def dirichlet_partition(labels, clients, alpha, seed):
    """Preserve the old script's RandomState and per-class allocation rule."""
    if clients < 2 or alpha <= 0:
        raise ValueError("Need clients>=2 and alpha>0")
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    by_class = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    for indices in by_class:
        rng.shuffle(indices)
    parts = [[] for _ in range(clients)]
    for indices in by_class:
        proportions = rng.dirichlet([alpha] * clients)
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for k, block in enumerate(np.split(indices, cuts)):
            parts[k].extend(block.tolist())
    for part in parts:
        rng.shuffle(part)
    if any(len(part) == 0 for part in parts):
        raise ValueError("An empty client was sampled; choose another seed/alpha")
    assert sorted(i for p in parts for i in p) == list(range(len(labels)))
    return [np.asarray(p, dtype=np.int64) for p in parts]


def select_queries(partition, per_client, seed):
    rng = np.random.RandomState(seed + 7)
    ids, sources = [], []
    for k, part in enumerate(partition):
        chosen = rng.choice(part, size=min(per_client, len(part)), replace=False)
        ids.extend(chosen.tolist())
        sources.extend([k] * len(chosen))
    return np.asarray(ids, dtype=np.int64), np.asarray(sources, dtype=np.int64)


def split_queries(sample_ids, source_labels, ratios=(0.6, 0.2, 0.2), seed=42):
    """Source-stratified partition of QUERY indices, never (query,client) pairs.

    With 100 queries/client the split is exactly 60/20/20 per client.
    For smaller pools, floor the train/validation counts and reserve the rest
    for testing. All three sets must contain at least one query per client.
    """
    ids, y = np.asarray(sample_ids), np.asarray(source_labels)
    if ids.ndim != 1 or y.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Query IDs must be unique, with one source label each")
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive split ratios must sum to 1")
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": [], "test": []}
    for k in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == k))
        ntr, nva = int(len(idx) * ratios[0]), int(len(idx) * ratios[1])
        if min(ntr, nva, len(idx) - ntr - nva) < 1:
            raise ValueError(f"Insufficient queries from client {k} for this split")
        for name, rows in zip(result, np.split(idx, [ntr, ntr + nva])):
            result[name].extend(rows.tolist())
    for name in result:
        result[name] = rng.permutation(result[name]).astype(int).tolist()
    validate_splits(ids, y, result)
    return result


def validate_splits(sample_ids, source_labels, splits):
    ids, y = np.asarray(sample_ids), np.asarray(source_labels)
    if ids.ndim != 1 or ids.shape != y.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Query IDs must be globally unique within this run")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Require train, validation, and test splits")
    all_rows = []
    for name, rows in splits.items():
        r = np.asarray(rows)
        if not np.issubdtype(r.dtype, np.integer) or r.ndim != 1 or len(r) == 0:
            raise ValueError(f"Invalid split: {name}")
        if np.any(r < 0) or np.any(r >= len(ids)):
            raise ValueError("Split contains out-of-range indices")
        if set(y[r].tolist()) != set(y.tolist()):
            raise ValueError(f"{name} is missing a source client")
        all_rows.extend(r.tolist())
    if len(all_rows) != len(ids) or set(all_rows) != set(range(len(ids))):
        raise ValueError("Splits must be a disjoint partition of all queries")


def split_counts(labels, splits):
    y = np.asarray(labels)
    return {name: {str(k): int((y[rows] == k).sum()) for k in np.unique(y)}
            for name, rows in splits.items()}
