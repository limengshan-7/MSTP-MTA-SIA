"""Synthetic observations for software checks ONLY, never paper experiments."""
import numpy as np
from sia.splits import split_queries, split_counts
from sia.utils import fresh_directory, save_json, sha256


def make_synthetic_observations(output, cfg):
    out = fresh_directory(output)
    rng = np.random.default_rng(cfg["seed"])
    clients, rounds, classes, per_client = 3, 5, 5, 20
    ids = np.arange(clients * per_client)
    y = np.repeat(np.arange(clients), per_client)
    # Independent of source labels. No fabricated success rate is built in.
    logits = rng.normal(size=(len(ids), clients, rounds, classes))
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
    np.save(out / "probabilities.npy", p, allow_pickle=False)
    splits = split_queries(ids, y, cfg["attack"]["split_ratios"], cfg["seed"])
    save_json(out / "queries.json", {"sample_ids": ids.tolist(), "source_labels": y.tolist()})
    save_json(out / "splits.json", splits)
    save_json(out / "metadata.json", {
        "format_version": 1, "complete": True, "synthetic": True,
        "dataset": "SYNTHETIC SOFTWARE CHECK ONLY", "num_clients": clients,
        "num_queries": len(ids), "num_rounds": rounds, "num_classes": classes,
        "split_counts": split_counts(y, splits), "config": cfg,
        "probabilities_sha256": sha256(out / "probabilities.npy"),
        "queries_sha256": sha256(out / "queries.json"),
        "splits_sha256": sha256(out / "splits.json")})
    return out
