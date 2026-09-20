"""Read-only interface from stored probability logs to the attack pipeline."""

from pathlib import Path
import numpy as np

from sia.features import features_from_probabilities
from sia.splits import validate_splits
from sia.utils import read_json, sha256


def load_observations(directory, rounds=None, allow_synthetic=False):
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    if metadata.get("format_version") != 1 or not metadata.get("complete"):
        raise ValueError("Unsupported or incomplete observation logs")
    if metadata.get("synthetic") and not allow_synthetic:
        raise ValueError("Synthetic logs require --allow-synthetic; never report them as paper results")
    for filename, key in (("probabilities.npy", "probabilities_sha256"),
                           ("queries.json", "queries_sha256"), ("splits.json", "splits_sha256")):
        if sha256(directory / filename) != metadata[key]:
            raise ValueError(f"Observation integrity check failed: {filename}")
    query = read_json(directory / "queries.json")
    ids, y = np.asarray(query["sample_ids"]), np.asarray(query["source_labels"])
    splits = read_json(directory / "splits.json")
    validate_splits(ids, y, splits)
    p = np.load(directory / "probabilities.npy", mmap_mode="r", allow_pickle=False)
    expected = (len(ids), metadata["num_clients"], metadata["num_rounds"], metadata["num_classes"])
    if p.shape != expected or set(y.tolist()) != set(range(metadata["num_clients"])):
        raise ValueError("Probability shape or source labels disagree with metadata")
    features = features_from_probabilities(p, rounds=rounds)
    return features, y.astype(np.int64), ids, splits, metadata
