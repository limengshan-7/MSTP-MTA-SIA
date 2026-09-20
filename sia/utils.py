"""Configuration, provenance, and safe output-directory helpers."""

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key in ("seed", "data", "federated", "attack", "model"):
        if key not in cfg:
            raise ValueError(f"Missing configuration key: {key}")
    f, a, m = cfg["federated"], cfg["attack"], cfg["model"]
    integer_fields = [(f, name) for name in ("clients", "classes", "rounds", "local_epochs",
                      "local_batch_size", "public_batch_size", "query_batch_size", "queries_per_client")]
    integer_fields += [(a, name) for name in ("epochs", "patience", "batch_size")]
    integer_fields += [(m, name) for name in ("hidden_dim", "tcn_width", "lstm_layers", "heads")]
    for section, name in integer_fields:
        value = section[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] < 2 ** 32 - 100:
        raise ValueError("seed must be an integer in [0,2**32-100)")
    if cfg["data"]["workers"] < 0 or cfg["data"]["augmentation_strength"] < 0:
        raise ValueError("workers and augmentation_strength must be nonnegative")
    for section, name in ((f, "local_lr"), (f, "distill_lr"), (a, "lr")):
        if section[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= m["dropout"] < 1:
        raise ValueError("dropout must lie in [0,1)")
    if f["clients"] < 2 or f["rounds"] < 1 or f["local_epochs"] < 1:
        raise ValueError("Invalid FL dimensions")
    if f["alpha"] <= 0 or f["distill_epochs"] < 0:
        raise ValueError("Invalid alpha/distillation count")
    if f["record_at"] not in ("after_local", "after_distill"):
        raise ValueError("record_at must be after_local or after_distill")
    if not m["scales"] or any(int(s) != s or s < 1 for s in m["scales"]):
        raise ValueError("scales must be positive integers")
    if len(set(m["scales"])) != len(m["scales"]) or 1 not in m["scales"]:
        raise ValueError("scales must be unique and include 1")
    if m["hidden_dim"] % 2 or m["hidden_dim"] % m["heads"]:
        raise ValueError("hidden_dim must be even and divisible by heads")
    if a["epochs"] < 1 or a["patience"] < 1 or a["batch_size"] < 1:
        raise ValueError("Invalid attack training settings")
    ratios = a["split_ratios"]
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Three positive attack split ratios must sum to 1")
    return cfg


def save_json(path, obj):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fresh_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}. Choose a new output path.")
    path.mkdir(parents=True)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    return torch.device(name)


def environment_info():
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
