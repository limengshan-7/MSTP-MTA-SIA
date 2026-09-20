"""Export raw Eq. (2) features for inspection; no normalization is fitted here."""
import argparse
from pathlib import Path
import numpy as np
from sia.observations import load_observations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", required=True)
    p.add_argument("--output", required=True, help="New .npz file")
    p.add_argument("--rounds", type=int)
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args()
    path = Path(args.output)
    if path.exists() or path.suffix != ".npz":
        raise ValueError("Choose a new output file with .npz extension")
    x, y, ids, _, _ = load_observations(args.observations, args.rounds, args.allow_synthetic)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=x, source_labels=y, sample_ids=ids)
    print(f"Saved {x.shape} raw features to {path}")


if __name__ == "__main__":
    main()
