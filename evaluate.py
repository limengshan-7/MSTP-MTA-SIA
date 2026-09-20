"""Evaluate the held-out test queries after checkpoint selection on validation."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--last-round", action="store_true", help="Label-free max-confidence heuristic, not HU-SIA")
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args()
    from sia.training import evaluate_attack
    evaluate_attack(args.observations, args.output, args.checkpoint, args.rounds,
                     args.device, args.allow_synthetic)


if __name__ == "__main__":
    main()
