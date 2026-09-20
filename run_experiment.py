"""One-command real-data collection, attack training, and held-out evaluation."""
import argparse
from sia.utils import load_config, fresh_directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/fedmd_cifar100.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    # Fail before creating output if a required dependency is unavailable.
    from sia.federated import collect_observations
    from sia.training import train_attack, evaluate_attack
    out = fresh_directory(args.output)
    obs = collect_observations(cfg, out / "observations", args.device)
    checkpoint = train_attack(cfg, obs, out / "attack", device_name=args.device)
    evaluate_attack(obs, out / "evaluation", checkpoint, device_name=args.device)


if __name__ == "__main__":
    main()
