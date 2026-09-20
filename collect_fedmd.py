"""Generate probability logs from the FedMD/CIFAR-100 reference setting."""
import argparse
from sia.utils import load_config


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
    from sia.federated import collect_observations
    collect_observations(cfg, args.output, args.device)


if __name__ == "__main__":
    main()
