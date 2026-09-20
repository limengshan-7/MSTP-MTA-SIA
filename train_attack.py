"""Train one attack using fixed, sample-disjoint probability logs."""
import argparse
from sia.utils import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/fedmd_cifar100.yaml")
    p.add_argument("--observations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["mstp", "single", "average", "tcn"], default="mstp")
    p.add_argument("--ablation", choices=["none", "scales_1_2", "no_position", "no_tcn",
                                          "no_temporal_attention", "no_scale_attention",
                                          "no_cross_client_attention"], default="none")
    p.add_argument("--rounds", type=int)
    p.add_argument("--seed", type=int, help="Attack-training seed; fixed observation splits are unchanged")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    from sia.training import train_attack
    train_attack(cfg, args.observations, args.output, args.method, args.rounds,
                  args.ablation, args.device, args.allow_synthetic)


if __name__ == "__main__":
    main()
