"""Train on auxiliary queries; select on validation; evaluate test separately."""

from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.mstp_mta_sia import (build_model, MSTPMTASIA, parameter_count,
                                matched_tcn_width)
from sia.features import fit_normalizer, normalize
from sia.observations import load_observations
from sia.utils import (fresh_directory, save_json, get_device, seed_everything,
                       environment_info)

ABLATIONS = {"no_position": "positional_encoding", "no_tcn": "tcn",
             "no_temporal_attention": "temporal_attention",
             "no_scale_attention": "scale_attention",
             "no_cross_client_attention": "cross_client_attention"}


def make_attack_loader(x, y, rows, batch_size, shuffle, seed):
    rows = np.asarray(rows, dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x[rows])),
                       torch.from_numpy(np.ascontiguousarray(y[rows])))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def resolved_model_config(cfg, method, ablation="none"):
    m = dict(cfg["model"])
    if method == "single":
        m["scales"] = [1]
    if ablation != "none":
        if method != "mstp":
            raise ValueError("Component ablations apply to --method mstp")
        if ablation == "scales_1_2":
            m["scales"] = [1, 2]
        elif ablation in ABLATIONS:
            m[ABLATIONS[ablation]] = False
        else:
            raise ValueError(f"Unknown ablation: {ablation}")
    if method == "tcn":
        target = parameter_count(MSTPMTASIA(cfg["model"]))
        m["baseline_width"] = matched_tcn_width(target)
        m["target_parameters"] = target
    return m


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for xb, yb in loader:
        logits = model(xb.to(device))
        loss = F.cross_entropy(logits, yb.to(device), reduction="sum")
        total += loss.item()
        count += len(yb)
    return total / count


def train_attack(cfg, observations, output, method="mstp", rounds=None,
                 ablation="none", device_name="auto", allow_synthetic=False):
    x, y, ids, splits, meta = load_observations(observations, rounds, allow_synthetic)
    if not meta.get("synthetic"):
        if cfg["federated"] != meta["config"]["federated"]:
            raise ValueError("FL config does not match the stored observations; use their original config")
        if cfg["attack"]["split_ratios"] != meta["config"]["attack"]["split_ratios"]:
            raise ValueError("Split ratios differ from the saved, fixed query split")
    model_cfg = resolved_model_config(cfg, method, ablation)
    seed_everything(cfg["seed"])  # Reset after parameter-count matching.
    device = get_device(device_name)
    model = build_model(method, model_cfg).to(device)
    mean, std = fit_normalizer(x[splits["train"]])
    scaled = normalize(x, mean, std)
    a = cfg["attack"]
    train_loader = make_attack_loader(scaled, y, splits["train"], a["batch_size"], True, cfg["seed"])
    valid_loader = make_attack_loader(scaled, y, splits["validation"], a["batch_size"], False, cfg["seed"])
    # One model contains encoders, all attention modules, and the scoring head.
    optimizer = torch.optim.AdamW(model.parameters(), lr=a["lr"], weight_decay=a["weight_decay"])
    optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    out = fresh_directory(output)
    manifest = {"method": method, "ablation": ablation, "observed_rounds": x.shape[2],
                "parameters": parameter_count(model), "config": cfg, "model_config": model_cfg,
                "synthetic": bool(meta.get("synthetic")), "environment": environment_info(),
                "probabilities_sha256": meta["probabilities_sha256"],
                "queries_sha256": meta["queries_sha256"], "splits_sha256": meta["splits_sha256"],
                "observation_metadata": meta}
    if method == "tcn":
        manifest["parameter_match_relative_error"] = abs(manifest["parameters"] - model_cfg["target_parameters"]) / model_cfg["target_parameters"]
    save_json(out / "run.json", manifest)
    best_loss, stale, best_epoch, history = float("inf"), 0, None, []
    print(f"Training {method}/{ablation}: {manifest['parameters']:,} parameters, T={x.shape[2]}", flush=True)
    for epoch in range(1, a["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite attack training loss")
            loss.backward()
            if a["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a["gradient_clip"])
            optimizer.step()
            total += loss.item() * len(yb)
            count += len(yb)
        val = validation_loss(model, valid_loader, device)
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite attack validation loss")
        history.append({"epoch": epoch, "train_loss": total / count, "validation_loss": val})
        save_json(out / "history.json", history)
        if val < best_loss - a["min_delta"]:
            best_loss, stale, best_epoch = val, 0, epoch
            # Real snapshot: no mutable state_dict references to later epochs.
            checkpoint = {"format_version": 1, "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                "method": method, "model_config": model_cfg, "manifest": manifest,
                "normalization_mean": torch.from_numpy(mean.copy()),
                "normalization_std": torch.from_numpy(std.copy()),
                "best_epoch": epoch, "best_validation_loss": val}
            torch.save(checkpoint, out / "best.pt")
        else:
            stale += 1
        print(f"epoch {epoch}: train={total / count:.5f}, validation={val:.5f}", flush=True)
        if stale >= a["patience"]:
            break
    save_json(out / "training_summary.json", {"best_epoch": best_epoch,
              "best_validation_loss": best_loss, "epochs_completed": len(history),
              "test_set_evaluated_during_training": False})
    return out / "best.pt"


@torch.no_grad()
def evaluate_attack(observations, output, checkpoint=None, rounds=None,
                    device_name="auto", allow_synthetic=False):
    device = get_device(device_name)
    ckpt = None
    if checkpoint is not None:
        # Only state tensors and plain metadata are needed; no arbitrary model unpickling.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        trained_rounds = ckpt["manifest"]["observed_rounds"]
        if rounds is not None and rounds != trained_rounds:
            raise ValueError("Evaluate a checkpoint with the SAME T used in attack training")
        rounds = trained_rounds
    x, y, ids, splits, meta = load_observations(observations, rounds, allow_synthetic)
    test_rows = np.asarray(splits["test"], dtype=np.int64)
    if ckpt is None:
        # Explicit label-free heuristic. It is NOT named HU-SIA.
        predictions = x[test_rows, :, -1, 0].argmin(axis=1)
        probabilities = None
        method = "last_round_max_confidence"
        best_epoch = None
    else:
        for key in ("probabilities_sha256", "queries_sha256", "splits_sha256"):
            if ckpt["manifest"][key] != meta[key]:
                raise ValueError(f"Checkpoint/observations mismatch: {key}")
        scaled = normalize(x, ckpt["normalization_mean"].numpy(), ckpt["normalization_std"].numpy())
        model = build_model(ckpt["method"], ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        loader = make_attack_loader(scaled, y, test_rows, 128, False, 0)
        probabilities = np.concatenate([model(xb.to(device)).softmax(dim=1).cpu().numpy()
                                         for xb, _ in loader], axis=0)
        predictions = probabilities.argmax(axis=1)
        method = f"{ckpt['method']}/{ckpt['manifest']['ablation']}"
        best_epoch = ckpt["best_epoch"]
    truth = y[test_rows]
    clients = meta["num_clients"]
    confusion = np.zeros((clients, clients), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    report = {"method": method, "synthetic": bool(meta.get("synthetic")),
              "asr_percent": float(100 * (predictions == truth).mean()),
              "uniform_guess_percent": 100.0 / clients, "test_queries": len(test_rows),
              "observed_rounds": x.shape[2],
              "recorded_responses_per_query": clients * x.shape[2],
              "responses_used_per_query": clients if ckpt is None else clients * x.shape[2],
              "best_epoch": best_epoch, "confusion_matrix": confusion.tolist(),
              "probabilities_sha256": meta["probabilities_sha256"],
              "result_status": "new reference run; not a reproduced paper table value"}
    out = fresh_directory(output)
    save_json(out / "metrics.json", report)
    arrays = {"sample_ids": ids[test_rows], "source_labels": truth,
              "predicted_sources": predictions}
    if probabilities is not None:
        arrays["source_probabilities"] = probabilities
    np.savez_compressed(out / "predictions.npz", **arrays)
    prefix = "SYNTHETIC SMOKE CHECK ONLY: " if meta.get("synthetic") else ""
    print(f"{prefix}{method}, T={x.shape[2]}, test ASR={report['asr_percent']:.2f}%", flush=True)
    return report
