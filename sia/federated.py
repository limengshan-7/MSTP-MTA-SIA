"""Local experiment simulator. Client weights stay outside the attack pipeline.

Uses the supplied script's heterogeneous CNNs, SGD and public probability KL
alignment. This is its FedMD-style protocol, without public supervised pretraining.
The public CIFAR-10 labels are never used as CIFAR-100 labels.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.clients import make_clients
from sia.splits import dirichlet_partition, select_queries, split_queries, split_counts
from sia.utils import (fresh_directory, save_json, sha256, seed_everything,
                       seed_worker, get_device, environment_info)

C100_MEAN, C100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
C10_MEAN, C10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def make_loader(dataset, batch_size, shuffle, workers, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=False,
                      generator=torch.Generator().manual_seed(seed),
                      worker_init_fn=seed_worker)


def local_epoch(model, loader, cfg, device):
    # Preserve legacy optimizer lifecycle: new SGD momentum state per local epoch.
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["local_lr"],
                                momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    model.train()
    total, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite local supervised loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        seen += len(x)
    return total / seen


@torch.no_grad()
def predict(model, loader, device):
    model.to(device).eval()
    values = []
    for x, _ in loader:
        values.append(model(x.to(device)).softmax(dim=-1).cpu())
    return torch.cat(values, dim=0)


def distill(models, public_loader, cfg, device):
    if cfg["distill_epochs"] == 0:
        return
    # Freeze the SAME ensemble target for all clients before updating any client.
    consensus = None
    for model in models:
        p = predict(model, public_loader, device)
        consensus = p if consensus is None else consensus + p
        model.cpu()
    consensus = consensus / len(models)
    for model in models:
        model.to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["distill_lr"],
                                    momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
        for _ in range(cfg["distill_epochs"]):
            offset = 0
            for x, _ in public_loader:  # loader MUST be deterministic, shuffle=False
                target = consensus[offset:offset + len(x)].to(device)
                offset += len(x)
                optimizer.zero_grad(set_to_none=True)
                loss = F.kl_div(F.log_softmax(model(x.to(device)), dim=-1),
                                target, reduction="batchmean")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite public alignment loss")
                loss.backward()
                optimizer.step()
        model.cpu()


def collect_observations(cfg, output, device_name="auto"):
    seed_everything(cfg["seed"])
    device = get_device(device_name)
    f, d = cfg["federated"], cfg["data"]
    if f["classes"] != 100:
        raise ValueError("Real-data collector supports CIFAR-100 (100 classes) only")
    root = Path(d["root"])
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=d["augmentation_strength"],
                               contrast=d["augmentation_strength"],
                               saturation=d["augmentation_strength"]),
        transforms.ToTensor(), transforms.Normalize(C100_MEAN, C100_STD)])
    deterministic = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(C100_MEAN, C100_STD)])
    public_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(C10_MEAN, C10_STD)])
    private_train = datasets.CIFAR100(root, train=True, transform=augment, download=d["download"])
    query_view = datasets.CIFAR100(root, train=True, transform=deterministic, download=False)
    public_data = datasets.CIFAR10(root, train=False, transform=public_transform, download=d["download"])
    partition = dirichlet_partition(private_train.targets, f["clients"], f["alpha"], cfg["seed"])
    ids, sources = select_queries(partition, f["queries_per_client"], cfg["seed"])
    splits = split_queries(ids, sources, cfg["attack"]["split_ratios"], cfg["seed"])
    out = fresh_directory(output)
    save_json(out / "queries.json", {"sample_ids": ids.tolist(), "source_labels": sources.tolist()})
    save_json(out / "splits.json", splits)
    # Indices only, not private raw data. These belong to the experiment simulator.
    save_json(out / "partition.json", {str(k): part.tolist() for k, part in enumerate(partition)})
    metadata = {"format_version": 1, "complete": False, "synthetic": False,
                "dataset": "CIFAR-100", "public_dataset": "CIFAR-10 test",
                "record_at": f["record_at"], "query_transform": "deterministic CIFAR-100 normalization",
                "num_queries": len(ids), "num_clients": f["clients"],
                "num_rounds": f["rounds"], "num_classes": 100,
                "split_counts": split_counts(sources, splits), "config": cfg,
                "environment": environment_info(),
                "implementation_status": "paper-aligned reference; table reproduction unverified"}
    save_json(out / "metadata.json", metadata)
    probability_path = out / "probabilities.npy"
    probabilities = np.lib.format.open_memmap(probability_path, mode="w+", dtype=np.float32,
                                              shape=(len(ids), f["clients"], f["rounds"], 100))
    query_loader = make_loader(Subset(query_view, ids.tolist()), f["query_batch_size"], False,
                                d["workers"], cfg["seed"])
    public_loader = make_loader(public_data, f["public_batch_size"], False, d["workers"], cfg["seed"])
    local_loaders = [make_loader(Subset(private_train, part.tolist()), f["local_batch_size"],
                                 True, d["workers"], cfg["seed"] + k) for k, part in enumerate(partition)]
    models = make_clients(f["clients"], 100)
    round_log = []
    print(f"Device={device}; client sizes={[len(p) for p in partition]}", flush=True)
    print(f"Queries={len(ids)}, split counts={metadata['split_counts']}", flush=True)
    print("Queries remain in local FL training sets; observing them does not update clients.", flush=True)

    def record(round_index):
        for k, model in enumerate(models):
            probabilities[:, k, round_index, :] = predict(model, query_loader, device).numpy()
            model.cpu()
        probabilities.flush()

    for t in range(f["rounds"]):
        losses = []
        for k, (model, loader) in enumerate(zip(models, local_loaders)):
            model.to(device)
            loss = None
            for _ in range(f["local_epochs"]):
                loss = local_epoch(model, loader, f, device)
            model.cpu()
            losses.append(loss)
            print(f"round {t + 1}/{f['rounds']} client {k + 1}/{f['clients']} local loss={loss:.4f}", flush=True)
        if f["record_at"] == "after_local":
            record(t)
        distill(models, public_loader, f, device)
        if f["record_at"] == "after_distill":
            record(t)
        round_log.append({"round": t + 1, "local_final_epoch_losses": losses})
        save_json(out / "federated_log.json", round_log)
        print(f"Recorded round {t + 1}/{f['rounds']} ({f['record_at']})", flush=True)
    probabilities.flush()
    del probabilities
    metadata["complete"] = True
    metadata["probabilities_sha256"] = sha256(probability_path)
    metadata["queries_sha256"] = sha256(out / "queries.json")
    metadata["splits_sha256"] = sha256(out / "splits.json")
    save_json(out / "metadata.json", metadata)
    print(f"Saved observations: {out}", flush=True)
    return out
