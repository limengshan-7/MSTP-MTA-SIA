# MSTP-MTA-SIA: FedMD / CIFAR-100 reference implementation

A single-setting research implementation of multi-scale temporal source inference
from client prediction probabilities. The task is to identify the source client
of a **known training member**, not to detect membership.

## Release status

This repository provides a compact reference implementation of MSTP-MTA-SIA for the representative FedMD/CIFAR-100 setting. It includes the core model, trajectory feature extraction, federated prediction collection, attack training, and evaluation pipeline.

The current release does not include the remaining datasets, all experimental configurations, pretrained weights, or complete multi-seed results. Implementations for the remaining datasets, federated learning protocols, and full experimental configurations will be released upon acceptance.

## Setting

| Item | Reference configuration |
| --- | --- |
| Private data | CIFAR-100 training set |
| Public alignment data | CIFAR-10 test set; labels are ignored |
| Clients | 10, alternating between two CNN architectures |
| Local partition | Per-class Dirichlet, alpha = 0.1 |
| Communication rounds | 20 |
| Local epochs | 10 per round |
| Public alignment epochs | 5 per round |
| Observations | After private local updates, before public alignment |
| Query pool | At most 100 queries per client, retained in local FL training |
| Attack split | 60% / 20% / 20%, stratified by source client |
| Temporal scales | 1, 2, 4 |
| Attack optimizer | AdamW, learning rate 0.001; validation-loss early stopping |

With 100 queries for every client this gives 600 attack-training, 200 validation,
and 200 test queries. Every query is stored as one group containing all client
trajectories. The same source-client population and FL run are used across these
splits; this is not a cross-run/client-transfer evaluation.

The FL simulator has access to the datasets and models to generate an experiment.
The attack stage only reads recorded probabilities and source labels for its
supervised training/evaluation. Task-class labels, raw images, model weights and
gradients are not attack features. Access to private-query client responses is an
explicit observer assumption, not a feature supplied by standard FedMD messages.

## Installation

Use Python 3.10 or newer. Install a compatible PyTorch/torchvision pair for your
machine using the [official installation instructions](https://pytorch.org/get-started/locally/).
For a CPU-only setup, one compatible pair is:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA, use the matching command from PyTorch's installer, then install
`requirements.txt`. Run commands below from this repository's root directory.
`workers: 0` is the portable default, including on Windows.



## Full single-setting experiment

```bash
python run_experiment.py --config configs/fedmd_cifar100.yaml --seed 42 --device cuda --output runs/reference_seed42
```

Use `--device cpu` when no CUDA device is available. This is a real FL workload,
not a quick demo; runtime depends heavily on hardware. Both datasets are
downloaded by torchvision if not already present. To use an existing dataset
cache, set `data.root` and `data.download: false` in the YAML configuration.

The query pool remains in the clients' FL datasets. Disjointness refers to
**attack-training/validation/test query identities**, not exclusion from FL
training. Removing the queries from FL training would change the task.

All output directories must be new. Existing runs are not overwritten. Data
collection marks its logs complete only after every round has been recorded;
incomplete logs cannot be used for training. FL checkpoint/resume is not
implemented in this initial release, so an interrupted collection needs a new run.

## Run the stages separately

Collect observations once:

```bash
python collect_fedmd.py --config configs/fedmd_cifar100.yaml --seed 42 --device cuda --output runs/observations_seed42
```

Train and evaluate the attack:

```bash
python train_attack.py --config configs/fedmd_cifar100.yaml --observations runs/observations_seed42 --method mstp --rounds 20 --device cuda --output runs/mstp_seed42_t20
python evaluate.py --observations runs/observations_seed42 --checkpoint runs/mstp_seed42_t20/best.pt --device cuda --output runs/mstp_seed42_t20/test
```

The best checkpoint is selected using validation loss. The test set is not used
for training or checkpoint selection. Evaluation reloads the saved normalizer,
checks observation/split hashes, and writes accuracy, predictions and a confusion
matrix. ASR is reported as a percentage; uniform guessing is 10% for 10 clients.

Optionally inspect the raw six-dimensional features:

```bash
python feature_extraction.py --observations runs/observations_seed42 --rounds 20 --output runs/raw_features_t20.npz
```

These exported features are not normalized. Normalization is fitted on training
queries inside `train_attack.py`, separately for each observation-window length.

## Comparisons and ablations

The trainable methods share observation logs, fixed query splits, input features,
positional-encoding convention, optimizer settings and checkpoint selection:

- `--method mstp`: full multi-scale temporal model with client comparison.
- `--method single`: same architecture and training protocol with scale set `{1}`.
- `--method average`: mean of projected round features before a shared scorer.
- `--method tcn`: residual dilated convolutions and global average pooling; width
  is selected before training to approximately match the full model's parameter
  count. The actual count and matching error are saved in `run.json`.

For example:

```bash
python train_attack.py --observations runs/observations_seed42 --method single --rounds 20 --output runs/single_t20 --device cuda
python evaluate.py --observations runs/observations_seed42 --checkpoint runs/single_t20/best.pt --output runs/single_t20/test --device cuda
```

Use `--rounds 5`, `10` or `20` to train separate attacks on prefixes of the same
recorded run. Truncation takes place **before** feature extraction and scaling.
Never select a prefix or method based on test accuracy and then report that test
set as unbiased. More recorded rounds also cover later FL stages; these are not
controlled experiments isolating query count alone.

For `--method mstp`, the available `--ablation` values are `scales_1_2`,
`no_position`, `no_tcn`, `no_temporal_attention`, `no_scale_attention` and
`no_cross_client_attention`. Use a separate output directory for every run.
The no-position variant still retains sequence order through the TCN and BiLSTM.

A separately named label-free final-round heuristic is available:

```bash
python evaluate.py --observations runs/observations_seed42 --last-round --rounds 20 --output runs/last_round_t20
```

This chooses the client with the largest final-round confidence (smallest
self-NLL); exact ties select the first client. It is **not claimed to be HU-SIA**
or the exact implementation behind the paper's Last-Round Only result. The
original baselines require a separate implementation audit before table-level
comparisons are claimed.

## Outputs and reproducibility

- Observation logs: `probabilities.npy` with shape `[N,K,T,C]`, `queries.json`,
  `splits.json`, `partition.json`, `federated_log.json`, `metadata.json`.
- Attack: `best.pt`, `history.json`, `run.json`, `training_summary.json`.
- Evaluation: `metrics.json`, `predictions.npz` with the original query IDs.

The default probability tensor occupies about 80 MB, excluding downloads. It is
memory mapped during collection. Dataset copies, raw logs, checkpoints and
private data are deliberately not included in this repository.

One seed is one run, not a five-run mean. For independent full runs, repeat
`run_experiment.py` with separately declared seeds and fresh output paths; this
changes the FL partition, model initialization, sampling and attack training.
Changing only `train_attack.py --seed` on cached observations instead measures
attack-training variability. Report which repetition protocol is used.

No test target accuracy is hard-coded. Hardware and software versions can affect
results even with fixed seeds. Do not insert paper numbers into this repository
as reproduced results until the actual commands have produced and verified them.

## Scope and responsible use

Use this code only with data and client prediction interfaces you are authorized
to access. Public CIFAR experiments do not authorize querying other deployments.

## Copyright

Copyright © 2026 MSTP-MTA-SIA Authors. All rights reserved.
