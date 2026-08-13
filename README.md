# NIMOSEF-R

Official repository for:

**NIMOSEF-R: Neural Implicit Motion and Segmentation Functions with Riemannian Embedding Priors and Breath-Motion Correction**
Gonçalo Canastra, Pedro M. Gordaliza, Ruud B. van Heeswijk, Jonas Richiardi, Jaume Banus
_Off-Grid Workshop @ MICCAI 2026_

---

## Status

🧹 This repository is being cleaned up and verified for the camera-ready release. Interfaces described below reflect the current state of the code but are still being re-checked end-to-end; expect some rough edges until this notice is removed.

---

## About

NIMOSEF-R is a coordinate-based implicit neural representation for joint segmentation, image reconstruction, and motion estimation from short-axis cardiac MRI. It extends our earlier work, [NIMOSEF](https://github.com/jbanusco/nimosef-v1) (MICCAI 2025), with two additions:

- **Riemannian latent geometry**: subject-specific shape codes are constrained to a hyperbolic Poincaré ball, and per-timepoint motion codes are projected onto a unit hypersphere, to encode hierarchical anatomical variation and cyclical cardiac motion respectively.
- **Learnable slice-alignment correction**: a per-slice rigid in-plane transform, jointly optimized with the network, that corrects inter-slice respiratory misalignment in short-axis stacks.

The method is evaluated on the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/), comparing the full Riemannian model (NIMOSEF-R) against an Euclidean-latent configuration (NIMOSEF-E) and ANTs-based rigid slice registration.

---

## Repository structure

```
nimosef/
  data/        # dataset loading and NIfTI -> parquet preprocessing
  models/      # network architecture: MLP/SIREN backbone, slice-correction module,
               # Riemannian (Poincare ball / hypersphere) latent embeddings
  training/    # train / inference / result-generation entrypoints
  losses/      # composite training loss (segmentation, reconstruction, displacement,
               # Jacobian, smoothness, manifold terms)
  metrics/     # evaluation metrics used in the paper
  analysis/    # latent-space and periodicity analysis utilities
  config/      # example training configs (JSON) for the experiments in the paper
  utils/       # shared helpers (core, geometry)
checkpoints/    # pretrained NIMOSEF-E / NIMOSEF-R weights (model.pth + hyperparameters.json)
docs/           # data_pipeline.md: expected input layout, preprocessing, splits, dataset loading
scripts/        # SLURM/Singularity launch scripts (edit the paths at the top for
                # your own environment before running)
paper_analysis/ # generates every table and figure in the paper from the two checkpoints
                # (Tables 1-3, Fig. 2-4); not a pytest suite, despite living next to tests/
tests/          # pytest unit tests only (pytest tests/)
```

---

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

See `pyproject.toml` for the full dependency list (PyTorch is expected to already be installed for your CUDA setup; it is not pinned here).

---

## Quick start

The scripts below mirror `scripts/run_train.sh`, `scripts/run_inference.sh`, and `scripts/run_preprocessing.sh`, which run the same commands inside a Singularity container on a SLURM cluster — edit the paths at the top of those scripts for your own environment.

### 1. Preprocess a BIDS-like short-axis CMR dataset (e.g. ACDC) into per-subject coordinate tables

The paper uses [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/), downloaded directly from the official source (not redistributed here — see its license/terms of use). `preprocess_data` below is the *second* stage of preprocessing: it expects a `derivatives/` tree with per-subject ROI crops, segmentations, and axis-transform files already in place — see [`docs/data_pipeline.md`](docs/data_pipeline.md) for the exact expected layout (the step that produces it from a raw ACDC download is not included in this repository) and for split generation / dataset loading.

```bash
python -m nimosef.data.preprocess_data \
  --root /path/to/ACDC \
  --use_roi True \
  --save_dir sa_implicit \
  --manifest_dir manifests_nimosef
```

### 2. Train

Training is driven by a flat set of CLI flags (see `nimosef/training/args.py` for the full list); the configs used for the paper's experiments are checked in under `nimosef/config/` as JSON files, e.g. `config_muon.json` (NIMOSEF-R) and `config_baseline_muon.json` (NIMOSEF-E). To launch a run from one of these configs:

```bash
config_params=$(python3 -c "
import json
c = json.load(open('nimosef/config/config_muon.json'))
print(' '.join(f'--{k} {v}' for k, v in c.items() if v is not None))
")
python -m nimosef.training.train ${config_params}
```

### 3. Generate results from a pretrained checkpoint

Pretrained checkpoints for the two paper configurations are included under `checkpoints/` (`nimosef_e/model.pth`, `nimosef_r/model.pth`, each with its `hyperparameters.json`) so you can reproduce the paper's reconstructions/tables without retraining. This uses `nimosef.training.generate_results` (verified end-to-end against both checkpoints on the real ACDC cohort — loads with zero missing/unexpected state-dict keys and produces reconstructions/segmentations/boundary parquets per subject):

```bash
python -m nimosef.training.generate_results \
  --data_folder /path/to/ACDC \
  --split_file /path/to/ACDC/derivatives/manifests_nimosef/dataset_manifest.json \
  --model_to_rec checkpoints/nimosef_r/model.pth \
  --riemannian True \
  --skip_correction False \
  --mode train \
  --save_rec_folders /path/to/output/reconstructions
```

Set `--riemannian False` for `checkpoints/nimosef_e/model.pth` (Euclidean latents) instead — it **must** match the checkpoint, since it selects a different model architecture (the flag's default, `False`, is *not* the value NIMOSEF-R needs). `--mode` must be `train`: both checkpoints only have subject-specific latent codes for the 105-subject training cohort they were fit on (see the paper's Discussion on the lack of a held-out test split) — `val`/`test` subjects aren't in `shape_code.weight` and will fail to load.

If subject manifests were written from inside a container (e.g. via `scripts/run_train.sh`'s Singularity/SLURM setup) they'll contain container-internal paths (`/usr/data/...`); set the `DATA_PATH` environment variable to your real dataset root and `nimosef.data.dataset.NiftiDataset` will substitute it automatically.

For test-time latent optimization on subjects *not* in the training cohort (freeze the network, fit new shape/motion codes — see `nimosef.training.inference` / `--initial_model_path`) and the full SLURM/Singularity invocation pattern, see `scripts/run_inference.sh` and `scripts/run_generate_results_data.sh`.

---

## Citation

If you use this code or build on this work, please cite:

```
[NIMOSEF-R bibtex -- to be added once proceedings are indexed]
```

and our earlier work:

```
Banus, J., Delaloye, A., Gordaliza, P.M., Georgantas, C., van Heeswijk, R.B., Richiardi, J.:
NIMOSEF: Neural implicit motion and segmentation functions.
In: International Conference on Medical Image Computing and Computer-Assisted
Intervention. pp. 444-454. Springer (2025)
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

This work was supported by the Nuvolos Fellowship Programme, awarded to J. Banus, and by the Swiss National Foundation Sinergia Grant CRSII5_202276/1.
