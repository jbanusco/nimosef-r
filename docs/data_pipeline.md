# NIMOSEF-R Data Pipeline

How to get data, preprocess it, generate train/val/test splits, and load the
result in PyTorch.

---

## 0. Getting data

The paper evaluates NIMOSEF-R on [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
**ACDC is not redistributed in this repository** — download it directly from
the official source and follow its license / terms of use. No demo data is
bundled here for the same reason.

---

## 1. Expected input layout

`nimosef.data.preprocess_data` (the entry point used in the top-level
README's Quick Start) is the **second stage** of preprocessing: it expects a
dataset already organized as a BIDS-like `derivatives/` tree, with per-subject
short-axis ROI crops, segmentations, and a per-subject imaging-plane geometry
transform:

```
<root>/derivatives/
  sa_roi/sub-XXXX/ses-01/sub-XXXX_ses-01_sa.nii.gz          # cropped short-axis cine image
  sa_segmentation/sub-XXXX/ses-01/sub-XXXX_sa_seg.nii.gz    # segmentation, same grid as sa_roi
  sa_coordinates/sub-XXXX/ses-01/
    sa_axis_affine.tfm      # voxel -> world affine
    sa_axis_transform.tfm   # additional per-subject local-axis realignment
```

Both `.tfm` files are standard linear-transform text files (ITK/ANTs format),
loaded via `nitransforms.linear.Affine.from_filename(...)._matrix` — see
`nimosef.data.preprocessing.get_rwc` / `get_combined_affine` for exactly how
the two are composed into the real-world coordinates the network trains on.

**The step that produces this layout from a raw ACDC download** — ROI
cropping, segmentation resampling onto the ROI grid, and estimating each
subject's axis transform — **is not included in this repository.** If you're
reproducing the pipeline end-to-end on your own data, you'll need an
equivalent step that outputs the layout above; there isn't a script here that
does it for you.

If `--use_roi False` is passed to `preprocess_data`, it instead reads full
(uncropped) images directly from `<root>/sub-XXXX/...` — but still requires
`sa_segmentation` and `sa_coordinates` under `derivatives/`.

---

## 2. Preprocess into per-subject coordinate tables

```bash
python -m nimosef.data.preprocess_data \
  --root /path/to/dataset \
  --use_roi True \
  --patients -1 \
  --save_dir sa_implicit \
  --manifest_dir manifests_nimosef
```

For each subject this writes, under `<root>/derivatives/sa_implicit/sub-XXXX/ses-01/`:

```
sub-XXXX_intensity.parquet
sub-XXXX_segmentation.parquet
sub-XXXX_coords_scaled_local.parquet
sub-XXXX_coords_local.parquet
sub-XXXX_indices.parquet
sub-XXXX_transform.json
sub-XXXX.npz            # same data, packed for fast loading at train time
```

plus a per-subject manifest JSON under `<root>/derivatives/manifests_nimosef/`,
and a dataset-level manifest (`train`/`val`/`test` subject lists + auto- or
user-supplied `bbox`) written alongside it. The train/val/test split here uses
a fixed 70/15/15 ratio (`sklearn.train_test_split`, `random_state=42`); use
`generate_splits.py` below if you need different ratios or a different seed.

---

## 3. (Optional) Regenerate train/val/test splits

`nimosef.data.generate_splits` recomputes only the split file (not the
parquet data), useful if you want different ratios/seed without rerunning
step 2:

```bash
python -m nimosef.data.generate_splits \
  --root /path/to/dataset \
  --split_file /path/to/dataset/derivatives/manifests_nimosef/dataset_manifest.json \
  --split_ratios 0.7 0.15 0.15 \
  --seed 42
```

Add `--use-roi` to match how step 2 was run, `--no-bbox` to skip bounding-box
computation, or `--bbox x y z` to set it manually. Output format:

```json
{
  "train": ["sub-0001", "sub-0002", ...],
  "val": ["sub-0100", ...],
  "test": ["sub-0200", ...],
  "bbox": [128.0, 128.0, 128.0]
}
```

(The `bbox` above is only an example.)

> **Important: `bbox` must match the checkpoint.** The network sees coordinates
> normalized as `(rwc_local - center) / (bbox / 2)`. If `--bbox` is not given,
> it is computed from your own data, which will differ from the one used to
> train the model. A pretrained checkpoint then receives coordinates scaled
> differently along each axis and produces stretched or squashed
> reconstructions. When using the released checkpoints, pass the training
> bbox explicitly, in both `preprocess_data` and `generate_splits`:
>
> ```bash
> --bbox 524.1553327773272 524.1553327773272 120.0
> ```
>
> This is the ACDC training bbox (mm) used for both `checkpoints/nimosef_e`
> and `checkpoints/nimosef_r`; it is also stored under `"bbox"` in each
> checkpoint's `hyperparameters.json`.

---

## 4. Loading the dataset in PyTorch

```python
from nimosef.data.dataset import NiftiDataset, nifti_collate_fn
from torch.utils.data import DataLoader

dataset_manifest = "/path/to/dataset/derivatives/manifests_nimosef/dataset_manifest.json"

train_set = NiftiDataset(dataset_manifest, mode="train")
val_set   = NiftiDataset(dataset_manifest, mode="val")

train_loader = DataLoader(
    train_set,
    batch_size=2,
    shuffle=True,
    collate_fn=nifti_collate_fn,
)

for batch in train_loader:
    rwc, t0_data, tgt_data, t, transform_info, sample_id = batch
    # training loop...
```

`NiftiDataset` reads `dataset_manifest["<mode>"]` (a list of per-subject
manifest paths written in step 2) plus `dataset_manifest["bbox"]`; see
`nimosef/data/dataset.py` for the full constructor signature
(`transform`, `use_local_axis`, `preload_data`).
