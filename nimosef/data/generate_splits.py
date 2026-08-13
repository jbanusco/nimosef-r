#!/usr/bin/env python
import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from nimosef.data.preprocessing import compute_bbox


def generate_split(args):
    # If split exists
    if os.path.exists(args.split_file) and not args.force_overwrite:
        print(f"Split file '{args.split_file}' already exists. Use --force_overwrite to overwrite.")
        return
    elif os.path.exists(args.split_file) and args.force_overwrite:
        print(f"Overwriting existing split file '{args.split_file}'...")
        os.remove(args.split_file)

    # --- patients ---
    if args.use_roi:
        img_root = os.path.join(args.root, "derivatives", "sa_roi")
    else:
        img_root = args.root

    patients = [p for p in os.listdir(img_root) if p.startswith("sub-")]
    if args.number_patients > 0:
        patients = patients[:args.number_patients]

    print(f"Found {len(patients)} patients. Creating split...")

    # --- split ---
    train, temp = train_test_split(
        patients,
        test_size=(1 - args.split_ratios[0]),
        random_state=args.seed
    )
    val, test = train_test_split(
        temp,
        test_size=args.split_ratios[2] / (args.split_ratios[1] + args.split_ratios[2]),
        random_state=args.seed
    )

    split_data = {"train": train, "val": val, "test": test}

    # --- bbox (optional) ---
    if not args.no_bbox:
        if args.bbox is None:
            print("Auto-computing bounding box...")
            coords_root = os.path.join(args.root, "derivatives", "sa_coordinates")
            bbox = compute_bbox(patients, img_root, coords_root, args.use_roi)
        else:
            bbox = np.array(args.bbox)
        split_data["bbox"] = bbox.tolist()
    else:
        split_data["bbox"] = None
        print("Skipping bounding box computation (--no-bbox).")

    # --- save ---
    os.makedirs(os.path.dirname(args.split_file), exist_ok=True)
    with open(args.split_file, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"✅ Split file saved to {args.split_file}")
    for mode in ["train", "val", "test"]:
        print(f"{mode}: {len(split_data[mode])} subjects")


def parse_split_ratios(ratios):
    if len(ratios) > 1:
        return tuple(float(r) for r in ratios)
    try:
        parsed = json.loads(ratios[0])
        return tuple(float(r) for r in parsed)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not parse split_ratios: {ratios}") from e


def get_parser():
    parser = argparse.ArgumentParser(description="Generate train/val/test split for NiftiDataset")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--number_patients", type=int, default=-1,
                        help="Number of patients to include (default: all)")
    parser.add_argument("--split_ratios", type=str, nargs='+', default=[0.7, 0.15, 0.15],
                        help="Ratios for train, val, test split (e.g. 0.7 0.15 0.15 or \"[0.7,0.15,0.15]\")")
    parser.add_argument("--split_file", type=str, default="train_val_test_split.json",
                        help="Path to save the split JSON")
    parser.add_argument("--force_overwrite", action="store_true",
                        help="Force overwrite existing split file")
    parser.add_argument("--use-roi", action="store_true",
                        help="Use ROI images for patient discovery")
    parser.add_argument("--bbox", type=float, nargs=3, default=None,
                        help="Optional bounding box (x y z). If not set, auto-computed.")
    parser.add_argument("--no-bbox", action="store_true",
                        help="Skip bounding box computation entirely (bbox will be null)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.split_ratios = parse_split_ratios(args.split_ratios)
    generate_split(args)


if __name__ == '__main__':
    main()
