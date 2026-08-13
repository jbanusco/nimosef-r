"""
Boxplots of Table 2's segmentation/boundary-tracking metrics (NIMOSEF-E vs.
NIMOSEF-R): per-chamber Dice (native/uncorrected coordinates) and point-cloud
boundary errors (Chamfer, HF95), all 105 subjects.

Reads the same per-subject CSV Table 2 is built from
(calculate_comparison_baseline_vs_muon.py -> baseline_vs_nimosef_comparison_raw.csv)
-- must be run after that script has populated it.

Output: paper/camera_ready/segmentation_dice.png, paper/camera_ready/boundary_errors.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison/baseline_vs_nimosef_comparison_raw.csv"
OUT_DIR = "/path/to/output/paper"

COLORS = {"NIMOSEF-E": "#ff7f00", "NIMOSEF-R": "#33a02c"}


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: raw comparison CSV not found at {CSV_PATH}. "
              f"Run calculate_comparison_baseline_vs_muon.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "DejaVu Sans"
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Dice, per chamber (native/uncorrected coordinates, matches Table 2) ---
    dice_rows = []
    for chamber in ["LV", "MYO", "RV"]:
        for model, col_prefix in [("NIMOSEF-E", "BaseMuon"), ("NIMOSEF-R", "Muon")]:
            for v in df[f"{col_prefix}_{chamber}_Dice_Uncorr"]:
                dice_rows.append({"Model": model, "Chamber": chamber, "Dice": v})
    df_dice = pd.DataFrame(dice_rows)

    fig1, ax1 = plt.subplots(figsize=(6.5, 5))
    sns.boxplot(data=df_dice, x="Chamber", y="Dice", hue="Model",
                palette=COLORS, linewidth=1.2, ax=ax1)
    ax1.set_xlabel("Cardiac Chamber", fontweight="semibold")
    ax1.set_ylabel("Dice Score", fontweight="semibold")
    ax1.set_ylim(0.45, 1.02)
    ax1.legend(title="Model", loc="lower left", fontsize=10)
    plt.tight_layout()
    dice_path = os.path.join(OUT_DIR, "segmentation_dice.png")
    fig1.savefig(dice_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {dice_path}")

    # --- Boundary point-cloud errors (Chamfer, HF95) ---
    pc_rows = []
    for metric in ["Chamfer", "HF95"]:
        for model, col_prefix in [("NIMOSEF-E", "BaseMuon"), ("NIMOSEF-R", "Muon")]:
            for v in df[f"{col_prefix}_{metric}"]:
                pc_rows.append({"Model": model, "Metric": metric, "Error (mm)": v})
    df_pc = pd.DataFrame(pc_rows)

    fig2, ax2 = plt.subplots(figsize=(4.5, 5))
    sns.boxplot(data=df_pc, x="Metric", y="Error (mm)", hue="Model",
                palette=COLORS, linewidth=1.2, ax=ax2)
    ax2.set_xlabel("Metric", fontweight="semibold")
    ax2.set_ylabel("Distance Error (mm)", fontweight="semibold")
    ax2.set_ylim(0, 22)
    ax2.legend(title="Model", loc="upper left")
    plt.tight_layout()
    pc_path = os.path.join(OUT_DIR, "boundary_errors.png")
    fig2.savefig(pc_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {pc_path}")


if __name__ == "__main__":
    main()
