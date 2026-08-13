"""
Boxplots of the four slice-alignment metrics (Original / Rigid Aligned / NIMOSEF-E
/ NIMOSEF-R), one panel per metric, all 105 subjects.

Reads the same cached per-slice CSVs test_slice_alignment.py /
generate_table_slice_alignment.py write to, using the LV+MYO label ([1, 2])
convention -- must be run after generate_table_slice_alignment.py has
populated the cache for label_1_2, or the CSVs won't exist yet.

Output: paper/camera_ready/alignment_boxplots_all4.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CACHE_DIR = "/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison"
OUT_DIR = "/path/to/output/paper"
LABEL_STR = "1_2"  # LV+MYO, matches generate_table_slice_alignment.py's LABEL = [1, 2]

SOURCE_FILES = {
    "Original": "sa_segmentation",
    "Rigid Aligned": "sa_slice_alignment",
    "NIMOSEF-E": "reconstructions_experiment_20260625_152351",
    "NIMOSEF-R": "reconstructions_experiment_20260701_142855",
}
ORDER = ["Original", "Rigid Aligned", "NIMOSEF-E", "NIMOSEF-R"]
COLORS = {"Original": "#999999", "Rigid Aligned": "#4C72B0", "NIMOSEF-E": "#DD8452", "NIMOSEF-R": "#55A868"}
SCATTER_METRICS = {"CoM_diff1", "CoM_diff2"}

METRICS = [
    ("Adjacent Slice Disp. (mm, 90th pct/subj)", "Disp_xy", False),
    ("CoM Z-Continuity 1st Diff (mm, 90th pct/subj)", "CoM_diff1", False),
    ("CoM Z-Continuity 2nd Diff (mm, 90th pct/subj)", "CoM_diff2", False),
    ("Adjacent Contour Dice (mean/subj)", "Contour_Dice", True),
]


def load(name):
    scatter = pd.read_csv(os.path.join(CACHE_DIR, f"{name}_label_{LABEL_STR}_scatter.csv"), dtype={"Subject": str})
    violin = pd.read_csv(os.path.join(CACHE_DIR, f"{name}_label_{LABEL_STR}_violin.csv"), dtype={"Subject": str})
    return scatter, violin


def per_subject(scatter, violin, metric, use_mean):
    df = scatter if metric in SCATTER_METRICS else violin
    g = df[df["Metric"] == metric].groupby("Subject")["Value"]
    return g.mean() if use_mean else g.quantile(0.9)


def main():
    sources = {label: load(fname) for label, fname in SOURCE_FILES.items()}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    for ax, (title, metric, use_mean) in zip(axes, METRICS):
        rows = []
        for method in ORDER:
            vals = per_subject(*sources[method], metric, use_mean)
            rows += [{"Method": method, "Value": v} for v in vals.values]
        df_plot = pd.DataFrame(rows)
        sns.boxplot(data=df_plot, x="Method", y="Value", order=ORDER,
                    palette=[COLORS[m] for m in ORDER], ax=ax, showfliers=True, width=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "alignment_boxplots_all4.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
