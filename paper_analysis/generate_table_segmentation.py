"""
Generate table_segmentation.tex (Table 2) from the precomputed CSV:
  baseline_vs_nimosef_comparison_raw.csv

This produces a transposed table (methods as rows, metrics as columns)
to reduce vertical height in the paper.

Source data: calculate_comparison_baseline_vs_muon.py -> baseline_vs_nimosef_comparison_raw.csv
Output:      paper/camera_ready/table_segmentation.tex
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def main():
    data_dir = "/path/to/ACDC_dataset/derivatives/nimosef_v2/alignment_comparison"
    paper_dir = "/path/to/output/paper"

    csv_path = os.path.join(data_dir, "baseline_vs_nimosef_comparison_raw.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run calculate_comparison_baseline_vs_muon.py first.")
        return

    df = pd.read_csv(csv_path)
    N = len(df)
    print(f"Loaded {N} subjects from {csv_path}")

    # We only compare NIMOSEF-E (BaseMuon) vs NIMOSEF-R (Muon) for the paper Table 2
    # Metrics: LV/MYO/RV Dice (uncorrected), Chamfer, HF95
    metrics = [
        ("LV Dice",  "BaseMuon_LV_Dice_Uncorr",  "Muon_LV_Dice_Uncorr",  True),
        ("MYO Dice", "BaseMuon_MYO_Dice_Uncorr", "Muon_MYO_Dice_Uncorr", True),
        ("RV Dice",  "BaseMuon_RV_Dice_Uncorr",  "Muon_RV_Dice_Uncorr",  True),
        ("Chamfer (mm)", "BaseMuon_Chamfer", "Muon_Chamfer", False),
        ("HF95 (mm)",    "BaseMuon_HF95",    "Muon_HF95",    False),
    ]

    # Compute stats
    raw_p_values = []
    rows = []
    for label, col_e, col_r, higher_is_better in metrics:
        vals_e = df[col_e].values
        vals_r = df[col_r].values
        _, p = wilcoxon(vals_r, vals_e)
        raw_p_values.append(p)

        rows.append({
            "label": label,
            "e_median": np.median(vals_e),
            "e_iqr": np.percentile(vals_e, [25, 75]),
            "r_median": np.median(vals_r),
            "r_iqr": np.percentile(vals_r, [25, 75]),
            "higher_is_better": higher_is_better,
        })

    # FDR correction
    _, adj_p, _, _ = multipletests(raw_p_values, alpha=0.05, method="fdr_bh")

    for i, row in enumerate(rows):
        row["adj_p"] = adj_p[i]

    # --- Generate TRANSPOSED LaTeX table ---
    # Layout: methods as rows, metrics as columns
    # This saves vertical space (only 2 rows instead of 5+)
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.0pt}",
        f"\\caption{{Segmentation quality (Dice, Median [IQR]) and boundary tracking errors, native (uncorrected) coordinates, vs. nnU-Net reference segmentations, comparing NIMOSEF-E vs. NIMOSEF-R ($N={N}$). Bold indicates statistically significant improvement ($p < 0.05$ after Benjamini-Hochberg FDR correction).}}",
        "\\label{tab:segmentation_results}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{\\textit{Dice (vs. Reference)}} & \\multicolumn{2}{c}{\\textit{Boundary Errors (mm)}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-6}",
        "\\textbf{Method} & \\textbf{LV} & \\textbf{MYO} & \\textbf{RV} & \\textbf{Chamfer} & \\textbf{HF95} \\\\",
        "\\midrule",
    ]

    def fmt(med, iqr, prec=2):
        return f"{med:.{prec}f} \\text{{\\scriptsize [{iqr[0]:.{prec}f},{iqr[1]:.{prec}f}]}}"

    # NIMOSEF-E row
    e_cells = []
    for row in rows:
        e_cells.append(f"${fmt(row['e_median'], row['e_iqr'])}$")
    latex_lines.append("NIMOSEF-E & " + " & ".join(e_cells) + " \\\\")

    # NIMOSEF-R row (bold the winning cells)
    r_cells = []
    for row in rows:
        s = fmt(row["r_median"], row["r_iqr"])
        hib = row["higher_is_better"]
        is_better = (row["r_median"] > row["e_median"]) if hib else (row["r_median"] < row["e_median"])
        if is_better and row["adj_p"] < 0.05:
            r_cells.append(f"$\\bm{{{s}}}$")
        else:
            r_cells.append(f"${s}$")
    latex_lines.append("NIMOSEF-R & " + " & ".join(r_cells) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_path = os.path.join(paper_dir, "table_segmentation.tex")
    with open(table_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"Saved transposed LaTeX Table 2 to {table_path}")

    # Also print for quick inspection
    print("\n--- Generated LaTeX ---")
    print("\n".join(latex_lines))

if __name__ == "__main__":
    main()
