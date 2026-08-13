"""
Generate table_slice_alignment.tex (Table 1) from the cached per-slice metrics
computed by test_slice_alignment.py.

Uses label=[1, 2] (LV+MYO, the epicardial contour) for all four metrics,
consistently -- not the whole heart (which would include RV) and not LV
alone; see test_slice_alignment.py for the metric definitions and the
scl_slope rounding note for the Rigid-Aligned (ANTs) source.

Source data: test_slice_alignment.py's visualize_dists(), cached to
  {alignment_comparison}/{source}_label_1_2_{scatter,violin}.csv
Output: paper/camera_ready/table_slice_alignment.tex
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
import test_slice_alignment as tsa

DERIV = "/path/to/ACDC_dataset/derivatives"
CACHE_DIR = os.path.join(DERIV, "nimosef_v2", "alignment_comparison")
PAPER_DIR = "/path/to/output/paper"
LABEL = [1, 2]  # LV + MYO ("full contour" / epicardial boundary), excludes RV

SOURCES = {
    "Original": os.path.join(DERIV, "sa_segmentation"),
    "Rigid Aligned": os.path.join(DERIV, "sa_slice_alignment"),
    "NIMOSEF-R": os.path.join(DERIV, "nimosef_v2", "muon_riemann_model",
                               "reconstructions_experiment_20260701_142855"),
    "NIMOSEF-E": os.path.join(DERIV, "nimosef_v2", "baseline_muon",
                               "reconstructions_experiment_20260625_152351"),
}

METRICS = [
    ("Adj. Slice Disp. (mm)", "Disp_xy", False, False),      # (label, metric, use_mean, higher_is_better)
    ("CoM Z-Cont. 1st Diff", "CoM_diff1", False, False),
    ("CoM Z-Cont. 2nd Diff", "CoM_diff2", False, False),
    ("Adj. Contour Dice", "Contour_Dice", True, True),
]
SCATTER_METRICS = {"CoM_diff1", "CoM_diff2"}


def compute_all(subjects, refresh=True):
    """Run/cache visualize_dists for every source, return {source: (scatter_df, violin_df)}."""
    out = {}
    for name, base_path in SOURCES.items():
        avail = sorted(s.split("-")[1] for s in os.listdir(base_path) if s.startswith("sub-"))
        subs = [s for s in subjects if s in avail]
        print(f"{name}: {len(subs)}/{len(subjects)} requested subjects available")
        scatter, _, _ = tsa.visualize_dists(label=LABEL, subjects=subs, base_path=base_path,
                                             is_png=False, is_scatter=True, is_plot=False, cache_dir=CACHE_DIR)
        _, violin, _ = tsa.visualize_dists(label=LABEL, subjects=subs, base_path=base_path,
                                            is_png=False, is_scatter=False, is_plot=False, cache_dir=CACHE_DIR)
        out[name] = (scatter, violin)
    return out


def per_subject(scatter, violin, metric, use_mean):
    df = scatter if metric in SCATTER_METRICS else violin
    g = df[df["Metric"] == metric].groupby("Subject")["Value"]
    return g.mean() if use_mean else g.quantile(0.9)


def fmt(values, prec):
    med = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    # IQR in \scriptsize (matches table_segmentation.tex's convention) -- narrower
    # than a same-size IQR, needed to fit 4 data columns within \linewidth.
    return f"{med:.{prec}f}\\text{{\\scriptsize [{q25:.{prec}f},{q75:.{prec}f}]}}"


def main():
    # The canonical 105-subject cohort is dataset_manifest.json's "train" list --
    # NOT os.listdir(sa_segmentation), which has 150 subject folders (this dataset
    # derivative includes extra subjects beyond the 105 actually used in the paper).
    manifest = json.load(open(
        "/path/to/ACDC_dataset/derivatives/manifests_nimosef/dataset_manifest.json"))
    orig_subjects = sorted(p.split("sub-")[1].split("_")[0] for p in manifest["train"])
    print(f"Reference (dataset_manifest.json train) subject count: {len(orig_subjects)}")

    data = compute_all(orig_subjects)

    per_method_by_metric = {}
    for label_name, metric, use_mean, higher_better in METRICS:
        per_method = {}
        for name, (scatter, violin) in data.items():
            v = per_subject(scatter, violin, metric, use_mean)
            per_method[name] = v[v.index.isin(orig_subjects)]
        per_method_by_metric[label_name] = (metric, higher_better, per_method)

    # Wilcoxon signed-rank vs Original, FDR-corrected across all (metric, candidate)
    # pairs together (3 candidates x 4 metrics). NIMOSEF-E is numerically ahead of
    # NIMOSEF-R on every alignment metric under this LV+MYO scope and wins the bold
    # mark on all 4 rows -- see the paper's Discussion for why (these metrics have
    # no ground truth to validate against; Fig. 2 is the qualitative check).
    CANDIDATES = ["Rigid Aligned", "NIMOSEF-E", "NIMOSEF-R"]
    raw_p, keys = [], []
    for label_name, (metric, higher_better, per_method) in per_method_by_metric.items():
        orig = per_method["Original"]
        for cand in CANDIDATES:
            common = orig.index.intersection(per_method[cand].index)
            o, c = orig.loc[common].values, per_method[cand].loc[common].values
            _, p = wilcoxon(c, o)
            raw_p.append(p)
            keys.append((label_name, cand, np.median(o), np.median(c), higher_better))
    _, adj_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")

    sig = {}
    for (label_name, cand, mo, mc, hib), p_adj in zip(keys, adj_p):
        better = (mc > mo) if hib else (mc < mo)
        sig[(label_name, cand)] = better and p_adj < 0.05

    # For each metric, bold whichever of {Rigid Aligned, NIMOSEF-E, NIMOSEF-R} is the
    # single best AND significant vs. Original (one method "wins" a given row).
    rows_tex = []
    for label_name, metric, use_mean, higher_better in METRICS:
        _, _, per_method = per_method_by_metric[label_name]
        prec = 3 if "Dice" in label_name else 2
        candidates = {c: np.median(per_method[c]) for c in CANDIDATES if sig[(label_name, c)]}
        winner = None
        if candidates:
            winner = max(candidates, key=candidates.get) if higher_better else min(candidates, key=candidates.get)

        cells = [fmt(per_method["Original"], prec)]
        for cand in CANDIDATES:
            s = fmt(per_method[cand], prec)
            cells.append(f"\\bm{{{s}}}" if cand == winner else s)
        rows_tex.append(f"{label_name} & ${cells[0]}$ & ${cells[1]}$ & ${cells[2]}$ & ${cells[3]}$ \\\\")

    latex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Slice alignment and stack continuity metrics, LV+MYO (Median [IQR], $N=105$; NIMOSEF-E $N=104$, one subject's reconstruction file failed to load). Bold indicates the single best method significantly improving over Original ($p < 0.05$, FDR corrected).}",
        r"\label{tab:slice_alignment_results}",
        # \resizebox guarantees an exact fit within \linewidth (4 numeric
        # columns don't fit at \footnotesize even with tightened \tabcolsep).
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Original} & \textbf{Rigid Aligned} & \textbf{NIMOSEF-E} & \textbf{NIMOSEF-R} \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    out_path = os.path.join(PAPER_DIR, "table_slice_alignment.tex")
    with open(out_path, "w") as f:
        f.write(latex + "\n")
    print(f"\nSaved {out_path}\n")
    print(latex)

    print("\n=== NIMOSEF-E values (for Discussion paragraph, LV+MYO) ===")
    for label_name, (metric, higher_better, per_method) in per_method_by_metric.items():
        prec = 3 if "Dice" in label_name else 2
        print(f"  {label_name}: {fmt(per_method['NIMOSEF-E'], prec)}")


if __name__ == "__main__":
    main()
