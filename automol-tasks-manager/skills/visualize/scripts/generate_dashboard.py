# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.0",
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "jinja2>=3.1",
#   "scipy>=1.10",
# ]
# ///
"""Generate an interactive HTML dashboard from AutoMol evaluation results.

Reads pipeline_state.json and evaluation CSVs, computes derived metrics,
and renders a self-contained HTML file with Plotly.js charts.
"""

import json
import math
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from jinja2 import Template
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Helper: centred moving average (mirrors stat_plotly_util.moving_average)
# ---------------------------------------------------------------------------

def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average with shrinking window at edges."""
    if len(a) < 2 * (n - 1) + 1:
        n = max(1, (len(a) - 1) // 2)
    if n <= 1:
        return a.copy()
    ret = np.cumsum(a, dtype=float)
    reverse = np.cumsum(a[::-1], dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    reverse[n:] = reverse[n:] - reverse[:-n]
    ret[n - 1 : -n + 1] = ret[n - 1 : -n + 1] - a[n - 1 : -n + 1]
    ret[n - 1 : -n + 1] = (ret[n - 1 : -n + 1] + reverse[-n : n - 2 : -1]) / (
        2 * (n - 1) + 1
    )
    for i in range(n - 1):
        ret[i] = reverse[-1 - i]
        for j in range(i):
            ret[i] = ret[i] + a[j]
            ret[-1 - i] = ret[-1 - i] + a[-1 - j]
        ret[i] = ret[i] / (n + i)
        ret[-1 - i] = ret[-1 - i] / (n + i)
    return ret


# ---------------------------------------------------------------------------
# Phase A — Load data
# ---------------------------------------------------------------------------

def load_pipeline_state(state_path: Path) -> dict:
    with open(state_path) as f:
        return json.load(f)


def load_evaluation_csv(csv_path: str, run_folder: Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.is_absolute():
        p = run_folder.parent.parent / p  # relative to repo root
    if not p.exists():
        # try relative to run folder
        p = run_folder / Path(csv_path).name
    return pd.read_csv(p)


def load_train_info(train_info_path: str, run_folder: Path) -> dict | None:
    p = Path(train_info_path)
    if not p.is_absolute():
        p = run_folder.parent.parent / p
    if not p.exists():
        p = run_folder / Path(train_info_path).name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Phase B — Compute derived data
# ---------------------------------------------------------------------------

def compute_regression_data(df: pd.DataFrame, prop: str, metrics_from_state: dict) -> dict:
    """Compute all regression-derived data for a single property."""
    true_col = f"true_{prop}"
    pred_col = f"predicted_{prop}"
    sd_col = f"SD_{prop}"

    y_true = df[true_col].values.astype(float)
    y_pred = df[pred_col].values.astype(float)

    # remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    smiles = df.loc[mask, "Stand_SMILES"].tolist() if "Stand_SMILES" in df.columns else []

    has_sd = sd_col in df.columns
    sd_vals = df.loc[mask, sd_col].values.astype(float).tolist() if has_sd else []

    n = len(y_true)
    residuals = (y_true - y_pred).tolist()
    abs_errors = np.abs(y_true - y_pred)

    # Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pcorr = float(pearsonr(y_true, y_pred)[0]) if n > 2 else 0.0

    # MAE band percentages
    fold1 = round(100 * float(np.sum(abs_errors <= 1 * mae)) / n, 1) if n > 0 else 0
    fold2 = round(100 * float(np.sum(abs_errors <= 2 * mae)) / n, 1) if n > 0 else 0

    # Error histogram
    hist_counts, hist_edges = np.histogram(abs_errors, bins=20)
    hist_bins = ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist()

    # Moving average error (sorted by true value)
    sort_idx = np.argsort(y_true)
    sorted_true = y_true[sort_idx]
    sorted_abs_err = np.abs(y_true[sort_idx] - y_pred[sort_idx])
    window = min(10, max(1, n // 6))
    ma_err = moving_average(sorted_abs_err, window)

    # --- Cutoff scatter (plotly_reg_model_with_cutoff) ---
    cutoff = float(np.median(y_true))
    t_label = y_true > cutoff
    p_label = y_pred > cutoff
    tp_idx = ((p_label) & (t_label)).tolist()
    tn_idx = ((~p_label) & (~t_label)).tolist()
    fp_idx = ((p_label) & (~t_label)).tolist()
    fn_idx = ((~p_label) & (t_label)).tolist()
    n_tp = int(sum(tp_idx)); n_tn = int(sum(tn_idx))
    n_fp = int(sum(fp_idx)); n_fn = int(sum(fn_idx))

    # --- Error bars (plotly_confusion_bars_from_continuos) ---
    n_bins = min(10, max(3, n // 15))
    true_edges = np.linspace(float(y_true.min()), float(y_true.max()), n_bins + 1)
    err_thresholds = [0.5 * mae, 1.0 * mae, 1.5 * mae, 2.0 * mae]
    err_labels = [
        f"<{0.5 * mae:.2f}",
        f"{0.5 * mae:.2f}-{1.0 * mae:.2f}",
        f"{1.0 * mae:.2f}-{1.5 * mae:.2f}",
        f"{1.5 * mae:.2f}-{2.0 * mae:.2f}",
        f">{2.0 * mae:.2f}",
    ]
    true_bin_labels = []
    error_bar_data = {lbl: [] for lbl in err_labels}
    error_bar_counts = []
    for i in range(n_bins):
        lo_b, hi_b = true_edges[i], true_edges[i + 1]
        if i < n_bins - 1:
            mask_bin = (y_true >= lo_b) & (y_true < hi_b)
        else:
            mask_bin = (y_true >= lo_b) & (y_true <= hi_b)
        bin_errs = abs_errors[mask_bin]
        cnt = int(len(bin_errs))
        error_bar_counts.append(cnt)
        true_bin_labels.append(f"{lo_b:.2f}-{hi_b:.2f}")
        if cnt == 0:
            for lbl in err_labels:
                error_bar_data[lbl].append(0)
        else:
            fracs = [
                float(np.sum(bin_errs < err_thresholds[0])) / cnt,
                float(np.sum((bin_errs >= err_thresholds[0]) & (bin_errs < err_thresholds[1]))) / cnt,
                float(np.sum((bin_errs >= err_thresholds[1]) & (bin_errs < err_thresholds[2]))) / cnt,
                float(np.sum((bin_errs >= err_thresholds[2]) & (bin_errs < err_thresholds[3]))) / cnt,
                float(np.sum(bin_errs >= err_thresholds[3])) / cnt,
            ]
            for lbl, frac in zip(err_labels, fracs):
                error_bar_data[lbl].append(round(frac, 4))

    # --- Threshold variation (plotly_acc_pre_for_reg) ---
    tv_cutoffs = np.linspace(float(y_pred.min()), float(y_pred.max()), 50)
    tv_acc, tv_pre, tv_rec, tv_posratio, tv_used = [], [], [], [], []
    for c in tv_cutoffs:
        tl = y_true > c
        pl = y_pred > c
        tp_c = int(np.sum(pl & tl))
        fp_c = int(np.sum(pl & ~tl))
        fn_c = int(np.sum(~pl & tl))
        if (tp_c + fp_c) > 0 and (tp_c + fn_c) > 0:
            tv_used.append(round(float(c), 4))
            tv_acc.append(round(100 * float(np.sum(tl == pl)) / n, 2))
            tv_pre.append(round(100 * tp_c / (tp_c + fp_c), 2))
            tv_rec.append(round(100 * tp_c / (tp_c + fn_c), 2))
            tv_posratio.append(round(100 * float(np.sum(tl)) / n, 2))

    # --- Hit enrichment (plotly_enrichment) ---
    enr_cutoffs = np.linspace(float(y_pred.min()), float(y_pred.max()), 50)
    binary_true = (y_true > cutoff).astype(int)
    n_pos = int(np.sum(binary_true == 1))
    enr_sf, enr_tpf = [], []
    enr_rec_list, enr_pre_list = [], []
    for c in enr_cutoffs:
        pl = (y_pred > c).astype(int)
        tp_e = int(np.sum((pl == 1) & (binary_true == 1)))
        fp_e = int(np.sum((pl == 1) & (binary_true == 0)))
        fn_e = int(np.sum((pl == 0) & (binary_true == 1)))
        s = int(np.sum(pl == 1))
        enr_tpf.append(round(tp_e / n_pos, 4) if n_pos > 0 else 0)
        enr_sf.append(round(s / n, 4))
        if (tp_e + fp_e) > 0 and (tp_e + fn_e) > 0:
            enr_rec_list.append(round(tp_e / (tp_e + fn_e), 4))
            enr_pre_list.append(round(tp_e / (tp_e + fp_e), 4))
    # Sort by SF
    paired = sorted(zip(enr_sf, enr_tpf))
    enr_sf = [p[0] for p in paired]
    enr_tpf = [p[1] for p in paired]
    # Sort PR by recall
    if enr_rec_list:
        pr_paired = sorted(zip(enr_rec_list, enr_pre_list))
        enr_rec_list = [p[0] for p in pr_paired]
        enr_pre_list = [p[1] for p in pr_paired]

    return {
        "property": prop,
        "task_type": "regression",
        "n": n,
        "smiles": smiles,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "has_sd": has_sd,
        "sd_vals": sd_vals,
        "residuals": residuals,
        "metrics": {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Pearson": round(pcorr, 4),
        },
        "mae_value": round(mae, 4),
        "fold1_pct": fold1,
        "fold2_pct": fold2,
        "error_hist": {
            "bins": hist_bins,
            "counts": hist_counts.tolist(),
        },
        "moving_avg": {
            "true_sorted": sorted_true.tolist(),
            "ma_error": ma_err.tolist(),
        },
        "cutoff_scatter": {
            "cutoff": round(cutoff, 4),
            "tp_idx": tp_idx, "tn_idx": tn_idx,
            "fp_idx": fp_idx, "fn_idx": fn_idx,
            "n_tp": n_tp, "n_tn": n_tn, "n_fp": n_fp, "n_fn": n_fn,
            "tp_pct": round(100 * n_tp / n, 1),
            "tn_pct": round(100 * n_tn / n, 1),
            "fp_pct": round(100 * n_fp / n, 1),
            "fn_pct": round(100 * n_fn / n, 1),
        },
        "error_bars": {
            "true_bin_labels": true_bin_labels,
            "err_labels": err_labels,
            "series": error_bar_data,
            "bin_counts": error_bar_counts,
        },
        "threshold_variation": {
            "cutoffs": tv_used,
            "accuracy": tv_acc,
            "precision": tv_pre,
            "recall": tv_rec,
            "positive_ratio": tv_posratio,
        },
        "enrichment": {
            "sf": enr_sf,
            "tpf": enr_tpf,
            "cutoff": round(cutoff, 4),
            "pr_recall": enr_rec_list,
            "pr_precision": enr_pre_list,
        },
    }


def compute_classification_data(df: pd.DataFrame, prop: str, metrics_from_state: dict) -> dict:
    """Compute all classification-derived data for a single property."""
    # Find the right columns — could be Class_{prop} or just {prop}
    true_col = pred_col = None
    for candidate in [f"true_Class_{prop}", f"true_{prop}"]:
        if candidate in df.columns:
            true_col = candidate
            break
    for candidate in [f"predicted_Class_{prop}", f"predicted_{prop}"]:
        if candidate in df.columns:
            pred_col = candidate
            break

    if true_col is None or pred_col is None:
        raise ValueError(
            f"Cannot find true/predicted columns for {prop}. "
            f"Available: {list(df.columns)}"
        )

    y_true = df[true_col].values.astype(int)
    y_pred = df[pred_col].values.astype(int)
    smiles = df["Stand_SMILES"].tolist() if "Stand_SMILES" in df.columns else []

    labels = sorted(set(y_true) | set(y_pred))
    n_classes = len(labels)
    label_names = [str(l) for l in labels]

    # Find probability columns
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    has_proba = len(prob_cols) >= 2
    y_proba = None
    if has_proba:
        y_proba = df[prob_cols].values.astype(float)

    n = len(y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_labels = label_names

    # Classification report
    clf_report = classification_report(
        y_true, y_pred, labels=labels, target_names=label_names, output_dict=True
    )
    accuracy = round(float(clf_report.get("accuracy", accuracy_score(y_true, y_pred))), 4)

    # Build report heatmap data
    report_rows = label_names + ["macro avg", "weighted avg"]
    report_cols = ["precision", "recall", "f1-score"]
    report_z = []
    for row_name in reversed(report_rows):
        z_row = []
        for col_name in report_cols:
            val = clf_report.get(row_name, {}).get(col_name, None)
            z_row.append(round(val, 3) if val is not None else None)
        report_z.append(z_row)

    # ROC curves (per class)
    roc_data = []
    if has_proba:
        for i, label in enumerate(labels):
            col_idx = i if i < y_proba.shape[1] else 0
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, col_idx], pos_label=label)
            roc_auc_val = float(auc(fpr, tpr))
            roc_data.append({
                "label": label_names[i],
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(roc_auc_val, 4),
            })

    # Precision-Recall curves (per class)
    pr_data = []
    if has_proba:
        for i, label in enumerate(labels):
            col_idx = i if i < y_proba.shape[1] else 0
            precision_arr, recall_arr, _ = precision_recall_curve(
                y_true, y_proba[:, col_idx], pos_label=label
            )
            pr_auc_val = float(auc(recall_arr, precision_arr))
            pr_data.append({
                "label": label_names[i],
                "precision": precision_arr.tolist(),
                "recall": recall_arr.tolist(),
                "auc": round(pr_auc_val, 4),
            })

    # Metrics summary
    metrics_summary = {
        "Accuracy": accuracy,
        "F1 (macro)": round(float(clf_report.get("macro avg", {}).get("f1-score", 0)), 4),
        "Precision (macro)": round(float(clf_report.get("macro avg", {}).get("precision", 0)), 4),
        "Recall (macro)": round(float(clf_report.get("macro avg", {}).get("recall", 0)), 4),
    }
    if has_proba and n_classes == 2:
        try:
            metrics_summary["AUC-ROC"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
        except Exception:
            pass

    # --- F1 threshold tuning (plotly_clf_f1) per class ---
    f1_data = []
    if has_proba:
        for i, label in enumerate(labels):
            col_idx = i if i < y_proba.shape[1] else 0
            prec_arr, rec_arr, thresh_arr = precision_recall_curve(
                y_true, y_proba[:, col_idx], pos_label=label
            )
            fscore = np.where(
                (prec_arr[:-1] + rec_arr[:-1]) > 0,
                2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]),
                0,
            )
            best_ix = int(np.nanargmax(fscore))
            f1_data.append({
                "label": label_names[i],
                "thresholds": thresh_arr.tolist(),
                "precision": prec_arr[:-1].tolist(),
                "recall": rec_arr[:-1].tolist(),
                "fscore": fscore.tolist(),
                "best_threshold": round(float(thresh_arr[best_ix]), 4),
                "best_f1": round(float(fscore[best_ix]), 4),
            })

    # --- Calibration / reliability diagram (plotly_clf_calbc) ---
    calib_data = []
    if has_proba:
        prob_cutoffs = np.linspace(0.0, 1.0, 6)
        calib_labels = [
            f"[{prob_cutoffs[i-1]:.2f},{prob_cutoffs[i]:.2f})"
            for i in range(1, len(prob_cutoffs))
        ]
        calib_labels[-1] = calib_labels[-1].replace(")", "]")
        for i, label in enumerate(labels):
            col_idx = i if i < y_proba.shape[1] else 0
            prob_bin_vals, acc_bin_vals, len_bin_vals = [], [], []
            df_cal = pd.DataFrame({"prob": y_proba[:, col_idx], "y_true": y_true})
            df_cal["bins"] = pd.cut(df_cal["prob"], bins=prob_cutoffs, include_lowest=True)
            for _, grp in df_cal.groupby("bins"):
                if len(grp) < 1:
                    continue
                acc_bin_vals.append(round(float(np.sum(grp["y_true"] == label) / len(grp)), 4))
                prob_bin_vals.append(round(float(grp["prob"].mean()), 4))
                len_bin_vals.append(int(len(grp)))
            ece = 0.0
            mce = 0.0
            if acc_bin_vals and prob_bin_vals:
                diffs = [abs(a - p) for a, p in zip(acc_bin_vals, prob_bin_vals)]
                total = sum(len_bin_vals)
                ece = round(sum(d * l for d, l in zip(diffs, len_bin_vals)) / total, 4) if total > 0 else 0
                mce = round(max(diffs), 4) if diffs else 0
            calib_data.append({
                "label": label_names[i],
                "prob_bins": prob_bin_vals,
                "acc_bins": acc_bin_vals,
                "bin_sizes": len_bin_vals,
                "ece": ece,
                "mce": mce,
            })

    # --- Hit enrichment for classification (plotly_enrichment_clf) ---
    enr_clf_data = []
    if has_proba:
        for i, label in enumerate(labels):
            col_idx = i if i < y_proba.shape[1] else 0
            n_pos_c = int(np.sum(y_true == label))
            if n_pos_c == 0:
                continue
            sf_list, tpf_list = [], []
            cutoffs_e = np.linspace(0, 1, 50)
            for c in cutoffs_e:
                pl = y_proba[:, col_idx] > c
                tp_e = int(np.sum(pl & (y_true == label)))
                s = int(np.sum(pl))
                tpf_list.append(round(tp_e / n_pos_c, 4))
                sf_list.append(round(s / n, 4))
            paired_e = sorted(zip(sf_list, tpf_list))
            enr_clf_data.append({
                "label": label_names[i],
                "sf": [p[0] for p in paired_e],
                "tpf": [p[1] for p in paired_e],
            })

    # --- Probability bars per class (plotly_confusion_bars_from_categories) ---
    prob_bars_data = None
    if has_proba:
        n_pbins = min(10, max(3, n // 15))
        # Use first class probability for binning
        pbin_edges = np.linspace(0, 1, n_pbins + 1)
        pbin_labels = [f"{pbin_edges[i]:.2f}-{pbin_edges[i+1]:.2f}" for i in range(n_pbins)]
        prob_bars_series = {}
        prob_bars_counts = []
        for label_val in labels:
            mask_cls = y_true == label_val
            cnt = int(np.sum(mask_cls))
            prob_bars_counts.append(cnt)
            if cnt == 0:
                for pl in pbin_labels:
                    prob_bars_series.setdefault(pl, []).append(0)
            else:
                # Use the probability for this class
                col_idx_pb = labels.index(label_val) if labels.index(label_val) < y_proba.shape[1] else 0
                cls_probs = y_proba[mask_cls, col_idx_pb]
                for j in range(n_pbins):
                    lo_b, hi_b = pbin_edges[j], pbin_edges[j + 1]
                    if j < n_pbins - 1:
                        frac = float(np.sum((cls_probs >= lo_b) & (cls_probs < hi_b))) / cnt
                    else:
                        frac = float(np.sum((cls_probs >= lo_b) & (cls_probs <= hi_b))) / cnt
                    prob_bars_series.setdefault(pbin_labels[j], []).append(round(frac, 4))
        prob_bars_data = {
            "class_labels": label_names,
            "prob_bin_labels": pbin_labels,
            "series": prob_bars_series,
            "class_counts": prob_bars_counts,
        }

    return {
        "property": prop,
        "task_type": "classification",
        "n": n,
        "smiles": smiles,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "labels": labels,
        "label_names": label_names,
        "metrics": metrics_summary,
        "confusion_matrix": {
            "z": cm.tolist(),
            "labels": cm_labels,
            "accuracy": accuracy,
        },
        "classification_report": {
            "row_labels": list(reversed(report_rows)),
            "col_labels": report_cols,
            "z": report_z,
            "accuracy": accuracy,
        },
        "roc_curves": roc_data,
        "pr_curves": pr_data,
        "f1_threshold": f1_data,
        "calibration": calib_data,
        "enrichment_clf": enr_clf_data,
        "prob_bars": prob_bars_data,
    }


# ---------------------------------------------------------------------------
# Phase C — Render HTML
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MolAgent Dashboard — {{ run_id }}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://unpkg.com/smiles-drawer@2.1.7/dist/smiles-drawer.min.js"></script>
<style>
  :root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --surface2: #0f3460;
    --accent: #e94560;
    --accent2: #533483;
    --text: #eee;
    --text-muted: #999;
    --border: #2a2a4a;
    --success: #4ecca3;
    --warning: #ffc107;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }
  .header h1 {
    font-size: 1.4em;
    font-weight: 600;
    background: linear-gradient(90deg, var(--accent), #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header .run-id {
    color: var(--text-muted);
    font-size: 0.9em;
  }
  .controls {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    padding: 16px 32px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .controls label {
    color: var(--text-muted);
    font-size: 0.85em;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  select, .btn {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.9em;
    cursor: pointer;
    transition: all 0.15s;
  }
  select:hover, .btn:hover {
    border-color: var(--accent);
  }
  .btn.active {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }
  .btn-group { display: flex; gap: 4px; flex-wrap: wrap; }
  .metrics-bar {
    display: flex;
    gap: 16px;
    padding: 16px 32px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .metric-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 20px;
    min-width: 120px;
    text-align: center;
  }
  .metric-card .value {
    font-size: 1.5em;
    font-weight: 700;
    color: var(--success);
  }
  .metric-card .label {
    font-size: 0.75em;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
  }
  .main-content {
    padding: 24px 32px;
  }
  .plot-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: visible;
    margin-bottom: 20px;
  }
  .plot-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .plot-grid.single {
    grid-template-columns: 1fr;
  }
  .plot-title-bar {
    padding: 12px 20px;
    background: rgba(0,0,0,0.2);
    font-size: 0.85em;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .prompt-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-top: 24px;
  }
  .prompt-section h3 {
    font-size: 0.9em;
    color: var(--text-muted);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .prompt-output {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.85em;
    line-height: 1.6;
    white-space: pre-wrap;
    position: relative;
    max-height: 400px;
    overflow-y: auto;
  }
  .copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: var(--surface2);
    color: var(--text-muted);
    border: 1px solid var(--border);
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.8em;
    cursor: pointer;
    transition: all 0.15s;
  }
  .copy-btn:hover {
    color: var(--text);
    border-color: var(--accent);
  }
  .copy-btn.copied {
    color: var(--success);
    border-color: var(--success);
  }
  .preset-group { margin-left: auto; }
  /* Molecule tooltip */
  #molTooltip {
    display: none; position: fixed; z-index: 9999;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px; pointer-events: none;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    max-width: 280px;
  }
  #molTooltip svg { display: block; margin: 0 auto 6px; border-radius: 6px; background: #fff; width: 220px; height: 150px; }
  #molTooltip .tt-smiles { font-size: 0.7em; color: var(--text-muted); word-break: break-all; margin-bottom: 4px; }
  #molTooltip .tt-vals { font-size: 0.82em; }
  #molTooltip .tt-vals span { display: block; }
  /* Slider controls */
  .slider-bar {
    display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
    padding: 10px 32px; background: var(--surface); border-bottom: 1px solid var(--border);
  }
  .slider-group { display: flex; align-items: center; gap: 8px; }
  .slider-group label { color: var(--text-muted); font-size: 0.82em; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; }
  .slider-group input[type=range] { width: 180px; accent-color: var(--accent); cursor: pointer; }
  .slider-val { color: var(--success); font-weight: 600; font-size: 0.9em; min-width: 52px; text-align: center; }
  @media (max-width: 900px) {
    .plot-grid { grid-template-columns: 1fr; }
    .controls, .metrics-bar { padding: 12px 16px; }
    .main-content { padding: 16px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>MolAgent Dashboard</h1>
  <span class="run-id">{{ run_id }} &mdash; {{ task_type }}</span>
</div>

<div class="controls">
  <label>Property</label>
  <select id="propSelect" onchange="switchProperty()">
    {% for p in properties %}
    <option value="{{ p }}">{{ p }}</option>
    {% endfor %}
  </select>

  <label style="margin-left:16px">View</label>
  <div class="btn-group" id="plotTypeGroup"></div>

  <div class="preset-group">
    <div class="btn-group">
      <button class="btn active" data-preset="overview" onclick="setPreset('overview')">Overview</button>
      <button class="btn" data-preset="detailed" onclick="setPreset('detailed')">Detailed</button>
    </div>
  </div>
</div>

<div class="metrics-bar" id="metricsBar"></div>

<div class="slider-bar" id="sliderBar" style="display:none">
  <div class="slider-group">
    <label>Cutoff</label>
    <input type="range" id="cutoffSlider" min="0" max="1" step="0.01" value="0.5">
    <span class="slider-val" id="cutoffVal">0.50</span>
  </div>
  <div class="slider-group">
    <label>Error bins</label>
    <input type="range" id="binsSlider" min="3" max="20" step="1" value="8">
    <span class="slider-val" id="binsVal">8</span>
  </div>
</div>

<div id="molTooltip">
  <svg id="molSvg" xmlns="http://www.w3.org/2000/svg" width="220" height="150"></svg>
  <div class="tt-smiles" id="ttSmiles"></div>
  <div class="tt-vals" id="ttVals"></div>
</div>

<div class="main-content">
  <div id="plotArea" class="plot-grid"></div>

  <div class="prompt-section">
    <h3>Findings</h3>
    <div class="prompt-output" id="promptOutput">
      <button class="copy-btn" onclick="copyFindings()">Copy</button>
      <span id="findingsText"></span>
    </div>
  </div>
</div>

<script>
const DATA = {{ data_json }};
const TASK_TYPE = DATA._task_type;

// ── Dark plotly layout ──────────────────────────────────────────
const DARK_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: '#111827',
  font: { color: '#ccc', size: 13 },
  margin: { t: 50, r: 30, b: 60, l: 60 },
  xaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
  yaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
  legend: {
    bgcolor: 'rgba(0,0,0,0.3)',
    font: { color: '#ccc', size: 11 }
  },
  hoverlabel: { font: { size: 11 } }
};

// ── SmilesDrawer + Tooltip ───────────────────────────────────────
let smiDrawerReady = false;
let smiDrawerInstance = null;
try {
  smiDrawerInstance = new SmilesDrawer.SmiDrawer({ width: 220, height: 150 });
  smiDrawerReady = true;
} catch(e) {
  try {
    smiDrawerInstance = new SmiDrawer({ width: 220, height: 150 });
    smiDrawerReady = true;
  } catch(e2) { console.warn('SmilesDrawer not available', e, e2); }
}

const tooltip = document.getElementById('molTooltip');
const ttSvg = document.getElementById('molSvg');
const ttSmiles = document.getElementById('ttSmiles');
const ttVals = document.getElementById('ttVals');
let lastDrawnSmiles = '';

function showTooltip(mouseEvt, smiles, trueVal, predVal, extra) {
  if (!smiDrawerReady || !smiles) return;
  tooltip.style.display = 'block';
  const ox = mouseEvt.clientX || mouseEvt.pageX || 200;
  const oy = mouseEvt.clientY || mouseEvt.pageY || 200;
  const ttW = 280, ttH = 260;
  const left = ox + ttW + 30 > window.innerWidth ? ox - ttW - 10 : ox + 18;
  const top = Math.max(8, Math.min(oy - 60, window.innerHeight - ttH - 8));
  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
  ttSmiles.textContent = smiles;
  let html = `<span><b>True:</b> ${Number(trueVal).toFixed(3)}</span><span><b>Predicted:</b> ${Number(predVal).toFixed(3)}</span>`;
  if (extra) html += `<span>${extra}</span>`;
  ttVals.innerHTML = html;
  if (smiles !== lastDrawnSmiles) {
    lastDrawnSmiles = smiles;
    ttSvg.innerHTML = '';
    try {
      smiDrawerInstance.draw(smiles, ttSvg, 'light');
    } catch(e) { /* parse/draw error */ }
  }
}

function hideTooltip() {
  tooltip.style.display = 'none';
  lastDrawnSmiles = '';
}

// Attach hover handlers to a scatter-plot div (replaces Plotly default hover)
function attachHover(div, d) {
  const sm = d.smiles || [];
  const yt = d.y_true || [];
  const yp = d.y_pred || [];
  if (sm.length === 0) return;
  div.on('plotly_hover', function(evtData) {
    const pt = evtData.points[0];
    const idx = pt.customdata != null ? pt.customdata : pt.pointIndex;
    if (idx != null && sm[idx]) {
      const err = Math.abs(yt[idx] - yp[idx]);
      showTooltip(evtData.event, sm[idx], yt[idx], yp[idx], `<b>|Error|:</b> ${err.toFixed(3)}`);
    }
  });
  div.on('plotly_unhover', hideTooltip);
}

// ── Slider state ────────────────────────────────────────────────
let currentCutoff = 0;
let currentBins = 8;

function initSliders(d) {
  const bar = document.getElementById('sliderBar');
  if (TASK_TYPE !== 'regression') { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';
  const lo = Math.min(...d.y_true);
  const hi = Math.max(...d.y_true);
  const cutoffEl = document.getElementById('cutoffSlider');
  cutoffEl.min = lo.toFixed(2);
  cutoffEl.max = hi.toFixed(2);
  cutoffEl.step = ((hi - lo) / 200).toFixed(4);
  const median = d.cutoff_scatter ? d.cutoff_scatter.cutoff : (lo + hi) / 2;
  cutoffEl.value = median;
  currentCutoff = median;
  document.getElementById('cutoffVal').textContent = median.toFixed(2);
  const binsEl = document.getElementById('binsSlider');
  currentBins = d.error_bars ? d.error_bars.true_bin_labels.length : 8;
  binsEl.value = currentBins;
  document.getElementById('binsVal').textContent = currentBins;
  cutoffEl.oninput = function() {
    currentCutoff = parseFloat(this.value);
    document.getElementById('cutoffVal').textContent = currentCutoff.toFixed(2);
    recomputeAndRender();
  };
  binsEl.oninput = function() {
    currentBins = parseInt(this.value);
    document.getElementById('binsVal').textContent = currentBins;
    recomputeAndRender();
  };
}

function recomputeAndRender() {
  const d = DATA[currentProp];
  recomputeCutoffData(d, currentCutoff);
  recomputeErrorBars(d, currentBins);
  renderPlots(d);
  renderFindings(d);
}

// ── JS-side recomputation (cutoff) ──────────────────────────────
function recomputeCutoffData(d, cutoff) {
  const yt = d.y_true, yp = d.y_pred, n = yt.length;
  const tp = [], tn = [], fp = [], fn = [];
  for (let i = 0; i < n; i++) {
    const tl = yt[i] > cutoff, pl = yp[i] > cutoff;
    tp.push(tl && pl); tn.push(!tl && !pl); fp.push(!tl && pl); fn.push(tl && !pl);
  }
  const ntp = tp.filter(Boolean).length, ntn = tn.filter(Boolean).length;
  const nfp = fp.filter(Boolean).length, nfn = fn.filter(Boolean).length;
  d.cutoff_scatter = {
    cutoff, tp_idx: tp, tn_idx: tn, fp_idx: fp, fn_idx: fn,
    n_tp: ntp, n_tn: ntn, n_fp: nfp, n_fn: nfn,
    tp_pct: +(100*ntp/n).toFixed(1), tn_pct: +(100*ntn/n).toFixed(1),
    fp_pct: +(100*nfp/n).toFixed(1), fn_pct: +(100*nfn/n).toFixed(1),
  };
  // Enrichment
  const binTrue = yt.map(v => v > cutoff ? 1 : 0);
  const nPos = binTrue.reduce((s,v) => s+v, 0);
  const predMin = Math.min(...yp), predMax = Math.max(...yp);
  const cuts = Array.from({length:50}, (_,i) => predMin + i*(predMax-predMin)/49);
  const sf = [], tpf = [];
  cuts.forEach(c => {
    let tpe = 0, s = 0;
    for (let i = 0; i < n; i++) { if (yp[i] > c) { s++; if (binTrue[i]) tpe++; } }
    sf.push(s/n); tpf.push(nPos > 0 ? tpe/nPos : 0);
  });
  const paired = sf.map((s,i) => [s,tpf[i]]).sort((a,b) => a[0]-b[0]);
  d.enrichment = { sf: paired.map(p=>p[0]), tpf: paired.map(p=>p[1]), cutoff, pr_recall: d.enrichment.pr_recall, pr_precision: d.enrichment.pr_precision };
  // Threshold variation
  const tvCuts = [], tvAcc = [], tvPre = [], tvRec = [], tvPos = [];
  cuts.forEach(c => {
    let tpe=0,fpe=0,fne=0,tl_sum=0,eq=0;
    for (let i=0;i<n;i++) { const tl=yt[i]>c,pl=yp[i]>c; if(tl)tl_sum++; if(tl===pl)eq++; if(pl&&tl)tpe++; if(pl&&!tl)fpe++; if(!pl&&tl)fne++; }
    if ((tpe+fpe)>0 && (tpe+fne)>0) {
      tvCuts.push(+c.toFixed(4)); tvAcc.push(+(100*eq/n).toFixed(2));
      tvPre.push(+(100*tpe/(tpe+fpe)).toFixed(2)); tvRec.push(+(100*tpe/(tpe+fne)).toFixed(2));
      tvPos.push(+(100*tl_sum/n).toFixed(2));
    }
  });
  d.threshold_variation = { cutoffs: tvCuts, accuracy: tvAcc, precision: tvPre, recall: tvRec, positive_ratio: tvPos };
}

// ── JS-side recomputation (error bins) ──────────────────────────
function recomputeErrorBars(d, nBins) {
  const yt = d.y_true, yp = d.y_pred, mae = d.mae_value, n = yt.length;
  const absErr = yt.map((v,i) => Math.abs(v - yp[i]));
  const lo = Math.min(...yt), hi = Math.max(...yt);
  const edges = Array.from({length: nBins+1}, (_,i) => lo + i*(hi-lo)/nBins);
  const thresholds = [0.5*mae, 1.0*mae, 1.5*mae, 2.0*mae];
  const errLabels = [
    '<'+(0.5*mae).toFixed(2),
    (0.5*mae).toFixed(2)+'-'+(1.0*mae).toFixed(2),
    (1.0*mae).toFixed(2)+'-'+(1.5*mae).toFixed(2),
    (1.5*mae).toFixed(2)+'-'+(2.0*mae).toFixed(2),
    '>'+(2.0*mae).toFixed(2),
  ];
  const trueBinLabels = [], series = {}, binCounts = [];
  errLabels.forEach(l => series[l] = []);
  for (let b = 0; b < nBins; b++) {
    const loB = edges[b], hiB = edges[b+1];
    trueBinLabels.push(loB.toFixed(2)+'-'+hiB.toFixed(2));
    const idxs = [];
    for (let i = 0; i < n; i++) {
      if (b < nBins-1 ? (yt[i] >= loB && yt[i] < hiB) : (yt[i] >= loB && yt[i] <= hiB)) idxs.push(i);
    }
    const cnt = idxs.length; binCounts.push(cnt);
    const bErr = idxs.map(i => absErr[i]);
    if (cnt === 0) { errLabels.forEach(l => series[l].push(0)); }
    else {
      const fracs = [
        bErr.filter(e => e < thresholds[0]).length / cnt,
        bErr.filter(e => e >= thresholds[0] && e < thresholds[1]).length / cnt,
        bErr.filter(e => e >= thresholds[1] && e < thresholds[2]).length / cnt,
        bErr.filter(e => e >= thresholds[2] && e < thresholds[3]).length / cnt,
        bErr.filter(e => e >= thresholds[3]).length / cnt,
      ];
      errLabels.forEach((l,i) => series[l].push(+fracs[i].toFixed(4)));
    }
  }
  d.error_bars = { true_bin_labels: trueBinLabels, err_labels: errLabels, series, bin_counts: binCounts };
}

// ── Plot registry ───────────────────────────────────────────────
const REG_PLOTS = ['scatter_mae', 'scatter_ma', 'scatter_cutoff', 'residuals', 'error_hist', 'error_bars', 'threshold_var', 'enrichment'];
const CLF_PLOTS = ['confusion', 'roc', 'pr', 'clf_report', 'f1_threshold', 'calibration', 'enrichment_clf', 'prob_bars'];

const PLOT_LABELS = {
  scatter_mae: 'Scatter + MAE Bands',
  residuals: 'Residual Plot',
  error_hist: 'Error Distribution',
  scatter_ma: 'Moving Avg Error',
  scatter_cutoff: 'Cutoff Scatter',
  error_bars: 'Error Barplot',
  threshold_var: 'Threshold Variation',
  enrichment: 'Hit Enrichment',
  confusion: 'Confusion Matrix',
  roc: 'ROC Curves',
  pr: 'Precision-Recall',
  clf_report: 'Classification Report',
  f1_threshold: 'F1 Threshold',
  calibration: 'Calibration',
  enrichment_clf: 'Hit Enrichment',
  prob_bars: 'Probability Bars',
};

let currentProp = DATA._properties[0];
let currentPreset = 'overview';
let currentPlotType = null;

// ── Initialise controls ─────────────────────────────────────────
function init() {
  const plots = TASK_TYPE === 'regression' ? REG_PLOTS : CLF_PLOTS;
  const group = document.getElementById('plotTypeGroup');
  plots.forEach((p, i) => {
    const btn = document.createElement('button');
    btn.className = 'btn' + (i === 0 ? ' active' : '');
    btn.textContent = PLOT_LABELS[p];
    btn.dataset.plot = p;
    btn.onclick = () => selectPlotType(p);
    group.appendChild(btn);
  });
  currentPlotType = plots[0];
  initSliders(DATA[currentProp]);
  render();
}

function switchProperty() {
  currentProp = document.getElementById('propSelect').value;
  initSliders(DATA[currentProp]);
  render();
}

function selectPlotType(plotType) {
  currentPlotType = plotType;
  document.querySelectorAll('#plotTypeGroup .btn').forEach(b => {
    b.classList.toggle('active', b.dataset.plot === plotType);
  });
  // If in detailed mode, re-render
  if (currentPreset === 'detailed') render();
  // If in overview, switch to detailed
  if (currentPreset === 'overview') {
    setPreset('detailed');
  }
}

function setPreset(preset, el) {
  currentPreset = preset;
  document.querySelectorAll('.preset-group .btn').forEach(b => {
    b.classList.toggle('active', b.dataset.preset === preset);
  });
  render();
}

// ── Render ──────────────────────────────────────────────────────
function render() {
  const d = DATA[currentProp];
  renderMetrics(d);
  renderPlots(d);
  renderFindings(d);
}

function renderMetrics(d) {
  const bar = document.getElementById('metricsBar');
  bar.innerHTML = '';
  const metrics = d.metrics;
  Object.entries(metrics).forEach(([k, v]) => {
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `<div class="value">${v}</div><div class="label">${k}</div>`;
    bar.appendChild(card);
  });
  // Add n
  const nCard = document.createElement('div');
  nCard.className = 'metric-card';
  nCard.innerHTML = `<div class="value">${d.n}</div><div class="label">Test Points</div>`;
  bar.appendChild(nCard);
}

const SCATTER_PLOTS = new Set(['scatter_mae', 'scatter_ma', 'scatter_cutoff', 'residuals']);

function renderPlots(d) {
  const area = document.getElementById('plotArea');
  area.innerHTML = '';

  const plots = TASK_TYPE === 'regression' ? REG_PLOTS : CLF_PLOTS;

  if (currentPreset === 'overview') {
    area.className = 'plot-grid';
    plots.forEach(p => {
      const container = makeContainer(PLOT_LABELS[p], 480);
      area.appendChild(container);
      const div = container.querySelector('.plot-div');
      renderSinglePlot(div, p, d);
      if (SCATTER_PLOTS.has(p)) attachHover(div, d);
    });
  } else {
    area.className = 'plot-grid single';
    const container = makeContainer(PLOT_LABELS[currentPlotType], 720);
    area.appendChild(container);
    const div = container.querySelector('.plot-div');
    renderSinglePlot(div, currentPlotType, d, true);
    if (SCATTER_PLOTS.has(currentPlotType)) attachHover(div, d);
  }
}

function makeContainer(title, height) {
  const c = document.createElement('div');
  c.className = 'plot-container';
  c.innerHTML = `<div class="plot-title-bar">${title}</div><div class="plot-div" style="width:100%;height:${height}px"></div>`;
  return c;
}

// ── Regression plots ────────────────────────────────────────────
function plotScatterMAE(div, d, large) {
  const { y_true, y_pred, smiles, mae_value, fold1_pct, fold2_pct } = d;
  const lo = Math.min(...y_true, ...y_pred) - 0.1;
  const hi = Math.max(...y_true, ...y_pred) + 0.1;
  const line = [lo, hi];
  const mae = mae_value;

  const traces = [
    // Data points
    {
      x: y_true, y: y_pred,
      mode: 'markers', type: 'scatter',
      name: 'Test samples',
      customdata: y_true.map((_,i) => i),
      hoverinfo: 'none',
      marker: { color: 'rgba(233,69,96,0.6)', size: 6, line: { width: 0.5, color: 'rgba(233,69,96,0.9)' } }
    },
    // +2 MAE
    { x: line, y: line.map(v => v + 2*mae), mode: 'lines', showlegend: false, line: { color: 'rgba(78,204,163,0.3)', width: 1 } },
    // +1 MAE (fill to +2)
    { x: line, y: line.map(v => v + 1*mae), mode: 'lines', showlegend: false, fill: 'tonexty', line: { color: 'rgba(78,204,163,0.4)', width: 1 }, fillcolor: 'rgba(78,204,163,0.1)' },
    // Identity (fill to +1)
    { x: line, y: line, mode: 'lines', showlegend: false, fill: 'tonexty', line: { color: 'rgba(78,204,163,0.6)', width: 1.5, dash: 'dash' }, fillcolor: 'rgba(78,204,163,0.2)' },
    // -1 MAE (fill to identity)
    { x: line, y: line.map(v => v - 1*mae), mode: 'lines', name: `${fold1_pct}% within 1x MAE`, fill: 'tonexty', line: { color: 'rgba(78,204,163,0.4)', width: 1 }, fillcolor: 'rgba(78,204,163,0.2)' },
    // -2 MAE (fill to -1)
    { x: line, y: line.map(v => v - 2*mae), mode: 'lines', name: `${fold2_pct}% within 2x MAE`, fill: 'tonexty', line: { color: 'rgba(78,204,163,0.3)', width: 1 }, fillcolor: 'rgba(78,204,163,0.1)' },
  ];

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Scatter + MAE Bands — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'True Value' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Predicted Value' },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

function plotResiduals(div, d, large) {
  const { y_true, y_pred, smiles, residuals } = d;

  const traces = [{
    x: y_true, y: residuals,
    mode: 'markers', type: 'scatter',
    name: 'Residuals',
    customdata: y_true.map((_,i) => i),
    hoverinfo: 'none',
    marker: { color: 'rgba(83,52,131,0.7)', size: 6, line: { width: 0.5, color: 'rgba(83,52,131,1)' } }
  }, {
    x: [Math.min(...y_true), Math.max(...y_true)],
    y: [0, 0],
    mode: 'lines', showlegend: false,
    line: { color: 'rgba(233,69,96,0.8)', width: 1.5, dash: 'dash' }
  }];

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Residual Plot — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'True Value' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Residual (True - Predicted)' },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

function plotErrorHist(div, d, large) {
  const { error_hist } = d;
  const traces = [{
    x: error_hist.bins, y: error_hist.counts,
    type: 'bar',
    marker: { color: 'rgba(78,204,163,0.7)', line: { color: 'rgba(78,204,163,1)', width: 1 } }
  }];

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Absolute Error Distribution — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Absolute Error' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Count' },
    height: large ? 700 : 460,
    bargap: 0.05,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

function plotScatterMA(div, d, large) {
  const { y_true, y_pred, smiles, moving_avg, mae_value } = d;
  const ma = moving_avg;
  const lo = Math.min(...y_true, ...y_pred) - 0.1;
  const hi = Math.max(...y_true, ...y_pred) + 0.1;

  const traces = [
    {
      x: y_true, y: y_pred,
      mode: 'markers', type: 'scatter',
      name: 'Test samples',
      customdata: y_true.map((_,i) => i),
      hoverinfo: 'none',
      marker: { color: 'rgba(233,69,96,0.5)', size: 5, line: { width: 0.5, color: 'rgba(233,69,96,0.8)' } }
    },
    // Identity line
    { x: [lo, hi], y: [lo, hi], mode: 'lines', showlegend: false, line: { color: 'rgba(78,204,163,0.6)', width: 1.5, dash: 'dash' } },
    // Moving average error band
    {
      x: ma.true_sorted, y: ma.true_sorted.map((v, i) => v - ma.ma_error[i]),
      mode: 'lines', name: 'Moving Avg Error',
      line: { color: 'rgba(100,149,237,0.8)', width: 2 }
    },
    {
      x: ma.true_sorted, y: ma.true_sorted.map((v, i) => v + ma.ma_error[i]),
      mode: 'lines', showlegend: false,
      line: { color: 'rgba(100,149,237,0.8)', width: 2 }
    },
  ];

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Scatter + Moving Avg Error — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'True Value' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Predicted Value' },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Regression: Cutoff Scatter (plotly_reg_model_with_cutoff) ────
function plotScatterCutoff(div, d, large) {
  const { y_true, y_pred, smiles, cutoff_scatter: cs } = d;
  const cutoff = cs.cutoff;
  // Build index arrays
  const groups = [
    { idx: cs.tp_idx, name: `True Pos: ${cs.n_tp} (${cs.tp_pct}%)`, color: 'rgba(0,100,0,0.6)' },
    { idx: cs.tn_idx, name: `True Neg: ${cs.n_tn} (${cs.tn_pct}%)`, color: 'rgba(233,69,96,0.6)' },
    { idx: cs.fp_idx, name: `False Pos: ${cs.n_fp} (${cs.fp_pct}%)`, color: 'rgba(0,191,255,0.6)' },
    { idx: cs.fn_idx, name: `False Neg: ${cs.n_fn} (${cs.fn_pct}%)`, color: 'rgba(255,213,0,0.6)' },
  ];
  const traces = groups.map(g => {
    const xi = [], yi = [], ti = [], ci = [];
    g.idx.forEach((v, i) => { if (v) { xi.push(y_true[i]); yi.push(y_pred[i]); ti.push(smiles[i] || ''); ci.push(i); } });
    return {
      x: xi, y: yi, customdata: ci, mode: 'markers', type: 'scatter', name: g.name,
      hoverinfo: 'none',
      marker: { color: g.color, size: 6, line: { width: 0.5 } }
    };
  });
  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Cutoff Scatter (cutoff=${cutoff}) — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'True Value' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Predicted Value' },
    height: large ? 700 : 460,
    shapes: [
      { type: 'line', x0: cutoff, x1: cutoff, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(112,128,144,0.8)', width: 2, dash: 'dash' } },
      { type: 'line', y0: cutoff, y1: cutoff, x0: 0, x1: 1, xref: 'paper', line: { color: 'rgba(112,128,144,0.8)', width: 2, dash: 'dash' } },
    ],
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Regression: Error Barplot (plotly_confusion_bars_from_continuos) ─
function plotErrorBars(div, d, large) {
  const eb = d.error_bars;
  const colors = ['rgba(0,100,0,0.8)', 'rgba(0,100,0,0.5)', 'rgba(255,213,0,0.6)', 'rgba(233,69,96,0.5)', 'rgba(233,69,96,0.8)'];
  const traces = eb.err_labels.map((lbl, i) => ({
    x: eb.true_bin_labels, y: eb.series[lbl], name: lbl, type: 'bar',
    marker: { color: colors[i % colors.length] },
    text: eb.series[lbl].map(v => v > 0.05 ? v.toFixed(2) : ''),
    textposition: 'inside', textfont: { size: 10 },
    hovertemplate: '<b>Bin</b>: %{x}<br><b>Fraction</b>: %{y:.2f}<br><b>Error range</b>: ' + lbl,
  }));
  const layout = {
    ...DARK_LAYOUT, barmode: 'stack',
    title: { text: `Absolute Error Barplot — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'True Value Bins', type: 'category' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Relative Counts', range: [0, 1.08] },
    height: large ? 700 : 460,
    annotations: eb.bin_counts.map((cnt, i) => ({
      x: eb.true_bin_labels[i], y: 1.0, text: String(cnt), showarrow: false, yshift: 8,
      font: { size: 10, color: '#ccc' }
    })),
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Regression: Threshold Variation (plotly_acc_pre_for_reg) ─────
function plotThresholdVar(div, d, large) {
  const tv = d.threshold_variation;
  const colors = ['#e94560', '#4ecca3', '#533483', '#ffc107'];
  const series = [
    { y: tv.accuracy, name: 'Accuracy', c: colors[0] },
    { y: tv.precision, name: 'Precision', c: colors[1] },
    { y: tv.recall, name: 'Recall', c: colors[2] },
    { y: tv.positive_ratio, name: 'Positive Ratio', c: colors[3] },
  ];
  const traces = series.map(s => ({
    x: tv.cutoffs, y: s.y, mode: 'lines', type: 'scatter', name: s.name,
    line: { color: s.c, width: 2 }
  }));
  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Threshold Variation (Positive > threshold) — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Threshold' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: '%' },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Regression: Hit Enrichment (plotly_enrichment) ──────────────
function plotEnrichment(div, d, large) {
  const enr = d.enrichment;
  const traces = [
    {
      x: enr.sf, y: enr.tpf, mode: 'lines', type: 'scatter',
      name: 'True Pos. Fraction', fill: 'tozeroy', fillcolor: 'rgba(78,204,163,0.2)',
      line: { color: '#4ecca3', width: 2 }
    },
    { x: [0, 1], y: [0, 1], mode: 'lines', showlegend: false,
      line: { color: 'rgba(112,128,144,0.6)', width: 2, dash: 'dash' } },
  ];
  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Hit Enrichment (cutoff=${enr.cutoff}) — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Selected Fraction', range: [0, 1] },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'True Positive Fraction', range: [0, 1.05] },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Classification plots ────────────────────────────────────────
function plotConfusion(div, d, large) {
  const cm = d.confusion_matrix;
  const n_labels = cm.labels.length;
  // Build annotation text
  const total = cm.z.flat().reduce((a, b) => a + b, 0);
  const textArr = cm.z.map((row, i) => {
    const rowSum = row.reduce((a, b) => a + b, 0);
    return row.map((v, j) => {
      const globalPct = total > 0 ? (100 * v / total).toFixed(1) : '0.0';
      const rowPct = rowSum > 0 ? (100 * v / rowSum).toFixed(1) : '0.0';
      let prefix = '';
      if (n_labels === 2) {
        const labels2x2 = [['TN', 'FP'], ['FN', 'TP']];
        prefix = labels2x2[i][j] + '<br>';
      }
      return `${prefix}Cnt: ${v}<br>Glob: ${globalPct}%<br>Row: ${rowPct}%`;
    });
  });

  const traces = [{
    z: cm.z, x: cm.labels, y: cm.labels,
    type: 'heatmap', colorscale: 'Blues',
    text: textArr, texttemplate: '%{text}',
    textfont: { size: Math.max(22 - 2 * n_labels, 10) },
    hovertemplate: 'True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
  }];

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Confusion Matrix — Accuracy: ${cm.accuracy}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Predicted Label', type: 'category' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'True Label', autorange: 'reversed', type: 'category' },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

function plotROC(div, d, large) {
  const curves = d.roc_curves;
  const colors = ['#e94560', '#4ecca3', '#533483', '#ffc107', '#00b4d8'];
  const traces = curves.map((c, i) => ({
    x: c.fpr, y: c.tpr,
    mode: 'lines', type: 'scatter',
    name: `${c.label} (AUC=${c.auc})`,
    line: { color: colors[i % colors.length], width: 2 }
  }));
  // Diagonal
  traces.push({
    x: [0, 1], y: [0, 1],
    mode: 'lines', showlegend: false,
    line: { color: 'rgba(112,128,144,0.6)', width: 2, dash: 'dash' }
  });

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `ROC Curves — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'False Positive Rate', range: [0, 1] },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'True Positive Rate', range: [0, 1.05] },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

function plotPR(div, d, large) {
  const curves = d.pr_curves;
  const colors = ['#e94560', '#4ecca3', '#533483', '#ffc107', '#00b4d8'];
  const traces = curves.map((c, i) => ({
    x: c.recall, y: c.precision,
    mode: 'lines', type: 'scatter',
    name: `${c.label} (AUC=${c.auc})`,
    line: { color: colors[i % colors.length], width: 2 }
  }));

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Precision-Recall Curves — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Recall', range: [0, 1] },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Precision', range: [0, 1.05] },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

function plotClfReport(div, d, large) {
  const rpt = d.classification_report;

  const traces = [{
    z: rpt.z, x: rpt.col_labels, y: rpt.row_labels,
    type: 'heatmap',
    colorscale: [
      [0.00, '#b2182b'],
      [0.20, '#d6604d'],
      [0.40, '#d4b8b0'],
      [0.50, '#f7f7f7'],
      [0.65, '#a8d4a8'],
      [0.80, '#4dac26'],
      [1.00, '#1a7a1a'],
    ],
    zmin: 0.3, zmax: 1.0,
    text: rpt.z.map(row => row.map(v => v !== null ? v.toString() : '')),
    texttemplate: '%{text}',
    textfont: { size: 16 },
    hovertemplate: '%{y} — %{x}: %{z:.3f}<extra></extra>',
  }];

  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Classification Report — Accuracy: ${rpt.accuracy}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, side: 'top', type: 'category' },
    yaxis: { ...DARK_LAYOUT.yaxis, type: 'category' },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Classification: F1 Threshold (plotly_clf_f1) ────────────────
function plotF1Threshold(div, d, large) {
  const f1d = d.f1_threshold;
  if (!f1d || f1d.length === 0) { div.innerHTML = '<p style="color:#999;padding:20px">No probability data available</p>'; return; }
  const colors = ['#e94560', '#4ecca3', '#533483', '#ffc107', '#00b4d8'];
  const traces = [];
  f1d.forEach((cls, ci) => {
    traces.push({ x: cls.thresholds, y: cls.precision, mode: 'lines', name: `Precision ${cls.label}`, line: { color: colors[(ci*3)%colors.length], width: 2 } });
    traces.push({ x: cls.thresholds, y: cls.recall, mode: 'lines', name: `Recall ${cls.label}`, line: { color: colors[(ci*3+1)%colors.length], width: 2 } });
    traces.push({ x: cls.thresholds, y: cls.fscore, mode: 'lines', name: `F1 ${cls.label}`, line: { color: colors[(ci*3+2)%colors.length], width: 2, dash: 'dot' } });
  });
  // Vertical line at 0.5
  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Precision, Recall & F1 vs Threshold — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Threshold', range: [0, 1] },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Score', range: [0, 1.05] },
    height: large ? 700 : 460,
    shapes: [
      { type: 'line', x0: 0.5, x1: 0.5, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(0,0,128,0.7)', width: 2, dash: 'dash' } },
      ...f1d.map(cls => ({ type: 'line', x0: cls.best_threshold, x1: cls.best_threshold, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(255,255,255,0.4)', width: 1.5, dash: 'dash' } })),
    ],
    annotations: f1d.map(cls => ({ x: cls.best_threshold, y: 1.02, yref: 'paper', text: `F1=${cls.best_f1}`, showarrow: false, font: { size: 10, color: '#ccc' } })),
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Classification: Calibration / Reliability (plotly_clf_calbc) ─
function plotCalibration(div, d, large) {
  const cal = d.calibration;
  if (!cal || cal.length === 0) { div.innerHTML = '<p style="color:#999;padding:20px">No probability data available</p>'; return; }
  const colors = ['#e94560', '#4ecca3', '#533483', '#ffc107', '#00b4d8'];
  const traces = cal.map((cls, i) => ({
    x: cls.prob_bins, y: cls.acc_bins, mode: 'lines+markers+text',
    name: `${cls.label} (ECE=${cls.ece}, MCE=${cls.mce})`,
    text: cls.bin_sizes.map(String), textposition: i % 2 === 0 ? 'top center' : 'bottom center',
    textfont: { size: 9, color: colors[i % colors.length] },
    line: { color: colors[i % colors.length], width: 2 },
    marker: { size: 6 },
    hovertemplate: '<b>Prob group</b>: %{x:.2f}<br><b>Observed ratio</b>: %{y:.2f}<br><b>Samples</b>: %{text}',
  }));
  traces.push({ x: [0, 1], y: [0, 1], mode: 'lines', showlegend: false,
    line: { color: 'rgba(112,128,144,0.6)', width: 2, dash: 'dash' } });
  const ece_avg = (cal.reduce((s, c) => s + c.ece, 0) / cal.length).toFixed(3);
  const mce_avg = (cal.reduce((s, c) => s + c.mce, 0) / cal.length).toFixed(3);
  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Reliability Diagram — ECE: ${ece_avg}, MCE: ${mce_avg}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Probability Group', range: [-0.05, 1.05] },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Observed Ratio', range: [-0.05, 1.05] },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Classification: Hit Enrichment (plotly_enrichment_clf) ──────
function plotEnrichmentClf(div, d, large) {
  const enr = d.enrichment_clf;
  if (!enr || enr.length === 0) { div.innerHTML = '<p style="color:#999;padding:20px">No probability data available</p>'; return; }
  const colors = ['#e94560', '#4ecca3', '#533483', '#ffc107', '#00b4d8'];
  const traces = enr.map((cls, i) => ({
    x: cls.sf, y: cls.tpf, mode: 'lines', type: 'scatter',
    name: `TPF ${cls.label}`, line: { color: colors[i % colors.length], width: 2 }
  }));
  traces.push({ x: [0, 1], y: [0, 1], mode: 'lines', showlegend: false,
    line: { color: 'rgba(112,128,144,0.6)', width: 2, dash: 'dash' } });
  const layout = {
    ...DARK_LAYOUT,
    title: { text: `Hit Enrichment — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Selected Fraction', range: [0, 1] },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'True Positive Fraction', range: [0, 1.05] },
    height: large ? 700 : 460,
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Classification: Probability Bars (plotly_confusion_bars_from_categories)
function plotProbBars(div, d, large) {
  const pb = d.prob_bars;
  if (!pb) { div.innerHTML = '<p style="color:#999;padding:20px">No probability data available</p>'; return; }
  const nBins = pb.prob_bin_labels.length;
  const nGreen = Math.floor(nBins / 2);
  const mkColor = (i) => {
    if (i < nGreen) { const a = 0.2 + 0.6 * (nGreen - 1 - i) / Math.max(nGreen - 1, 1); return `rgba(0,100,0,${a})`; }
    else { const a = 0.2 + 0.6 * (i - nGreen) / Math.max(nBins - nGreen - 1, 1); return `rgba(233,69,96,${a})`; }
  };
  const traces = pb.prob_bin_labels.map((lbl, i) => ({
    x: pb.class_labels, y: pb.series[lbl], name: lbl, type: 'bar',
    marker: { color: mkColor(i) },
    text: pb.series[lbl].map(v => v > 0.05 ? v.toFixed(2) : ''),
    textposition: 'inside', textfont: { size: 10 },
    hovertemplate: '<b>Class</b>: %{x}<br><b>Fraction</b>: %{y:.2f}<br><b>Prob bin</b>: ' + lbl,
  }));
  const layout = {
    ...DARK_LAYOUT, barmode: 'stack',
    title: { text: `Predicted Probability Barplot — ${d.property}`, font: { size: 15 } },
    xaxis: { ...DARK_LAYOUT.xaxis, title: 'Class', type: 'category' },
    yaxis: { ...DARK_LAYOUT.yaxis, title: 'Relative Counts', range: [0, 1.08] },
    height: large ? 700 : 460,
    annotations: pb.class_counts.map((cnt, i) => ({
      x: pb.class_labels[i], y: 1.0, text: String(cnt), showarrow: false, yshift: 8,
      font: { size: 10, color: '#ccc' }
    })),
  };
  Plotly.newPlot(div, traces, layout, { responsive: true });
}

// ── Router ──────────────────────────────────────────────────────
function renderSinglePlot(div, plotType, d, large) {
  const plotFn = {
    scatter_mae: plotScatterMAE,
    residuals: plotResiduals,
    error_hist: plotErrorHist,
    scatter_ma: plotScatterMA,
    scatter_cutoff: plotScatterCutoff,
    error_bars: plotErrorBars,
    threshold_var: plotThresholdVar,
    enrichment: plotEnrichment,
    confusion: plotConfusion,
    roc: plotROC,
    pr: plotPR,
    clf_report: plotClfReport,
    f1_threshold: plotF1Threshold,
    calibration: plotCalibration,
    enrichment_clf: plotEnrichmentClf,
    prob_bars: plotProbBars,
  }[plotType];
  if (plotFn) plotFn(div, d, large);
}

// ── Findings ────────────────────────────────────────────────────
function renderFindings(d) {
  const lines = [];
  lines.push(`Property: ${d.property} (${d.task_type})`);
  lines.push(`Test set size: ${d.n}`);
  lines.push('');
  lines.push('Metrics:');
  Object.entries(d.metrics).forEach(([k, v]) => {
    lines.push(`  ${k}: ${v}`);
  });

  if (d.task_type === 'regression') {
    lines.push('');
    lines.push(`MAE band coverage: ${d.fold1_pct}% within 1x MAE, ${d.fold2_pct}% within 2x MAE`);
    if (d.cutoff_scatter) {
      const cs = d.cutoff_scatter;
      lines.push(`Cutoff analysis (cutoff=${cs.cutoff}): TP=${cs.tp_pct}%, TN=${cs.tn_pct}%, FP=${cs.fp_pct}%, FN=${cs.fn_pct}%`);
    }
  }

  if (d.task_type === 'classification' && d.confusion_matrix) {
    lines.push('');
    lines.push('Confusion matrix (row=true, col=predicted):');
    const cm = d.confusion_matrix;
    const header = '  ' + ['', ...cm.labels].join('\t');
    lines.push(header);
    cm.z.forEach((row, i) => {
      lines.push('  ' + [cm.labels[i], ...row].join('\t'));
    });
  }

  document.getElementById('findingsText').textContent = lines.join('\n');
}

function copyFindings() {
  const text = document.getElementById('findingsText').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

// ── Go ──────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


def render_dashboard(
    run_id: str,
    task_type: str,
    properties: list[str],
    property_data: dict,
    output_path: Path,
):
    """Render the HTML dashboard to a file."""
    # Prepare the DATA json
    data_dict = {"_task_type": task_type, "_properties": properties}
    for prop, pdata in property_data.items():
        data_dict[prop] = pdata

    def sanitize_for_json(obj):
        """Replace NaN/Inf/numpy scalars with JSON-safe equivalents."""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    data_dict = sanitize_for_json(data_dict)
    data_json = json.dumps(data_dict, default=str)

    template = Template(HTML_TEMPLATE)
    html = template.render(
        run_id=run_id,
        task_type=task_type,
        properties=properties,
        data_json=data_json,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--pipeline-state",
    required=True,
    type=click.Path(exists=True),
    help="Path to pipeline_state.json",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output HTML file path",
)
@click.option("--verbose", is_flag=True, help="Print progress info")
def main(pipeline_state: str, output: str, verbose: bool):
    """Generate an interactive dashboard from AutoMol evaluation results."""
    state_path = Path(pipeline_state).resolve()
    output_path = Path(output).resolve()
    run_folder = state_path.parent

    if verbose:
        click.echo(f"Loading pipeline state from {state_path}")

    state = load_pipeline_state(state_path)
    run_id = state.get("run_id", "unknown")
    config = state.get("config", {})
    task_type = config.get("task_type", "regression")
    target_properties = config.get("target_properties", [])
    outputs = state.get("outputs", state.get("files", {}))
    eval_results = outputs.get("evaluation_results", {})
    train_info_paths = outputs.get("train_info", {})
    metrics_from_state = state.get("metrics", {})

    if not eval_results:
        click.echo("ERROR: No evaluation results found in pipeline state.", err=True)
        click.echo("Run the evaluate step (step 5) first.", err=True)
        sys.exit(1)

    if verbose:
        click.echo(f"Run: {run_id}")
        click.echo(f"Task type: {task_type}")
        click.echo(f"Properties: {target_properties}")

    property_data = {}
    for prop in target_properties:
        csv_path = eval_results.get(prop)
        if not csv_path:
            if verbose:
                click.echo(f"  Skipping {prop}: no evaluation CSV")
            continue

        if verbose:
            click.echo(f"  Loading evaluation CSV for {prop}: {csv_path}")

        df = load_evaluation_csv(csv_path, run_folder)

        if verbose:
            click.echo(f"    {len(df)} rows, columns: {list(df.columns)}")

        prop_metrics = metrics_from_state.get(prop, {})

        # Load train info if available
        train_info = None
        ti_path = train_info_paths.get(prop)
        if ti_path:
            train_info = load_train_info(ti_path, run_folder)

        if task_type == "regression":
            property_data[prop] = compute_regression_data(df, prop, prop_metrics)
        elif task_type == "classification":
            property_data[prop] = compute_classification_data(df, prop, prop_metrics)
        else:
            click.echo(f"  WARNING: Unknown task type '{task_type}', treating as regression")
            property_data[prop] = compute_regression_data(df, prop, prop_metrics)

    if not property_data:
        click.echo("ERROR: No valid evaluation data found for any property.", err=True)
        sys.exit(1)

    if verbose:
        click.echo(f"Rendering dashboard to {output_path}")

    render_dashboard(run_id, task_type, list(property_data.keys()), property_data, output_path)

    click.echo(f"Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()
