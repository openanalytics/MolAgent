#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "pandas", "scipy"]
# ///
"""
Self-contained dataset detection script for the training pipeline.
Detects SMILES column, task type, blender properties, and recommends features.
Outputs JSON to stdout for the orchestrator to capture.
"""

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import click
from _paths import default_output_folder, replace_csv_suffix  # noqa: E402
from _determinism import maybe_seed_everything, force_serial_jobs  # noqa: E402
import pandas as pd
from scipy import stats


def detect_smiles_column(df):
    """
    Detect which column contains SMILES strings.
    Priority: 'smiles' > 'canonical_smiles' > 'molecule' > 'compound' > 'drug'
    Columns with 'id' in the name are deprioritized (e.g. 'Drug' preferred over 'Drug_ID').
    """
    smiles_keywords_priority = [
        'smiles', 'canonical_smiles',
        'molecule', 'compound',
        'drug',
    ]

    # First pass: look for "smiles" in column name, deprioritize "id" variants
    id_fallback = None
    for col in df.columns:
        col_lower = col.lower()
        if 'smiles' in col_lower:
            if 'id' in col_lower:
                id_fallback = id_fallback or col
            else:
                return col
    if id_fallback:
        return id_fallback

    # Second pass: look for other priority keywords, deprioritize "id" variants
    for keyword in smiles_keywords_priority[2:]:
        id_fallback = None
        for col in df.columns:
            col_lower = col.lower()
            if keyword in col_lower:
                sample = df[col].dropna().head(3)
                if sample.empty:
                    continue
                first_val = str(sample.iloc[0]) if len(sample) > 0 else ""
                if any(c in first_val for c in ['C', 'N', 'O', 'S', '[', ']', '(', ')', '=', '#']):
                    if 'id' in col_lower:
                        id_fallback = id_fallback or col
                    else:
                        return col
        if id_fallback:
            return id_fallback

    # Fallback: check if any column contains SMILES-like strings
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(5)
            if len(sample) == 0:
                continue
            if all(
                isinstance(x, str) and
                any(c in x for c in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']) and
                any(c in x for c in ['(', ')', '[', ']', '=', '#', '@', '/', '\\'])
                for x in sample
            ):
                return col

    return None


def detect_task_type(df, target_column):
    """Detect if the task is regression or classification."""
    if target_column not in df.columns:
        return 'unknown'

    unique_count = df[target_column].nunique()
    dtype = df[target_column].dtype

    if unique_count < 20:
        return 'classification'
    elif dtype in ['int64', 'int32', 'object'] and unique_count < 50:
        return 'classification'
    else:
        return 'regression'


def detect_blender_properties(df, smiles_column, target_property):
    """Detect columns suitable as blender properties."""
    suggestions = []

    for col in df.columns:
        if col == smiles_column or col == target_property:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        reasons = []
        confidence = "low"
        correlation = df[col].corr(df[target_property]) if target_property in df.columns else 0

        similarity = SequenceMatcher(None, col.lower(), target_property.lower()).ratio()
        if similarity > 0.7:
            reasons.append(f"Similar name to target ({similarity:.0%} match)")
            confidence = "moderate"

        if "noisy" in col.lower():
            reasons.append('Contains "noisy" - likely noisy version of target')
            confidence = "strong"

        if abs(correlation) > 0.8:
            reasons.append(f"Very high correlation with target ({correlation:.2f})")
            confidence = "strong"
        elif abs(correlation) > 0.7:
            reasons.append(f"High correlation with target ({correlation:.2f})")
            confidence = "moderate"

        if col.endswith(("_2", "_replicate", "_alt", "_v2")):
            reasons.append("Name suggests replicate measurement")
            if confidence == "low":
                confidence = "moderate"

        if any(keyword in col.lower() for keyword in ["temperature", "temp", "ph", "conc", "batch"]):
            reasons.append("Appears to be experimental condition")
            if confidence == "low":
                confidence = "moderate"

        if reasons:
            suggestions.append({
                "column": col,
                "reasons": reasons,
                "correlation": round(float(correlation), 4),
                "confidence": confidence
            })

    order = {"strong": 0, "moderate": 1, "low": 2}
    suggestions.sort(key=lambda x: order[x["confidence"]])

    return suggestions


def recommend_features(df, has_3d_data=False):
    """Recommend feature set based on dataset characteristics."""
    n_samples = len(df)

    if has_3d_data:
        return ["Bottleneck", "rdkit", "prolif", "AffGraph"]
    else:
        return ["Bottleneck"]


def detect_all_targets(df, smiles_column):
    """Detect all numeric columns and their task types.

    For regression targets, also computes distribution metrics:
    - skewness: measure of asymmetry
    - is_skewed: True if |skewness| > 1.0
    - suggest_log_transform: True if skewed and all values > 0
    """
    targets = []
    for col in df.columns:
        if col == smiles_column:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            task_type = detect_task_type(df, col)
            target_info = {
                "column": col,
                "task_type": task_type,
                "unique_values": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
            }

            if task_type == "regression":
                data = df[col].dropna()
                if len(data) > 0:
                    import numpy as np

                    # Basic stats
                    target_info["mean"] = round(float(data.mean()), 4)
                    target_info["std"] = round(float(data.std()), 4)
                    target_info["min"] = round(float(data.min()), 4)
                    target_info["max"] = round(float(data.max()), 4)

                    # Distribution metrics
                    skewness = float(data.skew())
                    kurtosis = float(data.kurtosis())
                    is_skewed = abs(skewness) > 1.0

                    # Suggest log transform if skewed and all values positive
                    all_positive = bool((data > 0).all())
                    suggest_log = is_skewed and all_positive

                    target_info.update({
                        "skewness": round(skewness, 3),
                        "kurtosis": round(kurtosis, 3),
                        "is_skewed": is_skewed,
                        "suggest_log_transform": suggest_log,
                        "all_positive": all_positive,
                    })

                    # Histogram for threshold selection UI
                    hist_counts, hist_edges = np.histogram(data.values, bins=30)
                    target_info["histogram"] = {
                        "counts": hist_counts.tolist(),
                        "edges": [round(float(e), 4) for e in hist_edges.tolist()],
                    }

            if task_type == "classification":
                data = df[col].dropna()
                unique_vals = sorted(data.unique().tolist())
                n_classes = len(unique_vals)
                # Categorical if all values are small non-negative integers
                is_categorical = all(
                    isinstance(v, (int, float)) and float(v) == int(v) and v >= 0
                    for v in unique_vals
                )
                counts = data.value_counts(dropna=True)
                target_info.update({
                    "suggested_nb_classes": n_classes,
                    "suggested_categorical": is_categorical,
                    "class_values_detected": unique_vals,
                    "class_distribution": {str(k): int(v) for k, v in counts.items()},
                })

            targets.append(target_info)
    return targets


# ---------------------------------------------------------------------------
# Dataset characteristics
# ---------------------------------------------------------------------------

def compute_class_balance(df, targets):
    """For classification targets, compute minority/majority class ratios."""
    result = {}
    for t in targets:
        if t["task_type"] != "classification":
            continue
        col = t["column"]
        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            continue
        total = counts.sum()
        minority_pct = round(float(counts.min() / total * 100), 1)
        result[col] = {
            "minority_pct": minority_pct,
            "n_classes": int(counts.size),
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
    return result


def compute_target_correlations(df, targets):
    """Find highly correlated target pairs (|r| > 0.9)."""
    cols = [t["column"] for t in targets if t["task_type"] == "regression"]
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = df[a].corr(df[b])
            if abs(r) > 0.9:
                pairs.append({
                    "target_a": a,
                    "target_b": b,
                    "correlation": round(float(r), 4),
                })
    return pairs


def compute_null_rates(df, targets):
    """Null percentage per target column."""
    n = len(df)
    return {
        t["column"]: round(float(t["null_count"] / n * 100), 1) if n else 0.0
        for t in targets
    }


def count_valid_smiles(df, smiles_column):
    """SMILES validity check. Uses RDKit MolFromSmiles when available, falls back to pattern matching."""
    if smiles_column is None or smiles_column not in df.columns:
        return (0, len(df))
    total = len(df)
    series = df[smiles_column].dropna()

    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        valid = sum(1 for s in series if Chem.MolFromSmiles(str(s)) is not None)
    except ImportError:
        atom_chars = set("CNOSPFIBcnospfi")
        struct_chars = set("()[]=#@/\\")
        valid = sum(
            1 for val in series
            if any(c in atom_chars for c in str(val))
            and any(c in struct_chars for c in str(val))
        )
    return (valid, total)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def recommend_computational_load(n_samples, n_targets, has_3d, task_type):
    """Recommend computational load based on dataset characteristics."""
    return "cheap", "Good balance of speed and model quality, the user can specify otherwise if they see signal using the cheap setting"


def recommend_split_strategy(n_samples, targets, class_balance):
    """Recommend split strategy based on dataset characteristics."""
    for t in targets:
        col = t["column"]
        if col in class_balance and class_balance[col]["minority_pct"] < 15:
            return "stratified", "Imbalanced classes need balanced splits"
    if n_samples >= 1000:
        return "leave_group_out", "Enough data to test scaffold generalization"
    return "mixed", "Comprehensive validation (stratified + scaffold + activity cliffs)"


def recommend_use_advanced(n_targets, overall_task_type, class_balance):
    """Recommend whether to use advanced training."""
    if overall_task_type == "regression_classification":
        return False, "Standard training recommended — consider advanced for per-task control"
    if n_targets > 5:
        return False, "Standard training recommended — consider advanced for custom CV"
    return False, "Standard training builds a strong stacking ensemble automatically"


def recommend_target_transformations(targets):
    """Recommend transformations for skewed regression targets."""
    recommendations = []
    for t in targets:
        if t.get("task_type") != "regression":
            continue
        if t.get("suggest_log_transform"):
            recommendations.append({
                "column": t["column"],
                "transform": "log10",
                "reason": f"Skewed distribution (skewness={t.get('skewness', 0):.2f}), all values positive"
            })
        elif t.get("is_skewed") and not t.get("all_positive"):
            recommendations.append({
                "column": t["column"],
                "transform": "yeo_johnson",
                "reason": f"Skewed distribution (skewness={t.get('skewness', 0):.2f}) with negative/zero values"
            })
    return recommendations


def generate_warnings(n_samples, targets, null_rates, correlations, class_balance,
                      smiles_validity_rate=None):
    """Generate data-driven warnings for the user."""
    warnings = []

    # Warn about mixed target types
    task_types = set(t["task_type"] for t in targets)
    if len(task_types) > 1:
        reg_cols = [t["column"] for t in targets if t["task_type"] == "regression"]
        clf_cols = [t["column"] for t in targets if t["task_type"] == "classification"]
        warnings.append(
            f"Mixed target types detected: {', '.join(reg_cols)} (regression) "
            f"and {', '.join(clf_cols)} (classification). "
            f"Consider training separate models or selecting only one target type."
        )

    if smiles_validity_rate is not None and smiles_validity_rate < 90:
        if smiles_validity_rate < 50:
            warnings.append(
                f"SMILES validity very low ({smiles_validity_rate}%) — "
                f"check that the correct SMILES column is selected"
            )
        elif smiles_validity_rate < 70:
            warnings.append(
                f"SMILES validity low ({smiles_validity_rate}%) — "
                f"many invalid molecules will be dropped during preparation"
            )
        else:
            warnings.append(
                f"SMILES validity {smiles_validity_rate}% — "
                f"some molecules will be dropped during preparation"
            )

    if n_samples < 50:
        warnings.append(f"Very small dataset ({n_samples} samples) — results may be unreliable")
    elif n_samples < 100:
        warnings.append(f"Small dataset ({n_samples} samples) — consider cheap load for quick validation")

    # Target distribution warnings
    for t in targets:
        col = t["column"]
        if t.get("task_type") == "regression" and t.get("is_skewed"):
            skew = t.get("skewness", 0)
            if t.get("suggest_log_transform"):
                warnings.append(
                    f"Target {col} is skewed (skewness={skew:.2f}) — "
                    f"consider log10 transformation for better model performance"
                )
            elif t.get("all_positive") is False and t.get("is_skewed"):
                warnings.append(
                    f"Target {col} is skewed (skewness={skew:.2f}) with negative/zero values — "
                    f"consider Yeo-Johnson transformation (log10 not applicable)"
                )

    for pair in correlations:
        if abs(pair["correlation"]) > 0.95:
            warnings.append(
                f"Targets {pair['target_a']} and {pair['target_b']} highly correlated "
                f"(r={pair['correlation']}) — consider dropping one"
            )
    for t in targets:
        col = t["column"]
        if null_rates.get(col, 0) > 20:
            warnings.append(f"Target {col} has {null_rates[col]}% missing values — consider data cleaning first")
    for col, bal in class_balance.items():
        if bal["minority_pct"] < 10:
            warnings.append(
                f"Target {col} severely imbalanced (minority class: {bal['minority_pct']}%) "
                f"— stratified split recommended"
            )
        if bal["n_classes"] > 10:
            warnings.append(
                f"Target {col} has {bal['n_classes']} classes — verify this is classification, not regression"
            )
    return warnings


@click.command()
@click.option('--csv-file', required=True, help='Path to input CSV file')
@click.option('--sdf-file', default=None, help='Path to SDF file (indicates 3D data available)')
def main(csv_file, sdf_file):
    """Detect dataset characteristics for AutoMol training pipeline."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        result = {"error": f"File not found: {csv_file}"}
        print(json.dumps(result, indent=2))
        sys.exit(1)

    df = pd.read_csv(csv_path)

    smiles_column = detect_smiles_column(df)
    has_3d = sdf_file is not None and Path(sdf_file).exists()

    all_targets = detect_all_targets(df, smiles_column) if smiles_column else []

    # Determine overall task type from targets
    # "regression_classification" means ALL targets are binary (0/1) and will be
    # modeled with regression estimators (predictions clipped to [0,1] as probabilities).
    # It does NOT mean "some targets are regression, some are classification."
    task_types = set(t["task_type"] for t in all_targets)
    if len(task_types) == 1:
        tt = task_types.pop()
        # If all targets are classification AND all are binary with values in {0,1},
        # suggest regression_classification as an alternative approach.
        if tt == "classification":
            all_binary_01 = all(
                set(t.get("class_values_detected", [])) <= {0, 1, 0.0, 1.0}
                and t.get("suggested_nb_classes", 0) == 2
                for t in all_targets
            )
            overall_task_type = "regression_classification" if all_binary_01 else "classification"
        else:
            overall_task_type = tt
    elif len(task_types) > 1:
        # Mixed target types — pick the dominant one (majority wins)
        type_counts = {}
        for t in all_targets:
            type_counts[t["task_type"]] = type_counts.get(t["task_type"], 0) + 1
        overall_task_type = max(type_counts, key=type_counts.get)
    elif all_targets:
        overall_task_type = all_targets[0]["task_type"]
    else:
        overall_task_type = "unknown"

    # Detect blender properties across ALL targets, deduplicate by column name
    blender_properties = []
    seen_blender_cols = set()
    if smiles_column and all_targets:
        for target in all_targets:
            candidates = detect_blender_properties(df, smiles_column, target["column"])
            for c in candidates:
                if c["column"] not in seen_blender_cols:
                    seen_blender_cols.add(c["column"])
                    blender_properties.append(c)
                else:
                    # Keep highest confidence version
                    order = {"strong": 0, "moderate": 1, "low": 2}
                    for i, existing in enumerate(blender_properties):
                        if existing["column"] == c["column"]:
                            if order[c["confidence"]] < order[existing["confidence"]]:
                                blender_properties[i] = c
                            break
        blender_properties.sort(key=lambda x: {"strong": 0, "moderate": 1, "low": 2}[x["confidence"]])

    features = recommend_features(df, has_3d_data=has_3d)

    # Compute dataset characteristics
    n_samples = len(df)
    class_balance = compute_class_balance(df, all_targets)
    high_correlations = compute_target_correlations(df, all_targets)
    null_rates = compute_null_rates(df, all_targets)
    valid_smiles, total_smiles = count_valid_smiles(df, smiles_column)

    smiles_validity_rate = round(valid_smiles / total_smiles * 100, 1) if total_smiles > 0 else 0.0

    characteristics = {
        "valid_smiles": valid_smiles,
        "total_smiles": total_smiles,
        "smiles_validity_rate": smiles_validity_rate,
        "class_balance": class_balance,
        "high_correlations": high_correlations,
        "null_rates": null_rates,
    }

    # Generate recommendations
    n_targets = len(all_targets)
    load_value, load_reason = recommend_computational_load(n_samples, n_targets, has_3d, overall_task_type)
    split_value, split_reason = recommend_split_strategy(n_samples, all_targets, class_balance)
    adv_value, adv_reason = recommend_use_advanced(n_targets, overall_task_type, class_balance)
    target_transforms = recommend_target_transformations(all_targets)
    warnings = generate_warnings(n_samples, all_targets, null_rates, high_correlations, class_balance,
                                  smiles_validity_rate=smiles_validity_rate)

    recommendations = {
        "computational_load": {"value": load_value, "reason": load_reason},
        "split_strategy": {"value": split_value, "reason": split_reason},
        "use_advanced": {"value": adv_value, "reason": adv_reason},
        "target_transformations": target_transforms,
        "warnings": warnings,
    }

    result = {
        "file": str(csv_path),
        "file_stem": csv_path.stem,
        "n_samples": n_samples,
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "smiles_column": smiles_column,
        "smiles_detected": smiles_column is not None,
        "overall_task_type": overall_task_type,
        "targets": all_targets,
        "blender_properties": blender_properties,
        "has_3d_data": has_3d,
        "recommended_features": features,
        "characteristics": characteristics,
        "recommendations": recommendations,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
