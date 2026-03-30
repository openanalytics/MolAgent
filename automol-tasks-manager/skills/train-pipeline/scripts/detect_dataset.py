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
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import click
import pandas as pd
from scipy import stats


def detect_smiles_column(df):
    """
    Detect which column contains SMILES strings.
    Priority: 'smiles' > 'canonical_smiles' > 'molecule' > 'compound' > 'drug'
    """
    smiles_keywords_priority = [
        'smiles', 'canonical_smiles',
        'molecule', 'compound',
        'drug',
    ]

    # First pass: look for exact "smiles" in column name
    for col in df.columns:
        col_lower = col.lower()
        if 'smiles' in col_lower:
            return col

    # Second pass: look for other priority keywords
    for keyword in smiles_keywords_priority[2:]:
        for col in df.columns:
            col_lower = col.lower()
            if keyword in col_lower:
                sample = df[col].dropna().head(3)
                if sample.empty:
                    continue
                first_val = str(sample.iloc[0]) if len(sample) > 0 else ""
                if any(c in first_val for c in ['C', 'N', 'O', 'S', '[', ']', '(', ')', '=', '#']):
                    return col

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
    elif n_samples < 1500:
        return ["Bottleneck"]
    elif n_samples < 5000:
        return ["Bottleneck", "rdkit"]
    else:
        return ["Bottleneck", "rdkit", "fps_2048_2"]


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
    """Basic SMILES validity: non-null, contains atom chars, has structure chars."""
    if smiles_column is None or smiles_column not in df.columns:
        return (0, len(df))
    total = len(df)
    series = df[smiles_column].dropna()
    atom_chars = set("CNOSPFIBcnospfi")
    struct_chars = set("()[]=#@/\\")
    valid = 0
    for val in series:
        s = str(val)
        if any(c in atom_chars for c in s) and any(c in struct_chars for c in s):
            valid += 1
    return (valid, total)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def recommend_computational_load(n_samples, n_targets, has_3d, task_type):
    """Recommend computational load based on dataset characteristics."""
    if n_samples < 250:
        return "free", "Small dataset — single-model signal check is sufficient"
    return "cheap", "Minimal ensemble recommended; upgrade to moderate or expensive if higher accuracy is needed"


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


def generate_warnings(n_samples, targets, null_rates, correlations, class_balance):
    """Generate data-driven warnings for the user."""
    warnings = []
    if n_samples < 50:
        warnings.append(f"Very small dataset ({n_samples} samples) — results may be unreliable")
    elif n_samples < 100:
        warnings.append(f"Small dataset ({n_samples} samples) — results may have high variance")

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

def sdf_to_df(sdf_file):
    """Load 3D data from SDF file."""
    from rdkit import Chem
    import pandas as pd

    sdf_data={}
    sdf_data['original_smiles']=[]
    for mol in Chem.SDMolSupplier(sdf_file, removeHs=False):
        if mol is None:
            continue
        prop_dict = mol.GetPropsAsDict()

        for key,item in prop_dict.items():
            if key in sdf_data:
                sdf_data[key].append(item)
            else:
                sdf_data[key]=[item]
        sdf_data['original_smiles'].append(Chem.MolToSmiles(mol))

    return pd.DataFrame(sdf_data).dropna(axis="columns")

@click.command()
@click.option('--csv-file', default=None, help='Path to input CSV file, if None csv will be generated from SDF')
@click.option('--sdf-file', default=None, help='Path to SDF file (indicates 3D data available)')
@click.option('--protein-folder', default=None, help='Path to folder contain the different pdbs)')
def main(csv_file, sdf_file, protein_folder):

    if csv_file.endswith('.sdf') and sdf_file is None:
        sdf_file=csv_file
        csv_file=None

    if csv_file is None and sdf_file is not None:
        df = sdf_to_df(sdf_file)
        csv_file=sdf_file.replace('.sdf','.csv')
        df.to_csv(csv_file,index=False)

    """Detect dataset characteristics for AutoMol training pipeline."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        result = {"error": f"File not found: {csv_file}"}
        print(json.dumps(result, indent=2))
        sys.exit(1)

    df = pd.read_csv(csv_path)

    smiles_column = detect_smiles_column(df)
    has_3d = sdf_file is not None and Path(sdf_file).exists()
    if has_3d and protein_folder is None:
        sdf_directory=sdf_file.rsplit('/',1)[0]
        if os.path.exists(sdf_directory+'/pdbs'):
            protein_folder=sdf_directory+'/pdbs'




    all_targets = detect_all_targets(df, smiles_column) if smiles_column else []

    # Determine overall task type from targets
    task_types = set(t["task_type"] for t in all_targets)
    if len(task_types) == 1:
        overall_task_type = task_types.pop()
    elif task_types == {"regression", "classification"}:
        overall_task_type = "regression_classification"
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

    characteristics = {
        "valid_smiles": valid_smiles,
        "total_smiles": total_smiles,
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
    warnings = generate_warnings(n_samples, all_targets, null_rates, high_correlations, class_balance)

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

    if has_3d:
        result['sdf_file']=sdf_file
        result['protein_folder']=protein_folder

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
