"""Pydantic models shared by server.py and _pipeline.py."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """All parameters needed to run a full AutoMol training pipeline."""

    # Required inputs
    csv_file: str = Field(description="Absolute path to the input CSV file")
    smiles_column: str = Field(description="Name of the SMILES column in the CSV")
    properties: list[str] = Field(description="Target property column names to predict")
    task: Literal["Regression", "Classification", "RegressionClassification"] = Field(
        description="Modelling task type. RegressionClassification = binary classification via regression estimators (predictions clipped to [0,1] as probabilities)"
    )

    # Data options
    sep: str = Field(default=",", description="CSV separator (, tab ;)")

    # Model options
    computational_load: Literal["free", "cheap", "moderate", "expensive"] = Field(
        default="cheap", description="Budget: free=0-2min, cheap=2-10min, moderate=10-360min, expensive=1-48hr"
    )
    feature_keys: list[str] = Field(
        default=["Bottleneck"],
        description=(
            "Feature generators to use. Encoders: Bottleneck (default, ChEMBL 37 "
            "E-logD), Bottleneck_chembl37_base (no logD supervision), "
            "Bottleneck_chembl27 (legacy). Plus rdkit, fps_2048_2, ..."
        ),
    )

    # 3D structure features (populated by the server only, never set directly by a caller)
    sdf_file: Optional[str] = Field(default=None, description="Path to SDF file (3D features only)")
    protein_folder: Optional[str] = Field(default=None, description="Path to PDB folder (3D features only)")

    # Regression transforms
    use_log10: bool = Field(default=False, description="Apply log10 to regression targets")
    use_logit: bool = Field(default=False, description="Apply logit to regression targets")

    # Classification options
    categorical: bool = Field(default=False, description="Target is already categorical (0/1/2...)")
    nb_classes: Optional[list[int]] = Field(default=None, description="Number of classes per property")
    class_values: Optional[list] = Field(default=None, description="Explicit class cutoff thresholds")
    class_quantiles: Optional[list] = Field(default=None, description="Quantile-based class cutoffs")

    # Validation / CV
    split_strategy: Literal["mixed", "stratified", "leave_group_out"] = Field(
        default="mixed", description="Train/validation split strategy"
    )
    test_size: float = Field(default=0.25, ge=0.05, le=0.95, description="Validation set fraction")
    outer_folds: int = Field(default=4, ge=2, le=20, description="Number of outer CV folds")
    random_state: int = Field(default=42, description="Random seed for splits")
    random_state_list: list[int] = Field(
        default=[1, 7, 42, 55, 3], description="Random seeds for base estimators"
    )

    # Parallelism
    n_jobs_inner: int = Field(default=2, description="Inner loop jobs (-1 = all cores)")
    n_jobs_outer: Optional[int] = Field(default=None, description="Outer loop jobs (None = serial)")

    # Output
    output_folder: Optional[str] = Field(default=None, description="Run folder path (auto-generated if None)")
    refit: bool = Field(default=True, description="Refit model on train+valid after evaluation")
    include_test_in_refit: bool = Field(default=True, description="Also include test set during refit")

    # Tier 1 extensions — high-value params passed to underlying scripts
    blender_properties: Optional[list[str]] = Field(default=None, description="Blender property columns for prepare step")
    scorer: Optional[str] = Field(default=None, description="Scoring metric (e.g. r2, balanced_accuracy)")
    use_advanced: bool = Field(default=False, description="Use advanced training scripts (stacking of stacking)")
    use_sample_weight: bool = Field(default=False, description="Use sample weights in preparation")
    sample_weight_selection: Optional[str] = Field(
        default=None,
        description="Selection criterion for weighting, e.g. '<1', '>5' (applied uniformly to all properties)",
    )
    sample_weight_multiplier: Optional[float] = Field(
        default=None, ge=1.0, le=1000.0,
        description="Weight multiplier for selected samples (default behavior: 10x)",
    )
    clustering_method: str = Field(default="Bottleneck", description="Clustering method for splitting")
    n_clusters: int = Field(default=20, ge=2, le=200, description="Number of clusters for splitting")
    butina_cutoff: float = Field(default=0.6, ge=0.1, le=0.9, description="Butina cutoff for splitting")

    # Tier 2 extensions — advanced model configuration (None = let computational_load decide)
    base_list: Optional[list[str]] = Field(default=None, description="Base estimator keys (e.g. ['xgb', 'lgbm', 'lasso'])")
    blender_list: Optional[list[str]] = Field(default=None, description="Blender/final estimator keys (e.g. ['svr', 'lasso'])")
    red_dim_list: Optional[list[str]] = Field(default=None, description="Dimensionality reduction methods (e.g. ['passthrough', 'pca'])")
    ensemble_config: Optional[str] = Field(default=None, description="Model hierarchy (single_method, top_method, top_stacking, ...)")
    search_type: Optional[str] = Field(default=None, description="Hyperparameter search type (grid, randomized, hyperopt)")
    randomized_iterations: Optional[int] = Field(default=None, ge=10, le=5000, description="Iterations for randomized/hyperopt search")


class PredictConfig(BaseModel):
    """Parameters for running predictions on new molecules."""

    model_id: Optional[str] = Field(default=None, description="Model ID from the registry")
    model_file: Optional[str] = Field(default=None, description="Direct path to a .pt model file")
    smiles_list: Optional[list[str]] = Field(default=None, description="List of SMILES strings to predict")
    smiles_file: Optional[str] = Field(default=None, description="Path to CSV file with SMILES")
    smiles_column: str = Field(default="smiles", description="SMILES column name in CSV")
    properties: Optional[list[str]] = Field(default=None, description="Properties to predict (default: all in model)")
    compute_sd: bool = Field(default=True, description="Compute prediction uncertainty (SD)")
    blender_properties: Optional[list[str]] = Field(default=None, description="Blender property columns from input CSV")
    blender_values: Optional[dict[str, float]] = Field(default=None, description="Blender values for SMILES list input (property: value)")


class TrainingResult(BaseModel):
    """Outputs returned after a successful pipeline run."""

    run_id: str
    output_folder: str

    # Core deliverables
    model_id: str | None = Field(default=None, description="Model registry ID (use download_model to fetch)")
    model_filename: str = Field(description="Suggested filename when saving the model")
    dashboard_html: str = Field(description="Full self-contained interactive HTML dashboard")

    # Metadata
    metrics: dict = Field(description="Per-property metrics {prop: {R2/Accuracy/...}}")
    properties: list[str]
    task: str

    # Absolute paths for local clients
    model_path: str
    dashboard_path: str
    pipeline_state_path: str

    # Per-property training metadata (outer_folds, random_state, train_size, etc.)
    train_info: dict = Field(default_factory=dict, description="Per-property train_info dicts {prop: {...}}")
