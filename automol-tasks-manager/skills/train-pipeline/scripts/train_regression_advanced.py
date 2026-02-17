
from pathlib import Path
import click
import json
import shutil
import importlib.util
import sys


def setup_3d_feature_generators(feature_generators, sdf_file, protein_folder,
                                 processed_data_folder='processed_data',
                                 property_key='pChEMBL',
                                 target='target',
                                 chkpt_nm='abs_stratified_leakproof.ckpt',
                                 interactions=None):
    """Setup 3D feature generators for protein-ligand interaction modeling.

    Args:
        feature_generators: Dictionary to add generators to
        sdf_file: Path to SDF file with 3D ligand structures
        protein_folder: Path to folder containing protein PDB files
        processed_data_folder: Folder for processed 3D data
        property_key: Property key used in SDF file
        target: Target name for logging
        chkpt_nm: GradFormer checkpoint name
        interactions: List of ProLIF interaction types

    Returns:
        Updated feature_generators dictionary
    """
    if interactions is None:
        interactions = ['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking',
                       'Anionic', 'Cationic', 'CationPi', 'PiCation', 'VdWContact']

    from automol.structurefeatures.GradFormer.interaction_feature_generator import InteractionFeaturesGenerator
    from automol.structurefeatures.prolif_interactions import ProlifInteractionCountGenerator
    from automol.structurefeatures.GradFormer.absolute_config import get_config
    from importlib_resources import files

    # Setup GradFormer (AffGraph)
    config, train_name = get_config(data_type=1)
    model_f = str(files('automol.trained_models').joinpath(chkpt_nm))

    interaction_feature_generator = InteractionFeaturesGenerator(
        pdb_folder=protein_folder,
        sdf_file=sdf_file,
        processed_data_folder=processed_data_folder,
        config=config,
        name=target,
        properties=[property_key],
        model_f=model_f,
        nb_features=132
    )
    feature_generators['AffGraph'] = interaction_feature_generator

    # Setup ProLIF
    prolif_generator = ProlifInteractionCountGenerator(
        pdb_folder=protein_folder,
        sdf_file=sdf_file,
        name=target,
        properties=[property_key],
        interactions=interactions
    )
    feature_generators['prolif'] = prolif_generator

    return feature_generators


@click.command()
@click.option('--csv-file', required=True, help='Path to split CSV file (output from skill 02-data-split)')
@click.option('--output-folder', default='MolagentFiles/', help='Output folder for model and results')
@click.option('--separator', default=',', help='CSV separator')
@click.option('--smiles-column', default=None, help='SMILES column (auto-detected from JSON info)')
@click.option('--properties', multiple=True, default=None, help='Properties to model (auto-detected from JSON info)')
@click.option('--blender-properties', multiple=True, default=None, help='Additional properties to pass to blender/final estimator (auto-detected from prep info or specified here)')
@click.option('--feature-keys', multiple=True, default=None, help='Feature generators to use (default: Bottleneck rdkit). For 3D features, explicitly add "prolif" and/or "AffGraph"')
# 3D feature options:
@click.option('--sdf-file', default=None, help='Path to SDF file with 3D ligand structures (required for 3D features)')
@click.option('--protein-folder', default=None, help='Path to folder containing protein PDB files (required for 3D features)')
@click.option('--processed-data-folder', default='processed_data', help='Folder for processed 3D data (default: processed_data)')
@click.option('--gradformer-checkpoint', default='abs_stratified_leakproof.ckpt',
              type=click.Choice(['abs_leakproof.ckpt', 'abs_lk_bn_val.ckpt', 'abs_stratified_leakproof.ckpt']),
              help='GradFormer checkpoint name (default: abs_stratified_leakproof.ckpt)')
@click.option('--prolif-interactions', multiple=True,
              default=['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic', 'Cationic', 'CationPi', 'PiCation', 'VdWContact'],
              help='ProLIF interaction types to count (default: all)')
# Default mode (computational_load presets all advanced options):
@click.option('--computational-load', default='cheap', type=click.Choice(['free','cheap', 'moderate', 'expensive']),
              help='Computational load level (sets predefined defaults for all advanced options)')
# Advanced options (override computational_load defaults):
@click.option('--model-config', default=None, type=click.Choice(['single_method', 'inner_methods', 'inner_stacking', 'single_stack', 'top_method', 'top_stacking', 'stacking_stacking']),
              help='Model hierarchy configuration')
@click.option('--base-list', multiple=True, default=None, help='Base estimators (e.g., xgb lgbm lasso svm)')
@click.option('--blender-list', multiple=True, default=None, help='Blender/final estimators')
@click.option('--used-features', multiple=True, default=None, help='Feature sets per property (comma-separated per property)')
@click.option('--red-dim-list', multiple=True, default=None, help='Dimensionality reduction methods')
@click.option('--search-type', default=None, type=click.Choice(['grid', 'randomized', 'hyperopt']),
              help='Parameter search type')
@click.option('--randomized-iterations', default=None, type=int, help='Number of iterations for randomized search')
@click.option('--scorer', default='r2', type=str, help='Scoring metric (default: r2)')
@click.option('--normalizer/--no-normalizer', default=True, help='Normalize input features')
@click.option('--top-normalizer/--no-top-normalizer', default=False, help='Normalize top estimator input')
# Cross-validation options:
@click.option('--outer-folds', default=4, type=int, help='Number of outer CV folds')
@click.option('--cross-val-split', default='GKF', type=click.Choice(['GKF', 'SKF', 'LGO']),
              help='CV split type')
@click.option('--cv-clustering', default='Bottleneck', type=click.Choice(['Bottleneck', 'Butina', 'Scaffold', 'HierarchicalButina']),
              help='Clustering method for CV')
@click.option('--km-groups', default=30, type=int, help='Number of clusters for Bottleneck')
@click.option('--butina-cutoff', multiple=True, default=[0.4, 0.6], type=float,
              help='Cutoff for Butina clustering (can specify multiple)')
# Other options:
@click.option('--random-state', default=42, type=int, help='Random seed for reproducibility')
@click.option('--n-jobs-outer', default=None, type=int, help='Outer loop parallel jobs')
@click.option('--n-jobs-inner', default=2, type=int, help='Inner loop parallel jobs (-1 for all)')
@click.option('--n-jobs-method', default=2, type=int, help='Method fitting parallel jobs')
@click.option('--use-gpu/--no-gpu', default=False, help='Use GPU for training')
@click.option('--verbose/--no-verbose', default=False, help='Verbose output')
# Custom component options:
@click.option('--custom-feature-generator', multiple=True, default=None, help='Path to Python module with custom feature generator class (can specify multiple)')
@click.option('--feature-key', multiple=True, default=None, help='Name to assign to custom feature generator (use multiple times for multiple features)')
@click.option('--feature-params', multiple=True, default=None, help='JSON string with parameters for custom feature generator (use multiple times for multiple generators)')
@click.option('--custom-clustering', default=None, help='Path to Python module with custom clustering class')
@click.option('--clustering-params', default='{}', help='JSON string with parameters for custom clustering')
@click.option('--custom-estimator', multiple=True, default=None, help='Path to Python module with custom estimator class (can specify multiple)')
@click.option('--estimator-name', multiple=True, default=None, help='Name to assign to custom estimator (use multiple times for multiple estimators)')
@click.option('--estimator-params', default=None, help='JSON string with parameters for custom estimator (for single estimator)')
@click.option('--estimator-params-file', default=None, help='Path to JSON file with estimator parameters (supports multiple estimators)')
# Integrated evaluation options:
@click.option('--evaluate-after-training/--no-evaluate-after-training', default=False, help='Enable automatic evaluation after training')
@click.option('--eval-split-type', default='valid', type=click.Choice(['test', 'valid', 'train']),
              help='Which split to evaluate (default: valid)')
@click.option('--eval-generate-pdf/--no-eval-generate-pdf', default=True, help='Generate PDF evaluation report')
@click.option('--eval-title', default=None, help='PDF report title (auto-generated if not specified)')
@click.option('--eval-authors', default='AutoMol User', help='Report authors')
@click.option('--eval-summary', default=None, help='Report summary')
# Integrated refit options:
@click.option('--refit-after-training/--no-refit-after-training', default=False, help='Enable automatic refitting on combined data after training')
@click.option('--refit-include-test/--no-refit-include-test', default=False, help='Also include test set in refit')
@click.option('--keep-refitted-only/--no-keep-refitted-only', default=False, help='Delete original model after successful refit')
def main(**kwargs):
    """Train AutoMol regression model with custom components.

    This script extends the standard training functionality by allowing you to specify
    custom feature generators, custom clustering algorithms, and custom estimators.

    Custom components are specified as Python module paths that will be dynamically
    imported. Each module should contain a factory function:
        - Feature generators: create_feature_generator(**kwargs)
        - Clustering: create_clustering(**kwargs)
        - Estimators: create_estimator(**kwargs)

    Integrated Workflow:
        The script supports a complete workflow in a single command:
        1. Train model with custom components
        2. Optionally evaluate on a specified split (test/valid/train)
        3. Optionally refit on combined data

    Examples:
        # Training with custom estimator only
        uv run train_regression_advanced.py \\
            --csv-file MolagentFiles/automol_split_data.csv \\
            --custom-estimator MolagentFiles/my_regressor.py \\
            --estimator-name my_custom_reg \\
            --estimator-params '{"param1": [1.0, 10.0], "param2": [100]}'

        # Training with multiple custom feature generators
        uv run train_regression_advanced.py \\
            --csv-file MolagentFiles/automol_split_data.csv \\
            --custom-feature-generator custom_feat1.py \\
            --feature-key feat1 \\
            --custom-feature-generator custom_feat2.py \\
            --feature-key feat2

        # Training with custom clustering
        uv run train_regression_advanced.py \\
            --csv-file MolagentFiles/automol_split_data.csv \\
            --custom-clustering MolagentFiles/my_clustering.py \\
            --clustering-params '{"n_clusters": 10}'

        # Complete workflow: train + evaluate + refit
        uv run train_regression_advanced.py \\
            --csv-file MolagentFiles/automol_split_data.csv \\
            --custom-estimator MolagentFiles/my_regressor.py \\
            --estimator-name my_custom \\
            --evaluate-after-training \\
            --eval-split-type test \\
            --refit-after-training \\
            --refit-include-test \\
            --verbose

        # Using estimator parameters file (for multiple estimators)
        uv run train_regression_advanced.py \\
            --csv-file MolagentFiles/automol_split_data.csv \\
            --custom-estimator est1.py --estimator-name custom1 \\
            --custom-estimator est2.py --estimator-name custom2 \\
            --estimator-params-file params.json
    """
    import numpy as np
    import pandas as pd

    from automol.stacking_util import retrieve_search_options, ModelAndParams
    from automol.feature_generators import retrieve_default_offline_generators
    from automol.stacking import save_model, load_model
    from automol.plotly_util import save_result_dictionary_to_json, PlotlyDesigner, generate_report

    # Extract arguments
    csv_file = kwargs['csv_file']
    output_folder = kwargs['output_folder']
    sep = kwargs['separator']
    smiles_column = kwargs['smiles_column']
    properties = list(kwargs['properties']) if kwargs['properties'] else None
    feature_keys = list(kwargs['feature_keys']) if kwargs['feature_keys'] else None
    computational_load = kwargs['computational_load']
    random_state = kwargs['random_state']
    verbose = kwargs['verbose']

    # Custom component arguments
    custom_feature_generators = list(kwargs['custom_feature_generator']) if kwargs['custom_feature_generator'] else []
    feature_keys_custom = list(kwargs['feature_key']) if kwargs['feature_key'] else []
    feature_params_list = list(kwargs['feature_params']) if kwargs['feature_params'] else []
    custom_clustering = kwargs['custom_clustering']
    clustering_params_json = kwargs['clustering_params']
    custom_estimators = list(kwargs['custom_estimator']) if kwargs['custom_estimator'] else []
    estimator_names = list(kwargs['estimator_name']) if kwargs['estimator_name'] else []
    estimator_params_json = kwargs['estimator_params']
    estimator_params_file = kwargs['estimator_params_file']

    # Evaluation arguments
    evaluate_after_training = kwargs['evaluate_after_training']
    eval_split_type = kwargs['eval_split_type']
    eval_generate_pdf = kwargs['eval_generate_pdf']
    eval_title = kwargs['eval_title']
    eval_authors = kwargs['eval_authors']
    eval_summary = kwargs['eval_summary']

    # Refit arguments
    refit_after_training = kwargs['refit_after_training']
    refit_include_test = kwargs['refit_include_test']
    keep_refitted_only = kwargs['keep_refitted_only']

    # Helper functions
    def ensure_templates_exists(templates_dir, source_dir):
        """Auto-copy template files if they don't exist in MolagentFiles."""
        Path(templates_dir).mkdir(parents=True, exist_ok=True)
        template_files = [
            'custom_feature_generator.py',
            'custom_clustering.py',
            'custom_regressor.py',
            'custom_classifier.py'
        ]
        for template in template_files:
            dest = Path(templates_dir) / template
            if not dest.exists():
                src = Path(source_dir) / 'templates' / template
                if src.exists():
                    shutil.copy(src, dest)
                    if verbose:
                        print(f'Copied template: {template} to {templates_dir}')

    def load_custom_module(module_path, module_name):
        """Dynamically import a custom Python module."""
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def get_eval_split(df, split_type):
        """Extract the specified split from dataframe for evaluation."""
        if 'split' not in df.columns:
            if verbose:
                print('Warning: No split column found, using all data for evaluation')
            return df
        available = df['split'].unique()
        if split_type in available:
            return df[df['split'] == split_type].reset_index(drop=True)
        # Fallback logic
        if split_type == 'test' and 'valid' in available:
            if verbose:
                print(f'Warning: test split not found, using valid for evaluation')
            return df[df['split'] == 'valid'].reset_index(drop=True)
        # Use first available split
        fallback = available[0]
        if verbose:
            print(f'Warning: {split_type} split not found, using {fallback} for evaluation')
        return df[df['split'] == fallback].reset_index(drop=True)

    # Get the script's directory for templates
    script_dir = Path(__file__).parent.parent
    templates_output_dir = Path(output_folder) / 'templates'

    # Ensure templates exist
    ensure_templates_exists(templates_output_dir, script_dir)

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Determine if using Advanced or Default mode
    advanced_options = [
        kwargs.get('model_config'),
        kwargs.get('base_list'),
        kwargs.get('blender_list'),
        kwargs.get('search_type'),
    ]
    has_advanced_options = any(opt is not None for opt in advanced_options)

    # Read CSV file
    if verbose:
        print(f'Reading CSV file: {csv_file}')
    df = pd.read_csv(csv_file, sep=sep, na_values=['NAN', '?', 'NaN'])

    # Extract train/validation splits
    if 'split' not in df.columns:
        raise ValueError('CSV file must contain a "split" column with train/valid assignments')
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    valid_df = df[df['split'] == 'valid'].reset_index(drop=True)

    if verbose:
        print(f'Train samples: {len(train_df)}')
        print(f'Validation samples: {len(valid_df)}')

    # Read JSON info chain for metadata
    file_stem = Path(csv_file).stem
    if file_stem.endswith('.csv'):
        file_stem = Path(file_stem).stem

    # The split info JSON is always saved alongside the split CSV with _info.json suffix
    split_info_path = csv_file.replace('.csv', '_info.json')
    if not Path(split_info_path).exists():
        raise ValueError(f'Cannot find split info JSON file at {split_info_path}')

    with open(split_info_path) as f:
        split_info = json.load(f)

    # Read prep info (chained from split info)
    prep_input = split_info.get('input_file', '')
    if not prep_input:
        raise ValueError('Split info JSON missing input_file field')

    prep_info_path = f'{output_folder}{prep_input.replace(".csv", "_info.json")}'
    if not Path(prep_info_path).exists():
        if '/' in prep_input:
            prep_info_path = f"{output_folder}{prep_input.split('/')[-1].replace('.csv', '_info.json')}"
        else:
            prep_info_path = f"{output_folder}{prep_input.replace('.csv', '_info.json')}"

    if not Path(prep_info_path).exists():
        raise ValueError(f'Cannot find prep info JSON file at {prep_info_path}')

    with open(prep_info_path) as f:
        prep_info = json.load(f)

    # Auto-detect settings from JSON info chain
    if smiles_column is None:
        smiles_column = split_info.get('smiles_column', 'Stand_SMILES')
        if verbose:
            print(f'Auto-detected SMILES column: {smiles_column}')

    if properties is None:
        properties = split_info.get('property_columns', [])
        if not properties:
            properties = prep_info.get('properties', ['Y'])
        if verbose:
            print(f'Auto-detected properties: {properties}')

    # Auto-detect feature keys
    if feature_keys is None:
        feature_keys = ['Bottleneck', 'rdkit']
        # Note: 3D features (prolif, AffGraph) must be explicitly requested via --feature-keys
        if verbose:
            print(f'Using features: {feature_keys}')
    else:
        # Validate 3D feature requirements
        has_3d_features = any(fk in feature_keys for fk in ['prolif', 'AffGraph'])
        if has_3d_features:
            sdf_file_arg = kwargs.get('sdf_file')
            protein_folder_arg = kwargs.get('protein_folder')
            if not sdf_file_arg or not protein_folder_arg:
                raise ValueError('For 3D features (prolif, AffGraph), you must provide --sdf-file and --protein-folder')

            # Also check if prep info has 3D data prep
            if prep_info.get('preparation_type') != '3d_data_prep':
                print('Warning: 3D features requested but data was not prepared with 3D mode. Results may be unexpected.')

            if verbose:
                print(f'3D features requested: {[fk for fk in feature_keys if fk in ["prolif", "AffGraph"]]}')
                print(f'SDF file: {sdf_file_arg}')
                print(f'Protein folder: {protein_folder_arg}')

    if verbose:
        print(f'Using features: {feature_keys}')

    # Auto-detect or use specified blender properties
    cli_blender_properties = list(kwargs['blender_properties']) if kwargs.get('blender_properties') else None
    if cli_blender_properties:
        blender_properties = cli_blender_properties
    else:
        blender_properties = prep_info.get('blender_properties', [])

    if blender_properties:
        missing_props = [bp for bp in blender_properties if bp not in df.columns]
        if missing_props:
            print(f'Warning: Blender properties not found in CSV: {missing_props}')
            print(f'Available columns: {list(df.columns)}')
        else:
            if verbose:
                print(f'Using blender properties: {blender_properties}')

    # Setup n_jobs_dict - derive from computational_load if not explicitly set
    n_jobs_defaults = {
        'free': {'outer': 1, 'inner': 1, 'method': 1},
        'cheap': {'outer': 1, 'inner': 2, 'method': 2},
        'moderate': {'outer': -1, 'inner': 2, 'method': 2},
        'expensive': {'outer': -1, 'inner': -1, 'method': 2},
    }
    defaults = n_jobs_defaults.get(computational_load, {'outer': 2, 'inner': 2, 'method': 2})

    n_jobs_dict = {
        'outer_jobs': kwargs.get('n_jobs_outer') if kwargs.get('n_jobs_outer') is not None else defaults['outer'],
        'inner_jobs': kwargs.get('n_jobs_inner') if kwargs.get('n_jobs_inner') is not None else defaults['inner'],
        'method_jobs': kwargs.get('n_jobs_method') if kwargs.get('n_jobs_method') is not None else defaults['method'],
    }

    # Setup search options
    search_type = kwargs.get('search_type', None)
    randomized_iterations = kwargs.get('randomized_iterations', None)

    if search_type is not None:
        randomized_iterations, hyperopt_defaults, distribution_defaults = retrieve_search_options(
            search_type=search_type,
            use_distributions=False,
            n_iter=randomized_iterations
        )
    else:
        hyperopt_defaults = None
        distribution_defaults = False

    # Setup used_features (per-property feature specification)
    if kwargs.get('used_features'):
        used_features = [f.split(',') for f in kwargs['used_features']]
    else:
        used_features = [feature_keys] * len(properties)

    # Setup red_dim_list
    if kwargs.get('red_dim_list'):
        red_dim_list = list(kwargs['red_dim_list'])
    else:
        red_dim_list = ['passthrough']

    # Setup model configuration based on mode
    if has_advanced_options:
        model_config = kwargs.get('model_config', None)
        base_list = list(kwargs['base_list']) if kwargs.get('base_list') else None
        blender_list = list(kwargs['blender_list']) if kwargs.get('blender_list') else None
    else:
        model_config = None
        base_list = None
        blender_list = None

    # Cross-validation options
    outer_folds = kwargs.get('outer_folds', 4)
    cross_val_split = kwargs.get('cross_val_split', 'GKF')
    cv_clustering = kwargs.get('cv_clustering', 'Bottleneck')
    km_groups = kwargs.get('km_groups', 30)
    butina_cutoff = list(kwargs['butina_cutoff']) if kwargs.get('butina_cutoff') else [0.4, 0.6]

    scorer = kwargs.get('scorer', 'r2')
    normalizer = kwargs.get('normalizer', True)
    top_normalizer = kwargs.get('top_normalizer', False)
    use_gpu = kwargs.get('use_gpu', False)

    # Task is always Regression for this script
    task = 'Regression'
    labelnames = None

    if verbose:
        print(f'\nTraining configuration:')
        print(f'  Task: {task}')
        print(f'  Computational load: {computational_load}')
        print(f'  Properties: {properties}')
        print(f'  Features: {feature_keys}')
        print(f'  Outer folds: {outer_folds}')
        print(f'  CV split: {cross_val_split}')
        print(f'  CV clustering: {cv_clustering}')
        print(f'  Scorer: {scorer}')

    # Load custom feature generators
    feature_generators = retrieve_default_offline_generators(radius=2, nbits=2048)

    for idx, fg_path in enumerate(custom_feature_generators):
        fg_key = feature_keys_custom[idx] if idx < len(feature_keys_custom) else f'custom_{idx}'
        try:
            module = load_custom_module(fg_path, f'custom_fg_{fg_key}')
            fg_params = {}
            if idx < len(feature_params_list):
                fg_params = json.loads(feature_params_list[idx])
            custom_fg = module.create_feature_generator(**fg_params)
            feature_generators[fg_key] = custom_fg
            if verbose:
                print(f'Loaded custom feature generator: {fg_key} from {fg_path}')
        except Exception as e:
            print(f'Warning: Failed to load custom feature generator from {fg_path}: {e}')

    # Add custom feature keys to feature_keys list
    for fg_key in feature_keys_custom:
        if fg_key not in feature_keys:
            feature_keys = list(feature_keys) + [fg_key]

    # Setup ModelAndParams
    param_Factory = ModelAndParams(
        task=task,
        computional_load=computational_load,
        distribution_defaults=distribution_defaults,
        hyperopt_defaults=hyperopt_defaults,
        feature_generators=feature_generators,
        use_gpu=use_gpu,
        normalizer=normalizer,
        top_normalizer=top_normalizer,
        random_state=random_state,
        red_dim_list=red_dim_list,
        method_list=base_list,
        blender_list=blender_list,
        model_config=model_config,
        randomized_iterations=randomized_iterations,
        n_jobs_dict=n_jobs_dict,
        labelnames=labelnames,
        verbose=verbose
    )

    # Load estimator parameters
    all_estimator_params = {}
    if estimator_params_file:
        with open(estimator_params_file) as f:
            all_estimator_params = json.load(f)
    elif estimator_params_json:
        all_estimator_params = {'custom': json.loads(estimator_params_json)}

    # Add custom estimators to method archive
    if custom_estimators:
        method_archive = param_Factory.method_archive
        for idx, est_path in enumerate(custom_estimators):
            est_name = estimator_names[idx] if idx < len(estimator_names) else f'custom_{idx}'
            try:
                module = load_custom_module(est_path, f'custom_est_{est_name}')
                custom_est = module.create_estimator()

                params = all_estimator_params.get(est_name, {})

                method_archive.add_method(est_name, custom_est, params)

                # Add to base list
                if base_list:
                    base_list = list(base_list) + [est_name]
                else:
                    base_list = [est_name]

                if verbose:
                    print(f'Added custom estimator: {est_name} from {est_path}')
            except Exception as e:
                print(f'Warning: Failed to load custom estimator from {est_path}: {e}')

    # Setup the stacking model
    stacked_model, prefixes, params_grid, blender_params, paramsearch = param_Factory.get_model_and_params()

    # Set train/validation data
    stacked_model.Train = train_df
    stacked_model.Validation = valid_df
    stacked_model.smiles = smiles_column

    # Setup 3D feature generators if explicitly requested
    has_3d_features = any(fk in feature_keys for fk in ['prolif', 'AffGraph'])
    sdf_file_arg = kwargs.get('sdf_file')
    protein_folder_arg = kwargs.get('protein_folder')
    gradformer_checkpoint = kwargs.get('gradformer_checkpoint', 'abs_stratified_leakproof.ckpt')
    prolif_interactions = list(kwargs.get('prolif_interactions', ['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic', 'Cationic', 'CationPi', 'PiCation', 'VdWContact']))
    processed_data_folder = kwargs.get('processed_data_folder', 'processed_data')

    if has_3d_features and sdf_file_arg and protein_folder_arg:
        # Get property key from prep info
        property_key = prep_info.get('property_key', 'pChEMBL')
        target_name = prep_info.get('dataset', file_stem)

        if verbose:
            print(f'\nSetting up 3D feature generators...')
            print(f'  SDF file: {sdf_file_arg}')
            print(f'  Protein folder: {protein_folder_arg}')
            print(f'  Processed data folder: {processed_data_folder}')
            print(f'  Property key: {property_key}')

        # Setup only the requested 3D feature generators
        if 'AffGraph' in feature_keys:
            if verbose:
                print(f'  Adding AffGraph (GradFormer) feature generator')
            from automol.structurefeatures.GradFormer.interaction_feature_generator import InteractionFeaturesGenerator
            from automol.structurefeatures.GradFormer.absolute_config import get_config
            from importlib_resources import files

            config, train_name = get_config(data_type=1)
            model_f = str(files('automol.trained_models').joinpath(gradformer_checkpoint))

            interaction_feature_generator = InteractionFeaturesGenerator(
                pdb_folder=protein_folder_arg,
                sdf_file=sdf_file_arg,
                processed_data_folder=processed_data_folder,
                config=config,
                name=target_name,
                properties=[property_key],
                model_f=model_f,
                nb_features=132
            )
            stacked_model.feature_generators['AffGraph'] = interaction_feature_generator

        if 'prolif' in feature_keys:
            if verbose:
                print(f'  Adding ProLIF interaction feature generator')
            from automol.structurefeatures.prolif_interactions import ProlifInteractionCountGenerator

            prolif_generator = ProlifInteractionCountGenerator(
                pdb_folder=protein_folder_arg,
                sdf_file=sdf_file_arg,
                name=target_name,
                properties=[property_key],
                interactions=prolif_interactions
            )
            stacked_model.feature_generators['prolif'] = prolif_generator

        if verbose:
            print(f'3D feature generators setup complete.')

    # Perform clustering for CV
    if verbose:
        print(f'\nPerforming clustering for CV...')

    if custom_clustering:
        try:
            module = load_custom_module(custom_clustering, 'custom_clustering')
            clustering_params = json.loads(clustering_params_json)
            clustering_algo = module.create_clustering(**clustering_params)
            clustering_algo.cluster(df[smiles_column].tolist())
            if verbose:
                print(f'Using custom clustering algorithm from {custom_clustering}')
        except Exception as e:
            print(f'Warning: Failed to load custom clustering from {custom_clustering}: {e}')
            print('Falling back to standard clustering')
            if outer_folds > km_groups:
                if km_groups > 2:
                    outer_folds = km_groups - 1
                    if verbose:
                        print(f'Warning: outer_folds > km_groups, reducing to {outer_folds}')
                else:
                    km_groups = outer_folds + 1
                    if verbose:
                        print(f'Warning: increasing km_groups to {km_groups}')
            stacked_model.Data_clustering(
                method=cv_clustering,
                n_groups=km_groups,
                cutoff=butina_cutoff,
                random_state=random_state
            )
    else:
        if outer_folds > km_groups:
            if km_groups > 2:
                outer_folds = km_groups - 1
                if verbose:
                    print(f'Warning: outer_folds > km_groups, reducing to {outer_folds}')
            else:
                km_groups = outer_folds + 1
                if verbose:
                    print(f'Warning: increasing km_groups to {km_groups}')

        stacked_model.Data_clustering(
            method=cv_clustering,
            n_groups=km_groups,
            cutoff=butina_cutoff,
            random_state=random_state
        )

    # Train the model for each property
    results_dictionary = {}

    for prop in properties:
        if verbose:
            print(f'\nTraining model for property: {prop}')

        # Get feature-specific params
        prop_idx = properties.index(prop)
        features = used_features[prop_idx] if prop_idx < len(used_features) else feature_keys

        params_grid_prop, blender_params_prop = param_Factory.get_feature_params(
            selected_features=features,
            dim_list=None
        )

        # Prepare blender properties for this property (if any specified)
        # Pass empty list instead of None to avoid iteration errors
        prop_blender_properties = blender_properties if blender_properties else []

        stacked_model.search_model(
            df=None,
            prop=prop,
            smiles=smiles_column,
            params_grid=params_grid_prop,
            paramsearch=paramsearch,
            features=features,
            scoring=scorer,
            cv=outer_folds - 1,
            outer_cv_fold=outer_folds,
            split=cross_val_split,
            use_memory=True,
            refit=False,
            blender_params=blender_params_prop,
            blender_properties=prop_blender_properties,
            prefix_dict=prefixes,
            random_state=random_state
        )

        results_dictionary[prop] = {}

    if verbose:
        print(f'\nTraining complete!')
        print(stacked_model.print_metrics())

    # Generate predictions
    smiles_list = stacked_model.Validation[smiles_column]
    blender_properties_dict = None
    if blender_properties:
        blender_properties_dict = {bp: stacked_model.Validation[bp] for bp in blender_properties}
    out = stacked_model.predict(props=None, smiles=smiles_list, blender_properties_dict=blender_properties_dict, compute_SD=True)

    # Refit model on train+validation
    if verbose:
        print(f'\nRefitting models on combined train+validation data...')

    for prop in properties:
        stacked_model.refit_model(models=prop, prefix_dict=prefixes)

    # Save model and results
    if verbose:
        print(f'\nSaving models and results...')

    for prop in properties:
        # Clean before saving
        stacked_model.deep_clean()
        stacked_model.compute_SD = True

        # Model file path
        model_file = f'{output_folder}{prop}_stackingregmodel.pt'

        # Save model
        create_reproducability_output = not bool(blender_properties or has_3d_features)
        save_model(stacked_model, model_file, create_reproducability_output=create_reproducability_output)

        # Save results
        results_file = f'{output_folder}{prop}_results.json'
        save_result_dictionary_to_json(
            result_dict=results_dictionary,
            json_file=results_file
        )

        # Save training info
        train_info = {
            'input_file': f'automol_split_{file_stem}.csv',
            'dataset': file_stem,
            'property': prop,
            'task': task,
            'smiles_column': smiles_column,
            'feature_keys': feature_keys,
            'computational_load': computational_load,
            'model_config': model_config,
            'base_list': list(base_list) if base_list else None,
            'blender_list': blender_list,
            'red_dim_list': red_dim_list,
            'search_type': search_type,
            'scorer': scorer,
            'normalizer': normalizer,
            'top_normalizer': top_normalizer,
            'outer_folds': outer_folds,
            'cross_val_split': cross_val_split,
            'cv_clustering': cv_clustering,
            'km_groups': km_groups,
            'butina_cutoff': butina_cutoff,
            'random_state': random_state,
            'model_file': f'{prop}_stackingregmodel.pt',
            'results_file': f'{prop}_results.json',
            'train_size': len(train_df),
            'validation_size': len(valid_df),
            'custom_components': {
                'feature_generators': list(custom_feature_generators),
                'clustering': custom_clustering,
                'estimators': list(custom_estimators)
            },
            'prefix_dict': prefixes
        }

        if blender_properties:
            train_info['blender_properties'] = blender_properties

        # Add 3D configuration if used
        if has_3d_features and sdf_file_arg and protein_folder_arg:
            train_info['3d_config'] = {
                'sdf_file': sdf_file_arg,
                'protein_folder': protein_folder_arg,
                'processed_data_folder': processed_data_folder,
                'gradformer_checkpoint': gradformer_checkpoint,
                'prolif_interactions': prolif_interactions,
                'property_key': prep_info.get('property_key', 'pChEMBL')
            }

        train_info_path = f'{output_folder}{prop}_train_info.json'
        with open(train_info_path, 'w') as f:
            json.dump(train_info, f, indent=2)

        if verbose:
            print(f'  Property {prop}:')
            print(f'    Model: {model_file}')
            print(f'    Results: {results_file}')
            print(f'    Info: {train_info_path}')

    # ===== Integrated Evaluation =====
    if evaluate_after_training:
        if verbose:
            print(f'\n===== Running Evaluation =====')

        for prop in properties:
            model_file = f'{output_folder}{prop}_stackingregmodel.pt'

            if not Path(model_file).exists():
                print(f'Skipping evaluation for {prop}: model file not found')
                continue

            if verbose:
                print(f'Evaluating {prop} on {eval_split_type} split...')

            # Load the just-trained model
            model = load_model(model_file=model_file, use_gpu=False)

            # Get the evaluation split
            eval_split_df = get_eval_split(df, eval_split_type)
            smiles_list_eval = eval_split_df[smiles_column].tolist()

            # Prepare blender properties for prediction
            blender_properties_dict_eval = None
            if blender_properties:
                blender_properties_dict_eval = {
                    bp: eval_split_df[bp].tolist() for bp in blender_properties
                    if bp in eval_split_df.columns
                }

            # Generate predictions
            predictions = model.predict(
                props=None,
                smiles=smiles_list_eval,
                blender_properties_dict=blender_properties_dict_eval,
                compute_SD=True
            )

            # Generate evaluation report
            if eval_generate_pdf:
                try:
                    pred_key = f'predicted_{prop}'
                    figures = designer.show_regression_report(
                        prop=prop,
                        smiles=smiles_list_eval,
                        true_values=eval_split_df[prop].tolist(),
                        predictions=predictions[pred_key].tolist()
                    )
                    report_path = f'{output_folder}{prop}_evaluation_report.pdf'
                    title = eval_title or f'{prop} Model Evaluation ({eval_split_type} split)'
                    generate_report(
                        figures=figures,
                        f_path=report_path,
                        title=title,
                        authors=eval_authors,
                        mails=None,
                        summary=eval_summary or f'Evaluation of {prop} model on {eval_split_type} split.'
                    )
                    if verbose:
                        print(f'  Generated PDF report: {report_path}')
                except Exception as e:
                    print(f'Warning: Failed to generate PDF report: {e}')

            # Save predictions CSV
            pred_key = f'predicted_{prop}'
            predictions_df = pd.DataFrame({
                'SMILES': smiles_list_eval,
                'True': eval_split_df[prop].tolist(),
                'Predicted': predictions[pred_key].tolist(),
                'SD': predictions[f'{pred_key}_std'].tolist()
            })
            predictions_csv = f'{output_folder}{prop}_evaluation_predictions.csv'
            predictions_df.to_csv(predictions_csv, index=False)
            if verbose:
                print(f'  Saved predictions: {predictions_csv}')

            # Save evaluation info
            eval_info = {
                'model_file': model_file,
                'split_type': eval_split_type,
                'property': prop,
                'n_samples': len(eval_split_df),
                'timestamp': pd.Timestamp.now().isoformat(),
                'predictions_file': predictions_csv
            }
            eval_info_path = f'{output_folder}{prop}_evaluation_info.json'
            with open(eval_info_path, 'w') as f:
                json.dump(eval_info, f, indent=2)

    # ===== Integrated Refit =====
    if refit_after_training:
        if verbose:
            print(f'\n===== Running Refit =====')

        for prop in properties:
            model_file = f'{output_folder}{prop}_stackingregmodel.pt'

            if not Path(model_file).exists():
                print(f'Skipping refit for {prop}: model file not found')
                continue

            if verbose:
                print(f'Refitting {prop} on combined data...')

            # Load the just-trained model
            model = load_model(model_file=model_file, use_gpu=False)

            # Combine train+validation (and optionally test)
            if refit_include_test and 'test' in df['split'].values:
                combined_train = pd.concat([
                    df[df['split'] == 'train'],
                    df[df['split'] == 'valid'],
                    df[df['split'] == 'test']
                ], ignore_index=True)
            else:
                combined_train = pd.concat([
                    df[df['split'] == 'train'],
                    df[df['split'] == 'valid']
                ], ignore_index=True)

            if verbose:
                print(f'  Combined samples: {len(combined_train)}')

            # Get prefix_dict from train_info
            with open(f'{output_folder}{prop}_train_info.json') as f:
                train_info = json.load(f)
            prefix_dict = train_info.get('prefix_dict', {
                'method_prefix': 'reg',
                'dim_prefix': 'reduce_dim',
                'estimator_prefix': 'est_pipe'
            })

            # Split for refit (80/20 split, values don't matter since all data is used)
            n_samples = len(combined_train)
            split_idx = int(n_samples * 0.8)
            indices = np.random.RandomState(42).permutation(n_samples)

            model.Train = combined_train.iloc[indices[:split_idx]].reset_index(drop=True)
            model.Validation = combined_train.iloc[indices[split_idx:]].reset_index(drop=True)
            model.smiles = smiles_column

            # Refit model
            model.refit_model(models=prop, prefix_dict=prefix_dict)

            # Save refitted model
            model.deep_clean()
            model.compute_SD = True
            refit_model_file = f'{output_folder}{prop}_refitted_stackingregmodel.pt'
            save_model(model, refit_model_file, create_reproducability_output=not bool(blender_properties))

            # Save refit info
            refit_info = {
                'original_model': model_file,
                'refitted_model': refit_model_file,
                'property': prop,
                'refit_samples': len(combined_train),
                'include_test_set': refit_include_test,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            refit_info_path = f'{output_folder}{prop}_refit_info.json'
            with open(refit_info_path, 'w') as f:
                json.dump(refit_info, f, indent=2)

            if verbose:
                print(f'  Saved refitted model: {refit_model_file}')

            # Optionally delete original model
            if keep_refitted_only:
                Path(model_file).unlink()
                if verbose:
                    print(f'  Deleted original model: {model_file}')

    # Return summary
    summary_parts = [f'Training complete. Models saved to {output_folder}']
    if evaluate_after_training:
        summary_parts.append('Evaluation complete.')
    if refit_after_training:
        summary_parts.append('Refit complete.')

    return ' '.join(summary_parts)


if __name__ == '__main__':
    print(main())
