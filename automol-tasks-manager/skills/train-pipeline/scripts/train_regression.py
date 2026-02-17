
from pathlib import Path
import click
import json


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
def main(**kwargs):
    """Train AutoMol regression model for molecular property prediction.

    This script trains an ensemble stacking model for regression tasks using nested
    cross-validation. It follows the AutoMol pipeline and reads data prepared by
    the 02-data-split skill.

    Two operation modes:
    1. Default mode: Specify --computational-load (cheap/moderate/expensive) to use
       predefined configuration sets
    2. Advanced mode: Override individual options like --model-config, --base-list,
       --search-type, etc. to customize the training

    Examples:
        # Default mode - cheap computational load
        uv run python train_regression.py --csv-file MolagentFiles/automol_split_Caco2_wang.csv

        # Specify custom features
        uv run python train_regression.py --csv-file MolagentFiles/automol_split_Caco2_wang.csv \\
            --feature-keys Bottleneck rdkit fps_2048_2

        # With blender properties (additional features for final estimator)
        uv run python train_regression.py --csv-file MolagentFiles/automol_split_Caco2_wang.csv \\
            --blender-properties Y_noisy

        # Advanced mode - customize model configuration
        uv run python train_regression.py --csv-file MolagentFiles/automol_split_Caco2_wang.csv \\
            --model-config single_method --base-list xgb lgbm --scorer r2
    """
    import numpy as np
    import pandas as pd

    from automol.stacking_util import retrieve_search_options, ModelAndParams
    from automol.feature_generators import retrieve_default_offline_generators
    from automol.stacking import save_model
    from automol.plotly_util import save_result_dictionary_to_json

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

    prep_info_path = str(Path(output_folder) / prep_input.replace(".csv", "_info.json"))
    if not Path(prep_info_path).exists():
        # Try in the same folder as the split file
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
        # CLI override takes precedence
        blender_properties = cli_blender_properties
    else:
        # Auto-detect from prep info
        blender_properties = prep_info.get('blender_properties', [])

    # Validate blender properties exist in CSV (with warning)
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
        print(f'  n_jobs: outer={n_jobs_dict["outer_jobs"]}, inner={n_jobs_dict["inner_jobs"]}, method={n_jobs_dict["method_jobs"]}')

    # Setup ModelAndParams
    param_Factory = ModelAndParams(
        task=task,
        computional_load=computational_load,
        distribution_defaults=distribution_defaults,
        hyperopt_defaults=hyperopt_defaults,
        feature_generators=retrieve_default_offline_generators(radius=2, nbits=2048),
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
        print(f'\nPerforming {cv_clustering} clustering for CV...')

    # Check if outer_folds > km_groups
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
        prop_blender_properties = blender_properties if blender_properties is not None else None

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
    # Prepare blender properties dict for prediction if blender properties were used
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
        model_file = str(Path(output_folder) / f'{prop}_stackingregmodel.pt')

        # Save model (disable reproducability check when using blender properties)
        # Blender properties require actual data values, not available during reproducability check
        create_reproducability_output = not bool(blender_properties or has_3d_features)
        save_model(stacked_model, model_file, create_reproducability_output=create_reproducability_output)

        # Save results
        results_file = str(Path(output_folder) / f'{prop}_results.json')
        save_result_dictionary_to_json(
            result_dict=results_dictionary,
            json_file=results_file
        )

        # Save training info
        train_info = {
            'input_file': Path(csv_file).name,
            'dataset': file_stem,
            'property': prop,
            'task': task,
            'smiles_column': smiles_column,
            'feature_keys': feature_keys,
            'computational_load': computational_load,
            'model_config': model_config,
            'base_list': base_list,
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
            'validation_size': len(valid_df)
        }

        # Add blender properties if used
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

        train_info_path = str(Path(output_folder) / f'{prop}_train_info.json')
        with open(train_info_path, 'w') as f:
            json.dump(train_info, f, indent=2)

        if verbose:
            print(f'  Property {prop}:')
            print(f'    Model: {model_file}')
            print(f'    Results: {results_file}')
            print(f'    Info: {train_info_path}')

    # Return summary
    return f'Training complete. Models saved to {output_folder}'


if __name__ == '__main__':
    print(main())
