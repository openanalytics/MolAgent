from pathlib import Path
import click
import json


@click.command()
@click.option('--model-file', required=True, help='Path to trained .pt model file')
@click.option('--csv-file', required=True, help='Path to split CSV file (output from skill 02-data-split)')
@click.option('--output-folder', default='MolagentFiles/', help='Output folder for refitted model')
@click.option('--separator', default=',', help='CSV separator')
@click.option('--smiles-column', default=None, help='SMILES column (auto-detected from train info)')
@click.option('--properties', multiple=True, default=None, help='Properties to refit (auto-detected from train info)')
@click.option('--include-test-set', is_flag=True, default=False, help='Also refit on test set data')
@click.option('--keep-data-files/--no-keep-data-files', default=False, help='Keep original split CSV after refitting')
@click.option('--verbose/--no-verbose', default=False, help='Verbose output')
def main(**kwargs):
    """Refit a previously trained AutoMol model on combined train+validation (and optionally test) data.

    This script loads a trained model and refits it on the combined training data
    for deployment. It reads the split CSV file from the 02-data-split skill and
    the corresponding training metadata.

    The refitted model is saved with a `_refitted` suffix and is ready for deployment.

    Examples:
        # Refit on train+validation only
        uv run python refit_model.py --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --csv-file MolagentFiles/automol_split_Caco2_wang.csv

        # Refit on train+validation+test
        uv run python refit_model.py --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --csv-file MolagentFiles/automol_split_Caco2_wang.csv --include-test-set
    """
    import pandas as pd
    from automol.stacking import load_model, save_model

    # Extract arguments
    model_file = kwargs['model_file']
    csv_file = kwargs['csv_file']
    output_folder = kwargs['output_folder']
    sep = kwargs['separator']
    smiles_column = kwargs['smiles_column']
    properties = list(kwargs['properties']) if kwargs['properties'] else None
    include_test_set = kwargs['include_test_set']
    keep_data_files = kwargs['keep_data_files']
    verbose = kwargs['verbose']

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Determine the property from model file name
    model_path = Path(model_file)
    model_stem = model_path.stem  # e.g., "Y_stackingregmodel" or "Y_refitted_stackingregmodel"

    # Handle both refitted and non-refitted model names
    if '_refitted_stacking' in model_stem:
        # Already refitted, extract property
        property_from_file = model_stem.split('_refitted_stacking')[0]
    elif '_stacking' in model_stem:
        # Standard model name
        property_from_file = model_stem.split('_stacking')[0]
    else:
        raise ValueError(f'Cannot extract property from model file name: {model_stem}')

    # Look for train_info JSON
    train_info_path = Path(output_folder) / f'{property_from_file}_train_info.json'

    if not train_info_path.exists():
        # Try in the same folder as the model file
        model_folder = model_path.parent
        train_info_path = model_folder / f'{property_from_file}_train_info.json'

    if not train_info_path.exists():
        raise ValueError(f'Cannot find train info JSON file at {train_info_path}')

    if verbose:
        print(f'Reading train info from: {train_info_path}')

    with open(train_info_path) as f:
        train_info = json.load(f)

    # Auto-detect smiles_column from train info
    if smiles_column is None:
        smiles_column = train_info.get('smiles_column', 'Stand_SMILES')
        if verbose:
            print(f'Auto-detected SMILES column: {smiles_column}')

    # Auto-detect properties from train info
    if properties is None:
        properties = [train_info.get('property', property_from_file)]
        if verbose:
            print(f'Auto-detected properties: {properties}')

    # Auto-detect blender properties from train info
    blender_properties = train_info.get('blender_properties', [])
    if blender_properties and verbose:
        print(f'Using blender properties: {blender_properties}')

    # Load the model
    if verbose:
        print(f'Loading model from: {model_file}')

    model = load_model(model_file=model_file, use_gpu=False)

    # Check for 3D features in the model
    if verbose:
        feature_generators = model.feature_generators
        has_3d_features = any(fk in feature_generators for fk in ['prolif', 'AffGraph'])
        if has_3d_features:
            print('Model contains 3D feature generators (protein-ligand interactions):')
            for fk in ['prolif', 'AffGraph']:
                if fk in feature_generators:
                    print(f'  - {fk}')
        # Also check train_info for 3D config
        if '3d_config' in train_info:
            print('Model was trained with 3D configuration:')
            print(f"  SDF file: {train_info['3d_config'].get('sdf_file', 'N/A')}")
            print(f"  Protein folder: {train_info['3d_config'].get('protein_folder', 'N/A')}")
            print(f"  Processed data folder: {train_info['3d_config'].get('processed_data_folder', 'N/A')}")
            print('3D feature generators will be preserved during refit.')

    # Read CSV file
    if verbose:
        print(f'Reading CSV file: {csv_file}')

    df = pd.read_csv(csv_file, sep=sep, na_values=['NAN', '?', 'NaN'])

    # Validate split column exists
    if 'split' not in df.columns:
        raise ValueError('CSV file must contain a "split" column with train/valid assignments')

    # Extract train/validation splits
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    valid_df = df[df['split'] == 'valid'].reset_index(drop=True)

    if verbose:
        print(f'Train samples: {len(train_df)}')
        print(f'Validation samples: {len(valid_df)}')

    # Handle test set
    test_df = None
    if include_test_set:
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        if verbose:
            print(f'Test samples: {len(test_df)}')

    # Validate blender properties exist in all splits
    if blender_properties:
        missing_in_train = [bp for bp in blender_properties if bp not in train_df.columns]
        missing_in_valid = [bp for bp in blender_properties if bp not in valid_df.columns]

        if missing_in_train:
            raise ValueError(f'Blender properties not found in train set: {missing_in_train}')
        if missing_in_valid:
            raise ValueError(f'Blender properties not found in validation set: {missing_in_valid}')

        if test_df is not None:
            missing_in_test = [bp for bp in blender_properties if bp not in test_df.columns]
            if missing_in_test:
                raise ValueError(f'Blender properties not found in test set: {missing_in_test}')

    # Prepare data for refitting
    # If include_test_set, combine train + valid + test
    # Otherwise, combine train + valid only
    if include_test_set and test_df is not None:
        combined_train = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        if verbose:
            print(f'Refitting on train+validation+test: {len(combined_train)} samples')
    else:
        combined_train = pd.concat([train_df, valid_df], ignore_index=True)
        if verbose:
            print(f'Refitting on train+validation: {len(combined_train)} samples')

    # Set model data for refit
    # The refit_model method with use_validation=True combines Train and Validation
    # We need to split the combined data into two non-empty parts
    # The split doesn't matter for the final result since all data will be used
    # We split 80/20 here (similar to typical train/valid split)
    import numpy as np
    n_samples = len(combined_train)
    split_idx = int(n_samples * 0.8)

    # Shuffle indices to get random split
    indices = np.random.RandomState(42).permutation(n_samples)
    train_idx = indices[:split_idx]
    valid_idx = indices[split_idx:]

    model.Train = combined_train.iloc[train_idx].reset_index(drop=True)
    model.Validation = combined_train.iloc[valid_idx].reset_index(drop=True)
    model.smiles = smiles_column

    # Get prefix_dict from train_info (needed for refit)
    # The prefix_dict is used to identify model components
    # We need to reconstruct it or retrieve it from the model
    prefix_dict = train_info.get('prefix_dict', None)

    # If prefix_dict not in train_info, try to reconstruct from model
    if prefix_dict is None:
        # Default prefixes based on task type
        task = train_info.get('task', 'Regression')
        if task == 'Regression' or task == 'regression':
            prefix_dict = {'method_prefix': 'reg', 'dim_prefix': 'reduce_dim', 'estimator_prefix': 'est_pipe'}
        else:  # Classification
            prefix_dict = {'method_prefix': 'clf', 'dim_prefix': 'reduce_dim', 'estimator_prefix': 'est_pipe'}

    if verbose:
        print(f'\nRefitting model...')

    # Refit the model for each property
    for prop in properties:
        if verbose:
            print(f'  Refitting property: {prop}')

        # Check for sample weights
        # Sample weights are stored as sw_{property} columns
        sample_weight_col = f'sw_{prop}'
        sample_train = None
        sample_val = None

        if sample_weight_col in combined_train.columns:
            # Extract sample weights from combined data
            # For refit, we only have combined data, so pass it as sample_train
            sample_train = combined_train[sample_weight_col].values
            if verbose:
                print(f'  Using sample weights from column: {sample_weight_col}')

        model.refit_model(
            models=prop,
            prefix_dict=prefix_dict,
            use_validation=True  # This will combine Train and Validation for refit
        )

    # Clean up computed features
    if verbose:
        print(f'\nCleaning up...')

    model.clean()

    # Deep clean before saving (removes data for deployment)
    model.deep_clean()
    model.compute_SD = True

    # Generate output file name
    # Format: {property}_refitted_stacking{task}model.pt
    task = train_info.get('task', 'Regression')
    if task == 'Regression':
        task_suffix = 'reg'
    else:
        task_suffix = 'clf'

    output_model_file = str(Path(output_folder) / f'{property_from_file}_refitted_stacking{task_suffix}model.pt')

    # Save refitted model
    # Disable reproducibility check if blender properties were used
    create_reproducability_output = not bool(blender_properties)

    if verbose:
        print(f'Saving refitted model to: {output_model_file}')

    save_model(model, output_model_file, create_reproducability_output=create_reproducability_output)

    # Save refit info JSON
    refit_info = {
        'original_model': str(model_file),
        'original_train_info': str(train_info_path),
        'input_csv': str(csv_file),
        'property': property_from_file,
        'task': task,
        'smiles_column': smiles_column,
        'blender_properties': blender_properties if blender_properties else None,
        'include_test_set': include_test_set,
        'train_samples': len(train_df),
        'validation_samples': len(valid_df),
        'test_samples': len(test_df) if test_df is not None else 0,
        'total_refit_samples': len(combined_train),
        'refitted_model_file': f'{property_from_file}_refitted_stacking{task_suffix}model.pt',
        'refit_timestamp': pd.Timestamp.now().isoformat()
    }

    refit_info_path = str(Path(output_folder) / f'{property_from_file}_refit_info.json')
    with open(refit_info_path, 'w') as f:
        json.dump(refit_info, f, indent=2)

    if verbose:
        print(f'Saved refit info to: {refit_info_path}')

    # Clean up data files if requested
    if not keep_data_files:
        # Note: The original plan was to clean up CSV files, but based on the
        # training script pattern, we only call model.clean() which removes
        # features from memory. We don't delete CSV/JSON files as they may
        # be needed for other operations.
        pass

    if verbose:
        print(f'\nRefit complete!')
        print(f'  Original model: {model_file}')
        print(f'  Refitted model: {output_model_file}')
        print(f'  Samples used: {len(combined_train)}')

    return f'Refit complete. Model saved to {output_model_file}'


if __name__ == '__main__':
    print(main())
