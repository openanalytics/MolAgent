
from pathlib import Path
import click
import json
import re


def infer_property_from_filename(model_file):
    """Infer a single property name from a per-property model filename.

    Strips suffixes: _stackingregmodel.pt, _stackingclfmodel.pt, _refitted, Class_
    """
    name = Path(model_file).stem
    name = re.sub(r'_stacking(reg|clf)model$', '', name)
    name = name.replace('Class_', '')
    if name.endswith('_refitted'):
        name = name[:-len('_refitted')]
    return name


@click.command()
@click.option('--model-file', required=True, help='Path to .pt model file')
@click.option('--smiles-file', default=None, help='CSV file with SMILES column')
@click.option('--smiles-column', default='smiles', help='SMILES column in CSV')
@click.option('--smiles-list', multiple=True, default=None, help='SMILES strings (can specify multiple)')
@click.option('--properties', multiple=True, default=None,
              help='Properties to predict (default: all in model)')
@click.option('--output-folder', default='MolagentFiles/', help='Output folder')
@click.option('--output-file', default=None, help='Output CSV filename (default: {property}_predictions.csv)')
@click.option('--separator', default=',', help='CSV separator for input file')
@click.option('--compute-sd/--no-compute-sd', default=True, help='Compute prediction uncertainty')
@click.option('--blender-properties', multiple=True, default=None,
              help='Blender property columns to use from input CSV (auto-detected from train info)')
@click.option('--blender-values', multiple=True, default=None,
              help='Blender property values for CLI SMILES (format: property=value, e.g., Y_noisy=1.5)')
@click.option('--verbose/--no-verbose', default=False, help='Verbose output')
def main(**kwargs):
    """Make predictions on new molecules using a trained AutoMol model.

    This script loads a trained model and makes predictions on new SMILES strings.
    The SMILES can be provided via:
    1. A CSV file with a SMILES column
    2. Multiple --smiles-list options on the command line
    3. A single --smiles-list option

    The script auto-detects whether the model is for regression or classification
    and formats the output accordingly.

    Supports both single-property (legacy) and merged multi-property models.
    For merged models, all properties are predicted in a single call and output
    to one CSV. Use --properties to select a subset.

    Blender Properties:
        If the model was trained with blender properties, they must be provided
        during prediction. Use --blender-properties to specify which columns
        contain blender values when using CSV input, or --blender-values to
        provide values directly when using command-line SMILES.

    Examples:
        # Predict from CSV file (single-property model)
        uv run python predict.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --smiles-file new_molecules.csv

        # Predict from merged multi-property model
        uv run python predict.py \\
            --model-file MolagentFiles/merged_refitted_stackingregmodel.pt \\
            --smiles-file new_molecules.csv

        # Predict specific properties from merged model
        uv run python predict.py \\
            --model-file MolagentFiles/merged_stackingregmodel.pt \\
            --properties gamma1 --properties gamma3 \\
            --smiles-file new_molecules.csv

        # Predict from command-line SMILES
        uv run python predict.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --smiles-list "CCO" "c1ccccc1" "CC(C)CC(=O)O"

        # Single SMILES prediction
        uv run python predict.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --smiles-list "CC(=O)OC1=CC=CC=C1C(=O)O"

        # Without uncertainty (faster)
        uv run python predict.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --smiles-file data.csv \\
            --no-compute-sd

        # Custom output file
        uv run python predict.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --smiles-file data.csv \\
            --output-file my_predictions.csv

        # With blender properties from CSV
        uv run python predict.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --smiles-file data_with_blender.csv \\
            --blender-properties Y_noisy

        # With blender properties from CLI
        uv run python predict.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --smiles-list "CCO" \\
            --blender-values Y_noisy=1.5
    """
    import pandas as pd

    from automol.stacking import load_model

    # Extract arguments
    model_file = kwargs['model_file']
    smiles_file = kwargs['smiles_file']
    smiles_column = kwargs['smiles_column']
    smiles_list_cli = kwargs['smiles_list']
    output_folder = kwargs['output_folder']
    output_file = kwargs['output_file']
    sep = kwargs['separator']
    compute_sd = kwargs['compute_sd']
    verbose = kwargs['verbose']

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Determine SMILES input
    if smiles_file is not None:
        # Read SMILES from CSV file
        if verbose:
            print(f'Reading SMILES from: {smiles_file}')

        # Check if gzip compressed
        if smiles_file.endswith('.gz'):
            input_df = pd.read_csv(smiles_file, sep=sep, compression='gzip')
        else:
            input_df = pd.read_csv(smiles_file, sep=sep)

        if smiles_column not in input_df.columns:
            # Try common alternatives
            for alt_col in ['SMILES', 'smiles', 'standardized_smiles', 'Stand_SMILES']:
                if alt_col in input_df.columns:
                    smiles_column = alt_col
                    if verbose:
                        print(f'Using SMILES column: {smiles_column}')
                    break
            else:
                raise ValueError(f'SMILES column "{kwargs["smiles_column"]}" not found in CSV. '
                                f'Available columns: {list(input_df.columns)}')

        smiles_list = input_df[smiles_column].tolist()

        # Keep track of other columns from input file
        other_columns = [col for col in input_df.columns if col != smiles_column]

    elif smiles_list_cli:
        # Use SMILES from command line
        smiles_list = list(smiles_list_cli)
        if verbose:
            print(f'Using {len(smiles_list)} SMILES from command line')
        other_columns = []
    else:
        raise ValueError('Must provide either --smiles-file or at least one --smiles-list')

    # Convert single SMILES to list
    if len(smiles_list) == 1 and isinstance(smiles_list[0], str):
        # Still keep as list for consistent processing
        pass

    # Load model
    if verbose:
        print(f'Loading model from: {model_file}')

    stacked_model = load_model(model_file, use_gpu=False)

    # Discover properties from model
    model_path = Path(model_file)
    available_props = sorted(stacked_model.models.keys())
    properties_cli = list(kwargs['properties']) if kwargs['properties'] else None

    if properties_cli:
        # User specified properties — validate against model
        missing = [p for p in properties_cli if p not in available_props]
        if missing:
            raise ValueError(f'Properties {missing} not found in model. '
                           f'Available: {available_props}')
        properties_to_predict = properties_cli
    elif len(available_props) > 1:
        # Multi-property (merged) model — predict all
        properties_to_predict = available_props
    else:
        # Single-property (legacy) — infer from filename for backward compat
        properties_to_predict = available_props if available_props else [infer_property_from_filename(model_file)]

    is_merged = len(properties_to_predict) > 1

    if verbose:
        if is_merged:
            print(f'Multi-property model: {properties_to_predict}')
        else:
            print(f'Property: {properties_to_predict[0]}')

    # Check for 3D features and warn user
    feature_generators = stacked_model.feature_generators
    has_3d_features = any(fk in feature_generators for fk in ['prolif', 'AffGraph'])

    if has_3d_features:
        if verbose:
            print('WARNING: Model was trained with 3D features (protein-ligand interactions).')
            print('3D features require corresponding protein structures for each molecule.')
            print('Ensure that:')
            print('  1. The input molecules have corresponding entries in the original SDF file')
            print('  2. The protein PDB files are available in the expected folder')
            print('  3. The processed 3D data folder exists')
            print('')
            print('Missing protein data will cause prediction errors or incorrect results.')
            print('')
        # Check if train_info has 3d_config (use first property for lookup)
        first_prop = properties_to_predict[0]
        train_info_path = str(Path(output_folder) / f'{first_prop}_train_info.json')
        if not Path(train_info_path).exists():
            train_info_path = model_file.replace('.pt', '_train_info.json')
        if Path(train_info_path).exists():
            with open(train_info_path) as f:
                train_info = json.load(f)
            if '3d_config' in train_info and verbose:
                print('3D Configuration from training:')
                print(f"  SDF file: {train_info['3d_config'].get('sdf_file', 'N/A')}")
                print(f"  Protein folder: {train_info['3d_config'].get('protein_folder', 'N/A')}")
                print(f"  Processed data folder: {train_info['3d_config'].get('processed_data_folder', 'N/A')}")

    # Determine task type from model
    is_classification = 'clfmodel' in model_path.name.lower()
    if not is_classification and hasattr(stacked_model, 'is_FeatureGenerationClassifier'):
        is_classification = stacked_model.is_FeatureGenerationClassifier()

    if is_classification:
        task = 'Classification'
    else:
        task = 'Regression'

    if verbose:
        print(f'Task: {task}')

    # Get SMILES column from model if available
    model_smiles_col = None
    if hasattr(stacked_model, 'smiles'):
        model_smiles_col = stacked_model.smiles
        if verbose:
            print(f'Model SMILES column: {model_smiles_col}')

    # Read train info to get blender properties metadata
    # For merged models, try the first property's train info
    first_prop = properties_to_predict[0]
    train_info_path = str(Path(output_folder) / f'{first_prop}_train_info.json')
    if not Path(train_info_path).exists():
        train_info_path = model_file.replace('.pt', '_train_info.json')

    blender_properties = []
    if Path(train_info_path).exists():
        with open(train_info_path) as f:
            train_info = json.load(f)
        # Auto-detect from train info or use CLI override
        if kwargs.get('blender_properties'):
            blender_properties = list(kwargs['blender_properties'])
        else:
            blender_properties = train_info.get('blender_properties', [])
    else:
        # CLI override only
        if kwargs.get('blender_properties'):
            blender_properties = list(kwargs['blender_properties'])

    if blender_properties:
        if verbose:
            print(f'Blender properties required: {blender_properties}')

    # Handle blender properties based on input source
    blender_properties_dict = {}
    blender_values_cli = kwargs.get('blender_values')

    if blender_properties:
        if smiles_file is not None:
            # Read blender properties from CSV
            missing_props = [bp for bp in blender_properties if bp not in input_df.columns]
            if missing_props:
                raise ValueError(f'Blender properties not found in CSV: {missing_props}. '
                                f'Available columns: {list(input_df.columns)}')
            # Use the first row's values for all SMILES (or match by index if rows align)
            blender_properties_dict = {bp: input_df[bp].values for bp in blender_properties}
            if verbose:
                print(f'Read blender properties from CSV: {list(blender_properties_dict.keys())}')
        elif smiles_list_cli:
            # Parse CLI blender values
            if not blender_values_cli:
                raise ValueError(f'Model requires blender properties: {blender_properties}. '
                                f'Provide values via --blender-values (format: property=value)')
            # Parse property=value pairs
            blender_properties_dict = {}
            for bv in blender_values_cli:
                if '=' not in bv:
                    raise ValueError(f'Invalid blender value format: {bv}. Use: property=value')
                prop, val = bv.split('=', 1)
                try:
                    blender_properties_dict[prop] = float(val)
                except ValueError:
                    blender_properties_dict[prop] = val  # Keep as string if not numeric
            # Check all required properties are provided
            missing = [bp for bp in blender_properties if bp not in blender_properties_dict]
            if missing:
                raise ValueError(f'Missing blender values for: {missing}')
            # Broadcast to numpy array for predict
            import numpy as np
            blender_properties_dict = {k: np.array([v] * len(smiles_list)) for k, v in blender_properties_dict.items()}
            if verbose:
                print(f'Using CLI blender values: {blender_properties_dict}')

    # Make predictions
    if verbose:
        print(f'Generating predictions for {len(smiles_list)} SMILES...')

    # Determine convert_log10 based on model type and training
    # For classification, always False
    convert_log10 = False if is_classification else True

    predictions = stacked_model.predict(
        props=properties_to_predict,
        smiles=smiles_list,
        blender_properties_dict=blender_properties_dict,
        compute_SD=compute_sd,
        convert_log10=convert_log10
    )

    if verbose:
        print('Predictions generated successfully')

    # Create output DataFrame
    output_df = pd.DataFrame({smiles_column: smiles_list})

    # Add predictions for each property
    for prop in properties_to_predict:
        if is_classification:
            class_property = f'Class_{prop}'

            # Classification output
            if f'predicted_labels_{prop}' in predictions:
                output_df[f'predicted_{class_property}'] = predictions[f'predicted_labels_{prop}']

            # Add class probabilities
            labelnames = {}
            if hasattr(stacked_model, 'labelnames') and prop in stacked_model.labelnames:
                labelnames = stacked_model.labelnames[prop]
            else:
                # Try to infer from prediction keys
                proba_keys = [k for k in predictions.keys()
                              if 'predicted_proba' in k and prop in k]
                if proba_keys:
                    class_names = set()
                    for key in proba_keys:
                        parts = key.split('_class_')
                        if len(parts) > 1:
                            class_names.add(parts[-1])
                    labelnames = {i: name for i, name in enumerate(sorted(class_names))}

            for class_idx, class_name in labelnames.items():
                proba_key = f'predicted_proba_{prop}_class_{class_name}'
                if proba_key in predictions:
                    col_name = f'prob_{class_name}' if not is_merged else f'prob_{prop}_{class_name}'
                    output_df[col_name] = predictions[proba_key]
        else:
            # Regression output
            if f'predicted_{prop}' in predictions:
                output_df[f'predicted_{prop}'] = predictions[f'predicted_{prop}']

            # Add uncertainty if available
            if f'SD_{prop}' in predictions:
                output_df[f'SD_{prop}'] = predictions[f'SD_{prop}']

    # Add other columns from input file if available
    if other_columns and smiles_file is not None:
        for col in other_columns:
            output_df[col] = input_df[col].values

    # Determine output filename
    if output_file is None:
        if is_merged:
            output_file = 'predictions.csv'
        else:
            output_file = f'{properties_to_predict[0]}_predictions.csv'

    output_path = str(Path(output_folder) / output_file)

    # Save predictions
    output_df.to_csv(output_path, index=False)

    if verbose:
        print(f'Predictions saved to: {output_path}')

        # Show sample predictions
        print('\nSample predictions:')
        print(output_df.head(min(5, len(output_df))).to_string(index=False))

    # Save prediction info JSON
    info_path = str(Path(output_folder) / 'predictions_info.json')

    prediction_info = {
        'model_file': str(Path(model_file).name),
        'properties': properties_to_predict,
        'task': task,
        'input_source': 'CSV file' if smiles_file else 'Command line',
        'input_file': str(Path(smiles_file).name) if smiles_file else None,
        'smiles_column': smiles_column,
        'n_predictions': len(smiles_list),
        'compute_sd': compute_sd,
        'output_file': str(Path(output_file).name),
        'merged_model': is_merged,
    }

    # Add labelnames for classification
    if is_classification:
        prediction_info['class_labels'] = {}
        for prop in properties_to_predict:
            if hasattr(stacked_model, 'labelnames') and prop in stacked_model.labelnames:
                prediction_info['class_labels'][prop] = stacked_model.labelnames[prop]

    # Add blender properties if used
    if blender_properties:
        prediction_info['blender_properties'] = blender_properties

    with open(info_path, 'w') as f:
        json.dump(prediction_info, f, indent=2)

    if verbose:
        print(f'Prediction info saved to: {info_path}')

    return f'Predictions complete. Output: {output_path}'


if __name__ == '__main__':
    print(main())
