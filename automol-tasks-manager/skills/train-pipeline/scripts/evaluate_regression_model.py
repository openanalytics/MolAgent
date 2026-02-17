
from pathlib import Path
import click
import json
import shutil
import subprocess


@click.command()
@click.option('--model-file', required=True, help='Path to .pt model file')
@click.option('--test-file', required=True, help='Path to test CSV file (from 02-data-split)')
@click.option('--output-folder', default='MolagentFiles/', help='Output folder for report and predictions')
@click.option('--separator', default=',', help='CSV separator')
@click.option('--smiles-column', default=None, help='SMILES column (auto-detected from train info)')
@click.option('--property', default=None, help='Property to evaluate (auto-detected from model filename)')
@click.option('--results-file', default=None, help='Path to results JSON from training')
@click.option('--split-type', default='test', type=click.Choice(['test', 'valid', 'train']),
              help='Which data split to evaluate (default: test, falls back to valid if not available)')
@click.option('--blender-properties', multiple=True, default=None,
              help='Blender properties to use (auto-detected from train info or specified here)')
@click.option('--title', default=None, help='PDF report title')
@click.option('--authors', default='AutoMol User', help='Report authors')
@click.option('--mails', default=None, help='Contact email')
@click.option('--summary', default=None, help='Report summary')
@click.option('--generate-pdf/--no-generate-pdf', default=True, help='Generate PDF report')
@click.option('--convert-log10/--no-convert-log10', default=None,
              help='Convert log10/logit predictions back to original scale. Auto-detected by default.')
@click.option('--verbose/--no-verbose', default=False, help='Verbose output')
def main(**kwargs):
    """Evaluate AutoMol regression model and generate PDF report.

    This script loads a trained AutoMol model, evaluates it on test/validation data,
    generates evaluation figures, and creates a comprehensive PDF report.

    The script auto-detects the property name from the model filename and reads
    metadata from the training info JSON for proper configuration.

    Blender Properties:
        If the model was trained with blender properties, they are automatically
        detected from the train_info.json and used during evaluation. Use
        --blender-properties to override the auto-detected values.

    Convert Log10:
        The --convert-log10 option controls whether log10/logit predictions are
        converted back to the original scale. By default, this is auto-detected:
        - If data has log10_{property} or logit_{property} columns: convert_log10=False
        - Otherwise: convert_log10=True
        Use --convert-log10 or --no-convert-log10 to override the auto-detection.

    Examples:
        # Basic evaluation
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_Caco2_wang.csv

        # Evaluate on validation set instead of test
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_Caco2_wang.csv \\
            --split-type valid

        # With custom report metadata
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_Caco2_wang.csv \\
            --title "Caco-2 Permeability Model Evaluation" \\
            --authors "Jane Doe" \\
            --summary "Model trained on 1000 compounds with R² = 0.85"

        # With blender properties (auto-detected from train info)
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_test_blender.csv \\
            --split-type valid

        # Override blender properties
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_test_blender.csv \\
            --blender-properties Y_noisy

        # With auto-detected log10 handling
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_data.csv

        # Force conversion to original scale
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_data.csv \\
            --convert-log10

        # Force no conversion (keep log scale)
        uv run python evaluate_regression_model.py \\
            --model-file MolagentFiles/Y_stackingregmodel.pt \\
            --test-file MolagentFiles/automol_split_data.csv \\
            --no-convert-log10
    """
    import numpy as np
    import pandas as pd

    from automol.stacking import load_model
    from automol.plotly_util import PlotlyDesigner, generate_report, get_figures_from_types, create_figure_captions

    # Extract arguments
    model_file = kwargs['model_file']
    test_file = kwargs['test_file']
    output_folder = kwargs['output_folder']
    sep = kwargs['separator']
    smiles_column = kwargs['smiles_column']
    property_name = kwargs['property']
    results_file = kwargs['results_file']
    split_type = kwargs['split_type']
    title = kwargs['title']
    authors = kwargs['authors']
    mails = kwargs['mails']
    summary = kwargs['summary']
    generate_pdf = kwargs['generate_pdf']
    convert_log10 = kwargs['convert_log10']
    verbose = kwargs['verbose']

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Extract property from model filename if not provided
    if property_name is None:
        model_path = Path(model_file)
        # Expected format: {property}_stackingregmodel.pt or {property}_stackingclfmodel.pt
        property_name = model_path.name.replace('_stackingregmodel.pt', '').replace('_stackingclfmodel.pt', '')
        if verbose:
            print(f'Auto-detected property from model filename: {property_name}')

    # Read train info to get metadata
    train_info_path = str(Path(output_folder) / f'{property_name}_train_info.json')
    if not Path(train_info_path).exists():
        # Try in the same folder as the model file
        alt_path = model_file.replace('.pt', '_train_info.json')
        if Path(alt_path).exists():
            train_info_path = alt_path

    if Path(train_info_path).exists():
        with open(train_info_path) as f:
            train_info = json.load(f)

        # Auto-detect smiles column
        if smiles_column is None:
            smiles_column = train_info.get('smiles_column', 'Stand_SMILES')
            if verbose:
                print(f'Auto-detected SMILES column: {smiles_column}')
    else:
        if verbose:
            print(f'Warning: Train info JSON not found at {train_info_path}')
            print('Using default SMILES column: Stand_SMILES')
        smiles_column = 'Stand_SMILES'
        train_info = {}

    # Auto-detect or use specified blender properties
    cli_blender_properties = list(kwargs['blender_properties']) if kwargs.get('blender_properties') else None
    if cli_blender_properties:
        blender_properties = cli_blender_properties
    else:
        blender_properties = train_info.get('blender_properties', [])

    # Read test data
    if verbose:
        print(f'Reading test data from: {test_file}')

    # Check if gzip compressed
    if test_file.endswith('.gz'):
        df = pd.read_csv(test_file, sep=sep, compression='gzip')
    else:
        df = pd.read_csv(test_file, sep=sep, na_values=['NAN', '?', 'NaN'])

    # Filter by split type if 'split' column exists
    if 'split' in df.columns:
        available_splits = df['split'].unique()
        if verbose:
            print(f'Available splits: {available_splits}')

        if split_type in available_splits:
            test_df = df[df['split'] == split_type].reset_index(drop=True)
        else:
            # Fall back to first available split
            if split_type == 'test' and 'valid' in available_splits:
                fallback = 'valid'
                print(f'Warning: split_type "{split_type}" not available. Falling back to "{fallback}"')
                test_df = df[df['split'] == fallback].reset_index(drop=True)
            elif len(available_splits) > 0:
                fallback = available_splits[0]
                print(f'Warning: split_type "{split_type}" not available. Using "{fallback}"')
                test_df = df[df['split'] == fallback].reset_index(drop=True)
            else:
                test_df = df
        if verbose:
            print(f'Using split: {split_type}, samples: {len(test_df)}')
    else:
        test_df = df
        if verbose:
            print(f'No split column found, using all {len(test_df)} samples')

    # Check for property column
    property_col = property_name
    is_log10_data = property_name.startswith('log10')
    is_logit_data = property_name.startswith('logit')

    # Smart default: don't convert if we have transformed data
    if convert_log10 is None:
        convert_log10 = not (is_log10_data or is_logit_data)
        if verbose and (is_log10_data or is_logit_data):
            print(f'Auto-detected transformed data: convert_log10={convert_log10}')

    # Warn about potential scale mismatch
    if is_log10_data and not convert_log10:
        if verbose:
            print(f'Predictions will be on log scale (predicted_log10_{property_name})')
    elif (is_log10_data or is_logit_data) and convert_log10:
        if verbose:
            print(f'Predictions will be converted back to original scale')

    if property_col not in test_df.columns:
        raise ValueError(f'Property column "{property_col}" not found in test data. '
                        f'Available columns: {list(test_df.columns)}')

    # Validate blender properties exist in test_df
    if blender_properties:
        missing_props = [bp for bp in blender_properties if bp not in test_df.columns]
        if missing_props:
            print(f'Warning: Blender properties not found in test data: {missing_props}')
            print(f'Available columns: {list(test_df.columns)}')
            blender_properties = [bp for bp in blender_properties if bp in test_df.columns]
        if verbose and blender_properties:
            print(f'Using blender properties: {blender_properties}')
    else:
        blender_properties = []

    # Load model
    if verbose:
        print(f'Loading model from: {model_file}')

    stacked_model = load_model(model_file, use_gpu=False)
    feature_generators = stacked_model.feature_generators
    has_3d_features = any(fk in feature_generators for fk in ['prolif', 'AffGraph'])

    if has_3d_features and '3d_config' in train_info:
        # Re-instantiate 3D feature generators with proper paths
        if verbose:
            print('Re-initializing 3D feature generators...')

        from importlib_resources import files

        sdf_file = train_info['3d_config'].get('sdf_file')
        protein_folder = train_info['3d_config'].get('protein_folder')
        processed_data_folder = train_info['3d_config'].get('processed_data_folder', 'processed_data')
        gradformer_checkpoint = train_info['3d_config'].get('gradformer_checkpoint', 'abs_stratified_leakproof.ckpt')
        prolif_interactions = train_info['3d_config'].get('prolif_interactions',
            ['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic', 'Cationic', 'CationPi', 'PiCation', 'VdWContact'])
        property_key = train_info['3d_config'].get('property_key', 'pChEMBL')

        # Resolve relative paths to absolute paths (train info paths are relative to run folder)
        # First check if they are already absolute paths
        if not Path(sdf_file).is_absolute():
            # Try relative to run folder first, then to repository root
            run_folder = Path(output_folder)
            sdf_path = run_folder / sdf_file
            if not sdf_path.exists():
                sdf_path = Path(sdf_file)
            sdf_file = str(sdf_path)

        if not Path(protein_folder).is_absolute():
            run_folder = Path(output_folder)
            protein_path = run_folder / protein_folder
            if not protein_path.exists():
                protein_path = Path(protein_folder)
            protein_folder = str(protein_path)

        # Use SDF file stem for generator name (matches training script behavior)
        # The generators create processed_data/{name}/ folders based on the SDF file stem
        target_name = Path(sdf_file).stem

        if verbose:
            print(f'  SDF file: {sdf_file}')
            print(f'  Protein folder: {protein_folder}')

        # Re-create AffGraph generator if needed
        if 'AffGraph' in feature_generators:
            from automol.structurefeatures.GradFormer.interaction_feature_generator import InteractionFeaturesGenerator
            from automol.structurefeatures.GradFormer.absolute_config import get_config

            config, train_name = get_config(data_type=1)
            model_f = str(files('automol.trained_models').joinpath(gradformer_checkpoint))

            interaction_feature_generator = InteractionFeaturesGenerator(
                pdb_folder=protein_folder,
                sdf_file=sdf_file,
                processed_data_folder=processed_data_folder,
                config=config,
                name=target_name,
                properties=[property_key],
                model_f=model_f,
                nb_features=132
            )
            stacked_model.feature_generators['AffGraph'] = interaction_feature_generator
            if verbose:
                print(f'  Re-initialized AffGraph feature generator')

        # Re-create ProLIF generator if needed
        if 'prolif' in feature_generators:
            from automol.structurefeatures.prolif_interactions import ProlifInteractionCountGenerator

            prolif_generator = ProlifInteractionCountGenerator(
                pdb_folder=protein_folder,
                sdf_file=sdf_file,
                name=target_name,
                properties=[property_key],
                interactions=prolif_interactions
            )
            stacked_model.feature_generators['prolif'] = prolif_generator
            if verbose:
                print(f'  Re-initialized ProLIF feature generator')

        if verbose:
            print('3D feature generators re-initialized successfully.')

    # Display 3D feature information if present
    if verbose:
        if has_3d_features:
            print(f'3D features detected in model:')
            for fk in ['prolif', 'AffGraph']:
                if fk in feature_generators:
                    print(f'  - {fk}')
        # Also check train_info for 3D config
        if '3d_config' in train_info:
            print('Model was trained with 3D configuration:')
            print(f"  SDF file: {train_info['3d_config'].get('sdf_file', 'N/A')}")
            print(f"  Protein folder: {train_info['3d_config'].get('protein_folder', 'N/A')}")

    # Check if model requires blender properties that we don't have
    if hasattr(stacked_model, 'tasksfeatures_parameters'):
        for prop in stacked_model.tasksfeatures_parameters:
            model_blender_props = stacked_model.tasksfeatures_parameters[prop].get('blender_properties', [])
            if model_blender_props and not blender_properties:
                raise ValueError(
                    f'Model was trained with blender properties: {model_blender_props}. '
                    f'Please provide test data with these columns or train a new model without blender properties.'
                )

    # Get model metadata
    if hasattr(stacked_model, 'smiles'):
        model_smiles_col = stacked_model.smiles
        if verbose:
            print(f'Model SMILES column: {model_smiles_col}')

    # Get SMILES list
    if smiles_column not in test_df.columns:
        # Try Stand_SMILES as fallback (the default standardized column name)
        if 'Stand_SMILES' in test_df.columns:
            smiles_column = 'Stand_SMILES'
            if verbose:
                print(f'Using fallback SMILES column: {smiles_column}')
        else:
            raise ValueError(f'SMILES column "{smiles_column}" not found in test data. '
                            f'Available columns: {list(test_df.columns)}')

    smiles_list = test_df[smiles_column].tolist()

    # Make predictions
    if verbose:
        print('Generating predictions...')

    # Prepare blender properties dict for prediction
    blender_properties_dict = {}
    if blender_properties:
        blender_properties_dict = {bp: test_df[bp].values for bp in blender_properties}


    predictions = stacked_model.predict(
        props=None,
        smiles=smiles_list,
        blender_properties_dict=blender_properties_dict,
        compute_SD=True,
        convert_log10=convert_log10  # Use smart-detected or CLI-specified value
    )

    if verbose:
        print(f'Predictions generated for {len(smiles_list)} samples')

    # Get true values
    y_true = test_df[property_col].values

    # Generate evaluation figures
    if verbose:
        print('Generating evaluation figures...')

    plotly_designer = PlotlyDesigner()

    figures = plotly_designer.show_regression_report(
        properties=[property_name],
        out=predictions,
        y_true=[y_true],
        smiles=smiles_list,
        results_dict=None
    )

    # Additional figures (error analysis)
    additional_figures = plotly_designer.show_additional_regression_report(
        properties=[property_name],
        out=predictions,
        y_true=[y_true],
        smiles=smiles_list,
        results_dict=None
    )

    # Combine figures
    all_figures = {**figures, **additional_figures}

    if verbose:
        print(f'Generated {len(all_figures)} figures')

    # Select figures for PDF report
    # Type D: Scatter plot with metrics
    # Type E: Scatter plot with moving average error
    # Type F: Barplot for regression
    selected_figure_keys = get_figures_from_types(
        plotly_dictionary=all_figures,
        properties=[property_name],
        types=['D', 'E', 'F']
    )

    # Create captions
    captions, types_details, figure_types = create_figure_captions(selected_figure_keys)

    # Generate report metadata
    dataset_name = train_info.get('dataset', Path(test_file).stem.replace('automol_split_', ''))

    if title is None:
        title = f'{dataset_name} - {property_name} Evaluation Report'

    if summary is None:
        n_samples = len(test_df)
        r2_key = [k for k in predictions.keys() if 'r2' in k.lower()]
        r2_value = predictions.get(r2_key[0], 'N/A') if r2_key else 'N/A'
        summary = (f'Regression model evaluation for property {property_name}. '
                  f'Tested on {n_samples} samples. ')

    # Prepare model parameters for report
    model_param = {
        'property': property_name,
        'task': 'Regression',
        'test_samples': len(test_df),
        'split_type': split_type,
        'features': train_info.get('feature_keys', ['Bottleneck', 'rdkit']),
        'computational_load': train_info.get('computational_load', 'N/A'),
    }

    # Add metrics from predictions if available
    for key, value in predictions.items():
        if isinstance(value, (int, float)) and not key.startswith('predicted'):
            model_param[key] = value

    # Generate PDF report
    pdf_path = str(Path(output_folder) / f'{property_name}_evaluation_report.pdf')

    if generate_pdf:
        # Check if wkhtmltopdf is available
        wkhtmltopdf_available = shutil.which('wkhtmltopdf') is not None

        if not wkhtmltopdf_available:
            print('Warning: wkhtmltopdf not found. Skipping PDF generation.')
            print('To install wkhtmltopdf:')
            print('  Linux/Ubuntu: sudo apt-get install wkhtmltopdf')
            print('  macOS: brew install wkhtmltopdf')
            generate_pdf = False

    if generate_pdf:
        try:
            if verbose:
                print('Generating PDF report...')

            pdf_data = generate_report(
                plotly_dictionary=all_figures,
                selected_figures=selected_figure_keys,
                captions=captions,
                types=types_details,
                title=title,
                authors=authors,
                mails=mails if mails else ' ',
                summary=summary,
                model_param=model_param,
                model_tostr_list=[f'<b>Model:</b> {Path(model_file).name}'],
                properties=[property_name],
                nb_columns=2,
                remove_titles=True
            )

            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)

            if verbose:
                print(f'PDF report saved to: {pdf_path}')

        except Exception as e:
            print(f'Error generating PDF: {e}')
            print('Continuing with CSV output...')
            raise e

    # Save predictions to CSV
    predictions_csv_path = str(Path(output_folder) / f'{property_name}_evaluation_predictions.csv')

    # Create output DataFrame
    output_df = test_df[[smiles_column]].copy()

    # Add true values
    output_df[f'true_{property_name}'] = y_true

    # Add predictions - handle both converted and unconverted naming
    pred_key = f'predicted_{property_name}'
    pred_key_log = f'predicted_log10_{property_name}'
    pred_key_logit = f'predicted_logit_{property_name}'

    if pred_key in predictions:
        output_df[pred_key] = predictions[pred_key]
    elif pred_key_log in predictions:
        output_df[pred_key_log] = predictions[pred_key_log]
    elif pred_key_logit in predictions:
        output_df[pred_key_logit] = predictions[pred_key_logit]

    # Add uncertainty if available - handle both naming conventions
    sd_key = f'SD_{property_name}'
    sd_key_log = f'SD_log10_{property_name}'
    if sd_key in predictions:
        output_df[sd_key] = predictions[sd_key]
    elif sd_key_log in predictions:
        output_df[sd_key_log] = predictions[sd_key_log]

    # Add any other columns from original data
    for col in test_df.columns:
        if col != smiles_column and col not in output_df.columns:
            output_df[col] = test_df[col]

    output_df.to_csv(predictions_csv_path, index=False)

    if verbose:
        print(f'Predictions saved to: {predictions_csv_path}')

    # Save evaluation info JSON
    info_path = str(Path(output_folder) / f'{property_name}_evaluation_info.json')

    evaluation_info = {
        'model_file': str(Path(model_file).name),
        'test_file': str(Path(test_file).name),
        'property': property_name,
        'task': 'Regression',
        'split_type': split_type,
        'smiles_column': smiles_column,
        'test_samples': len(test_df),
        'report_pdf': str(Path(pdf_path).name) if generate_pdf else None,
        'predictions_csv': str(Path(predictions_csv_path).name),
        'title': title,
        'authors': authors,
        'figures_generated': len(all_figures),
        'convert_log10': convert_log10,
        'property_column_used': property_col,
    }

    # Add metrics
    for key, value in predictions.items():
        if isinstance(value, (int, float, str, bool, list)):
            evaluation_info[key] = value

    # Add blender properties if used
    if blender_properties:
        evaluation_info['blender_properties'] = blender_properties

    with open(info_path, 'w') as f:
        json.dump(evaluation_info, f, indent=2)

    if verbose:
        print(f'Evaluation info saved to: {info_path}')

    return f'Evaluation complete. Report: {pdf_path if generate_pdf else "skipped"}, Predictions: {predictions_csv_path}'


if __name__ == '__main__':
    print(main())
