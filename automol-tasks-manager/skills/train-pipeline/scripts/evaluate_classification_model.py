
from pathlib import Path
import click
import json
import shutil


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
@click.option('--title', default=None, help='PDF report title')
@click.option('--authors', default='AutoMol User', help='Report authors')
@click.option('--mails', default=None, help='Contact email')
@click.option('--summary', default=None, help='Report summary')
@click.option('--generate-pdf/--no-generate-pdf', default=True, help='Generate PDF report')
@click.option('--convert-log10/--no-convert-log10', default=False,
              help='Convert log10/logit predictions back to original scale (default: False)')
@click.option('--verbose/--no-verbose', default=False, help='Verbose output')
def main(**kwargs):
    """Evaluate AutoMol classification model and generate PDF report.

    This script loads a trained AutoMol classification model, evaluates it on
    test/validation data, generates classification figures (confusion matrix,
    ROC curves, etc.), and creates a comprehensive PDF report.

    The script auto-detects the property name from the model filename and reads
    metadata from the training info JSON for proper configuration.

    Examples:
        # Basic evaluation
        uv run python evaluate_classification_model.py \\
            --model-file MolagentFiles/Class_Y_stackingclfmodel.pt \\
            --test-file MolagentFiles/automol_split_data.csv

        # Evaluate on validation set
        uv run python evaluate_classification_model.py \\
            --model-file MolagentFiles/Class_Y_stackingclfmodel.pt \\
            --test-file MolagentFiles/automol_split_data.csv \\
            --split-type valid

        # With custom report metadata
        uv run python evaluate_classification_model.py \\
            --model-file MolagentFiles/Class_Y_stackingclfmodel.pt \\
            --test-file MolagentFiles/automol_split_data.csv \\
            --title "Drug Target Classification Evaluation" \\
            --authors "Jane Doe" \\
            --summary "Binary classification model with AUC = 0.92"

        # With log10 conversion (default is False for classification)
        uv run python evaluate_classification_model.py \\
            --model-file MolagentFiles/Class_Y_stackingclfmodel.pt \\
            --test-file MolagentFiles/automol_split_data.csv \\
            --convert-log10
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
        # Expected format: {property}_stackingclfmodel.pt or Class_{property}_stackingclfmodel.pt
        property_name = model_path.name.replace('_stackingclfmodel.pt', '').replace('_stackingregmodel.pt', '')
        # For classification, prediction keys use the full property name (e.g., 'Class_Y')
        # while the display name uses the stripped version (e.g., 'Y')
        if property_name.startswith('Class_'):
            prediction_property = property_name  # Keep 'Class_Y' for prediction keys
            property_name = property_name.replace('Class_', '')  # Use 'Y' for display
        else:
            prediction_property = f'Class_{property_name}'  # Construct 'Class_Y'
        if verbose:
            print(f'Auto-detected property from model filename: {property_name}')
            print(f'Prediction property name: {prediction_property}')
    else:
        # If property_name is provided via CLI, construct prediction_property
        if property_name.startswith('Class_'):
            prediction_property = property_name
            property_name = property_name.replace('Class_', '')
        else:
            prediction_property = f'Class_{property_name}'

    # The actual class column is typically Class_{property}
    class_property = prediction_property

    # Read train info to get metadata
    train_info_path = str(Path(output_folder) / f'{property_name}_train_info.json')
    if not Path(train_info_path).exists():
        # Try in the same folder as the model file
        alt_path = model_file.replace('.pt', '_train_info.json')
        if Path(alt_path).exists():
            train_info_path = alt_path

    labelnames = None
    if Path(train_info_path).exists():
        with open(train_info_path) as f:
            train_info = json.load(f)

        # Auto-detect smiles column
        if smiles_column is None:
            smiles_column = train_info.get('smiles_column', 'Stand_SMILES')
            if verbose:
                print(f'Auto-detected SMILES column: {smiles_column}')

        # Get train_properties to find class column name
        train_properties = train_info.get('train_properties', [])
        if train_properties:
            class_property = train_properties[0]
    else:
        if verbose:
            print(f'Warning: Train info JSON not found at {train_info_path}')
            print('Using default SMILES column: Stand_SMILES')
        smiles_column = 'Stand_SMILES'
        train_info = {}

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

    # Check for class column
    if class_property not in test_df.columns:
        # Try other possible class column names
        possible_class_cols = [
            f'Class_{property_name}',
            property_name,
            f'{property_name}_class',
        ]
        for col in possible_class_cols:
            if col in test_df.columns:
                class_property = col
                break
        if class_property not in test_df.columns:
            raise ValueError(f'Class column not found in test data. '
                            f'Tried: {possible_class_cols}. '
                            f'Available columns: {list(test_df.columns)}')

    # Load model
    if verbose:
        print(f'Loading model from: {model_file}')

    stacked_model = load_model(model_file, use_gpu=False)

    # Resolve prediction_property from actual model keys (handles categorical data
    # where Class_ prefix is NOT added by ClassBuilder)
    if hasattr(stacked_model, 'models'):
        model_keys = list(stacked_model.models.keys())
        if prediction_property not in model_keys and property_name in model_keys:
            prediction_property = property_name
            class_property = property_name
            if verbose:
                print(f'Corrected prediction property to match model key: {prediction_property}')

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

    # Get labelnames from model if available
    if hasattr(stacked_model, 'labelnames') and prediction_property in stacked_model.labelnames:
        labelnames = {prediction_property: stacked_model.labelnames[prediction_property]}
        if verbose:
            print(f'Model labelnames: {labelnames}')
    else:
        # Infer from unique values in test data
        unique_classes = sorted(test_df[class_property].unique())
        labelnames = {prediction_property: {i: str(c) for i, c in enumerate(unique_classes)}}
        if verbose:
            print(f'Inferred labelnames from data: {labelnames}')

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

    predictions = stacked_model.predict(
        props=None,
        smiles=smiles_list,
        compute_SD=True,
        convert_log10=convert_log10  # Use CLI parameter instead of hardcoded False
    )

    if verbose:
        print(f'Predictions generated for {len(smiles_list)} samples')

    # Get true values (convert to integer class indices)
    # First, map true labels to indices
    true_labels = test_df[class_property].values
    # Create reverse mapping from label to index
    label_to_index = {v: k for k, v in labelnames[prediction_property].items()}
    y_true = np.array([label_to_index.get(str(label), label) for label in true_labels])

    # Generate classification figures
    if verbose:
        print('Generating classification figures...')

    plotly_designer = PlotlyDesigner()

    # Main classification report (A: classification metrics, B: confusion matrix, C: ROC)
    youden_dict, figures = plotly_designer.show_classification_report(
        properties=[prediction_property],
        out=predictions,
        y_true=[y_true],
        labelnames=labelnames,
        results_dict=None
    )

    # Additional classification report (L: PR curve, M: reliability diagram, K: enrichment)
    additional_figures = plotly_designer.show_additional_classification_report(
        properties=[prediction_property],
        out=predictions,
        y_true=[y_true],
        labelnames=labelnames,
        results_dict=None
    )

    # Threshold report (H: F1 vs threshold, I: probability bars)
    f1_dict, threshold_figures = plotly_designer.show_clf_threshold_report(
        properties=[prediction_property],
        out=predictions,
        y_true=[y_true],
        labelnames=labelnames,
        youden_dict=youden_dict,
        results_dict=None
    )

    # Combine all figures
    all_figures = {**figures, **additional_figures, **threshold_figures}

    if verbose:
        print(f'Generated {len(all_figures)} figures')

    # Select figures for PDF report
    # Type A: Classification report
    # Type B: Confusion matrix
    # Type C: ROC curve
    # Type H: F1-score curves
    # Type I: Predicted probability barplot
    # Type L: Precision-Recall curve
    # Type M: Reliability diagram
    selected_figure_keys = get_figures_from_types(
        plotly_dictionary=all_figures,
        properties=[prediction_property],
        types=['A', 'B', 'C', 'H', 'I', 'L', 'M']
    )

    # Create captions
    captions, types_details, figure_types = create_figure_captions(selected_figure_keys)

    # Generate report metadata
    dataset_name = train_info.get('dataset', Path(test_file).stem.replace('automol_split_', ''))

    if title is None:
        title = f'{dataset_name} - {property_name} Classification Evaluation Report'

    if summary is None:
        n_samples = len(test_df)
        # Try to get AUC from predictions
        auc_key = [k for k in predictions.keys() if 'auc' in k.lower() or 'AUC' in k]
        auc_value = predictions.get(auc_key[0], 'N/A') if auc_key else 'N/A'
        nb_classes = len(labelnames[prediction_property])
        summary = (f'Classification model evaluation for property {property_name}. '
                  f'{nb_classes}-class problem evaluated on {n_samples} samples. ')

    # Prepare model parameters for report
    nb_classes = len(labelnames[prediction_property])
    class_labels = list(labelnames[prediction_property].values())

    model_param = {
        'property': property_name,
        'task': 'Classification',
        'nb_classes': nb_classes,
        'class_labels': class_labels,
        'test_samples': len(test_df),
        'split_type': split_type,
        'features': train_info.get('feature_keys', ['Bottleneck', 'rdkit']),
        'computational_load': train_info.get('computational_load', 'N/A'),
    }

    # Add metrics from predictions if available
    for key, value in predictions.items():
        if isinstance(value, (int, float, str, bool, list)):
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
                properties=[prediction_property],
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

    # Save predictions to CSV
    predictions_csv_path = str(Path(output_folder) / f'{property_name}_evaluation_predictions.csv')

    # Create output DataFrame
    output_df = test_df[[smiles_column]].copy()

    # Add true class labels
    output_df[f'true_{class_property}'] = true_labels

    # Add predicted class labels
    if f'predicted_labels_{prediction_property}' in predictions:
        output_df[f'predicted_{class_property}'] = predictions[f'predicted_labels_{prediction_property}']

    # Add predicted probabilities for each class
    for class_idx, class_name in labelnames[prediction_property].items():
        proba_key = f'predicted_proba_{prediction_property}_class_{class_name}'
        if proba_key in predictions:
            output_df[f'prob_{class_name}'] = predictions[proba_key]

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
        'class_property': class_property,
        'task': 'Classification',
        'nb_classes': nb_classes,
        'class_labels': class_labels,
        'split_type': split_type,
        'smiles_column': smiles_column,
        'test_samples': len(test_df),
        'report_pdf': str(Path(pdf_path).name) if generate_pdf else None,
        'predictions_csv': str(Path(predictions_csv_path).name),
        'title': title,
        'authors': authors,
        'figures_generated': len(all_figures),
        'convert_log10': convert_log10,
    }

    # Add metrics from predictions
    for key, value in predictions.items():
        if isinstance(value, (int, float, str, bool, list)):
            evaluation_info[key] = value

    with open(info_path, 'w') as f:
        json.dump(evaluation_info, f, indent=2)

    if verbose:
        print(f'Evaluation info saved to: {info_path}')

    return f'Evaluation complete. Report: {pdf_path if generate_pdf else "skipped"}, Predictions: {predictions_csv_path}'


if __name__ == '__main__':
    print(main())

