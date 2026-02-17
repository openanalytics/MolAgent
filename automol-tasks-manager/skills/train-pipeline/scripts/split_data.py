
from pathlib import Path
import click
import json


@click.command()
@click.option('--csv-file', required=True, help='Path to prepared CSV file')
@click.option('--output-folder', default='MolagentFiles/', help='Output folder')
@click.option('--separator', default=',', help='CSV separator')
@click.option('--smiles-column', default='Stand_SMILES', help='Standardized SMILES column')
@click.option('--properties', multiple=True, required=True, help='Property column(s) to use for splitting (must exist in dataframe)')
@click.option('--split-strategy', default='mixed', type=click.Choice(['stratified', 'leave_group_out', 'mixed']),
              help='Validation strategy')
@click.option('--clustering-method', default='Bottleneck', type=click.Choice(['Bottleneck', 'Butina', 'Scaffold', 'HierarchicalButina']),
              help='Clustering method')
@click.option('--test-size', default=0.25, type=float, help='Validation set fraction')
@click.option('--add-test-set', is_flag=True, default=False, help='Add test set (3-way split)')
@click.option('--test-size-test', default=0.15, type=float, help='Test set fraction (when --add-test-set)')
@click.option('--n-clusters', default=20, type=int, help='Number of clusters for Bottleneck clustering')
@click.option('--butina-cutoff', default=0.6, type=float, help='Cutoff for Butina clustering')
@click.option('--include-chirality', is_flag=True, default=False, help='Include chirality for Scaffold')
@click.option('--split-column', default='split', help='Name of column to add for split assignment')
@click.option('--split-type-column', default='split_type', help='Name of column to add for split type')
@click.option('--random-state', default=42, type=int, help='Random seed for reproducibility')
@click.option('--verbose', is_flag=True, default=False, help='Verbose output')
def main(**kwargs):
    """Split data into train/validation (and optionally test) sets using AutoMol validation strategies.

    The split assignments are added as columns to the input CSV:
    - `split_column` (default: "split"): Values are 'train', 'valid', or 'test'
    - `split_type_column` (default: "split_type"): Values indicate the split type
    """
    import numpy as np
    import pandas as pd
    from automol.validation import stratified_validation, leave_grp_out_validation, mixed_validation
    from automol.clustering import MurckoScaffoldClustering, ButinaSplitReassigned, HierarchicalButina, KmeansForSmiles
    from automol.feature_generators import retrieve_default_offline_generators

    # Extract arguments
    csv_file = kwargs['csv_file']
    output_folder = kwargs['output_folder']
    sep = kwargs['separator']
    smiles_column = kwargs['smiles_column']
    properties = list(kwargs['properties'])
    split_strategy = kwargs['split_strategy']
    clustering_method = kwargs['clustering_method']
    test_size = kwargs['test_size']
    add_test_set = kwargs['add_test_set']
    test_size_test = kwargs['test_size_test']
    n_clusters = kwargs['n_clusters']
    butina_cutoff = kwargs['butina_cutoff']
    include_chirality = kwargs['include_chirality']
    split_column = kwargs['split_column']
    split_type_column = kwargs['split_type_column']
    random_state = kwargs['random_state']
    verbose = kwargs['verbose']

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Read CSV file
    if verbose:
        print(f'Reading CSV file: {csv_file}')
    if csv_file.endswith('.gz'):
        df = pd.read_csv(csv_file, sep=sep, na_values=['NAN', '?', 'NaN'], compression='gzip')
    else:
        df = pd.read_csv(csv_file, sep=sep, na_values=['NAN', '?', 'NaN'])

    # Validate required columns exist
    if smiles_column not in df.columns:
        raise ValueError(f'SMILES column "{smiles_column}" not found in dataframe. Available columns: {df.columns.tolist()}')

    missing_props = [p for p in properties if p not in df.columns]
    if missing_props:
        raise ValueError(f'Properties {missing_props} not found in dataframe. Available columns: {df.columns.tolist()}')

    if verbose:
        print(f'Data shape: {df.shape}')
        print(f'Properties: {properties}')
        print(f'Split strategy: {split_strategy}')
        print(f'Clustering method: {clustering_method}')

    # Initialize feature generators
    if verbose:
        print('Initializing feature generators...')
    feature_generators = retrieve_default_offline_generators(radius=2, nbits=2048)

    # Create clustering algorithm
    if clustering_method == 'Scaffold':
        clustering_algorithm = MurckoScaffoldClustering(include_chirality=include_chirality)
    elif clustering_method == 'Butina':
        clustering_algorithm = ButinaSplitReassigned(cutoff=butina_cutoff)
    elif clustering_method == 'HierarchicalButina':
        clustering_algorithm = HierarchicalButina(cutoff=butina_cutoff)
    elif clustering_method == 'Bottleneck':
        clustering_algorithm = KmeansForSmiles(
            n_groups=n_clusters,
            feature_generators=feature_generators,
            used_features=['Bottleneck'],
            random_state=random_state
        )
    else:
        raise ValueError(f'Unknown clustering method: {clustering_method}')

    # Perform split based on strategy
    if verbose:
        print(f'Computing {split_strategy} split...')

    # Helper function to apply split strategy and add split columns
    def apply_split_strategy(data, test_frac, split_label, split_type_value='not_specified'):
        """Apply the selected split strategy to data and return train/validation with split columns.

        Args:
            data: Input dataframe
            test_frac: Fraction for validation set
            split_label: Label to use for the validation set ('valid' or 'test')
            split_type_value: Value to use for split_type column (used for 'test' set)

        Returns:
            tuple of (train_df, valid_df, lgo_indices, activity_cliffs_index)
            The returned dataframes have 'split' and 'split_type' columns added
        """
        # Store original columns to identify which are new
        original_cols = set(data.columns)

        if split_strategy == 'stratified':
            train, valid = stratified_validation(
                data,
                properties=properties,
                standard_smiles_column=smiles_column,
                test_size=test_frac,
                feature_generators=feature_generators,
                clustering=clustering_method,
                n_clusters=n_clusters,
                cutoff=butina_cutoff,
                include_chirality=include_chirality,
                verbose=int(verbose),
                random_state=random_state,
                clustering_algorithm=clustering_algorithm
            )
            # Add split columns
            train[split_column] = 'train'
            train[split_type_column] = 'stratified'
            valid[split_column] = split_label
            valid[split_type_column] = 'stratified'
            return train, valid, None, None

        elif split_strategy == 'leave_group_out':
            train, valid = leave_grp_out_validation(
                data,
                properties=properties,
                standard_smiles_column=smiles_column,
                test_size=test_frac,
                feature_generators=feature_generators,
                clustering=clustering_method,
                n_clusters=n_clusters,
                cutoff=butina_cutoff,
                include_chirality=include_chirality,
                verbose=int(verbose),
                random_state=random_state,
                clustering_algorithm=clustering_algorithm
            )
            # Add split columns
            train[split_column] = 'train'
            train[split_type_column] = 'leave_group_out'
            valid[split_column] = split_label
            valid[split_type_column] = 'leave_group_out'
            return train, valid, None, None

        elif split_strategy == 'mixed':
            train, valid, lgo_indices, activity_cliffs_index = mixed_validation(
                data,
                properties=properties,
                standard_smiles_column=smiles_column,
                test_size=test_frac,
                feature_generators=feature_generators,
                clustering=clustering_method,
                n_clusters=n_clusters,
                cutoff=butina_cutoff,
                include_chirality=include_chirality,
                verbose=int(verbose),
                random_state=random_state,
                clustering_algorithm=clustering_algorithm
            )

            # For the test set in 3-way split, use a simple split_type
            if split_label == 'test':
                train[split_column] = 'train'
                train[split_type_column] = 'not_specified'
                valid[split_column] = split_label
                valid[split_type_column] = split_type_value
            else:
                # For validation set, mark split types based on mixed strategy
                train[split_column] = 'train'
                train[split_type_column] = 'not_specified'

                # Start with 'stratified' for all validation samples
                valid[split_column] = split_label
                valid[split_type_column] = 'stratified'

                # Mark leave-group-out samples
                if lgo_indices is not None and len(lgo_indices) > 0:
                    # lgo_indices are relative to the validation dataframe
                    valid.iloc[lgo_indices, valid.columns.get_loc(split_type_column)] = 'leave_group_out'

                # Mark activity cliff samples
                if activity_cliffs_index is not None:
                    for prop, cliff_indices in activity_cliffs_index.items():
                        if cliff_indices and len(cliff_indices) > 0:
                            valid.iloc[cliff_indices, valid.columns.get_loc(split_type_column)] = f'{prop}_activity_cliff'

            return train, valid, lgo_indices, activity_cliffs_index

    # Perform 2-way or 3-way split using nested main strategy
    if add_test_set:
        # 3-way split: test | (train/valid)
        if verbose:
            print(f'Creating 3-way split: test ({test_size_test*100}%) + validation ({test_size*100}%) + training ({(1-test_size_test-test_size)*100}%)')

        # Adjust validation fraction for the remaining data after test set removal
        valid_frac_remaining = test_size / (1 - test_size_test)

        # First split: separate test set from full dataset
        train_valid, test, _, _ = apply_split_strategy(df, test_size_test, 'test', split_type_value='not_specified')

        # Clean up train_valid for the second split
        # Drop 'index' column if it exists (added by validation functions)
        if 'index' in train_valid.columns:
            train_valid = train_valid.drop(columns=['index'])
        # Also drop the split columns we just added
        train_valid = train_valid.drop(columns=[split_column, split_type_column])

        # Create a fresh copy with sequential indices for the second split
        # This is needed because mixed_validation uses iloc with stored indices
        train_valid = train_valid.reset_index(drop=True)

        # Second split: get train/validation from remaining data
        train, valid, _, _ = apply_split_strategy(train_valid, valid_frac_remaining, 'valid')

        # Concatenate all splits - but first clean up the 'index' column from each
        for split_df in [train, valid, test]:
            if 'index' in split_df.columns:
                split_df.drop(columns=['index'], inplace=True)

        df = pd.concat([train, valid, test], ignore_index=True)

    else:
        # 2-way split: train/valid
        if verbose:
            print(f'Creating 2-way split: validation ({test_size*100}%) + training ({(1-test_size)*100}%)')

        train, valid, _, _ = apply_split_strategy(df, test_size, 'valid')

        # Clean up the 'index' column from each split
        for split_df in [train, valid]:
            if 'index' in split_df.columns:
                split_df.drop(columns=['index'], inplace=True)

        # Concatenate train and valid
        df = pd.concat([train, valid], ignore_index=True)

    # Save output
    file_stem = Path(csv_file).stem
    # Handle case where file might have .csv already removed
    if file_stem.endswith('.csv'):
        file_stem = Path(file_stem).stem

    # Strip 'automol_prepared_' prefix to avoid cascading prefixes
    # (e.g., 'automol_split_automol_prepared_X' → 'automol_split_X')
    if file_stem.startswith('automol_prepared_'):
        file_stem = file_stem[len('automol_prepared_'):]

    output_path = str(Path(output_folder) / f'automol_split_{file_stem}.csv')
    df.to_csv(output_path, index=False)

    # Create split info JSON
    split_info = {
        'input_file': str(Path(csv_file).name),
        'dataset': file_stem,
        'properties': properties,
        'split_strategy': split_strategy,
        'clustering_method': clustering_method,
        'test_size': test_size,
        'n_clusters': n_clusters,
        'butina_cutoff': butina_cutoff,
        'include_chirality': include_chirality,
        'random_state': random_state,
        'has_test_set': add_test_set,
        'test_size_test': test_size_test if add_test_set else None,
        'train_size': int((df[split_column] == 'train').sum()),
        'validation_size': int((df[split_column] == 'valid').sum()),
        'test_size_count': int((df[split_column] == 'test').sum()) if add_test_set else 0,
        'smiles_column': smiles_column,
        'property_columns': list(properties),
        'split_column': split_column,
        'split_type_column': split_type_column,
    }

    info_path = str(Path(output_folder) / f'automol_split_{file_stem}_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    if verbose:
        print(f'\nSplit summary:')
        print(f'  Train: {split_info["train_size"]}')
        print(f'  Validation: {split_info["validation_size"]}')
        if add_test_set:
            print(f'  Test: {split_info["test_size_count"]}')
        print(f'\nSplit type distribution:')
        print(df[split_type_column].value_counts())
        print(f'\nSaved output to:')
        print(f'  {output_path}')
        print(f'  {info_path}')

    return f'Saved split data to {output_path} and info to {info_path}'


if __name__ == '__main__':
    print(main())
