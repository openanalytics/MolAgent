
from pathlib import Path
import click
import json


def retrieve_prop_from_mol(mol, guesses, start, remove_q=True):
    """Extract property from molecule with qualifier handling."""
    val = None
    prop_dict = mol.GetPropsAsDict()
    for p in guesses:
        if p in prop_dict:
            val = mol.GetProp(p)
            break
    if val is None:
        for key in prop_dict.keys():
            if key.startswith(start):
                val = mol.GetProp(key)
                break
    if val is None:
        return None
    if remove_q:
        if val[0] == '<' or val[0] == '>':
            val = val[1:]
        if val[0] == '=':
            val = val[1:]
    return val


def load_3d_data_from_sdf(sdf_file, property_key='pChEMBL', pdb_key='pdb'):
    """Load 3D data from SDF file."""
    from rdkit import Chem
    import pandas as pd

    pdb = []
    original_smiles = []
    aff_val = []

    for mol in Chem.SDMolSupplier(sdf_file, removeHs=False):
        if mol is None:
            continue
        pdb_nm = mol.GetProp(pdb_key)
        pic50 = float(retrieve_prop_from_mol(mol, guesses=[property_key], start=property_key, remove_q=False))
        pdb.append(pdb_nm)
        aff_val.append(pic50)
        original_smiles.append(Chem.MolToSmiles(mol))

    return pd.DataFrame({
        'original_smiles': original_smiles,
        property_key: aff_val,
        'pdb': pdb
    })


@click.command()
@click.option('--csv-file', default='data.csv',
              help='csv file with the data (use --sdf-file for 3D data instead)')
@click.option('--sdf-file', default=None,
              help='Path to SDF file with 3D ligand structures (for 3D features)')
@click.option('--protein-folder', default=None,
              help='Path to folder containing protein PDB files (for 3D features)')
@click.option('--property-key', default='pChEMBL',
              help='Property key in SDF file to extract (for 3D features)')
@click.option('--pdb-key', default='pdb',
              help='pdb key in SDF file to extract (for 3D features)')
@click.option('--output-folder', default='MolagentFiles/',
              help='The folder to save files to')
@click.option('--separator', default=',',
              help='The separator of the csv file')
@click.option('--smiles-column', default='SMILES',
              help='The column contain the smiles in the csv')
@click.option('--standardized-smiles', default='Stand_SMILES',
              help='The name of the column that will contain the standardized smiles')
@click.option('--check-rdkit', is_flag=True, default=False,
              help='Check smiles for use with rdkit')
@click.option('--verbose', is_flag=True, default=False,
              help='Print more output')
@click.option('--properties', multiple=True, default=['Y'],
              help='Properties to be classified (can specify multiple)')
@click.option('--nb-classes', multiple=True, type=int, default=[2],
              help='Number of classes per property (must match properties order)')
@click.option('--class-values', multiple=True, type=float, default=None,
              help='Class boundary values (e.g., for 3 classes specify 2 values: --class-values 1.0 2.0)')
@click.option('--categorical', is_flag=True, default=False,
              help='Data is already categorical (no transformation needed)')
@click.option('--use-sample-weight', is_flag=True, default=False,
              help='Define sample weights for classes')
@click.option('--weighted-class-index', multiple=True, type=int, default=[0],
              help='Class index to weight per property (0-based, must match properties order)')
@click.option('--property-weight', multiple=True, default=[10],
              help='Weight value per property (must match properties order)')
def main(**kwargs):
    import numpy as np
    import pandas as pd
    from automol.property_prep import validate_rdkit_smiles, add_rdkit_standardized_smiles, ClassBuilder

    sdf_file = kwargs.get('sdf_file', None)
    out_dir = kwargs.get('output_folder', 'MolagentFiles/')
    verbose = kwargs.get('verbose', False)
    sep = kwargs.get('separator', ',')
    property_key = kwargs.get('property_key', 'pChEMBL')
    pdb_key = kwargs.get('pdb_key', 'pdb')
    protein_folder = kwargs.get('protein_folder', None)

    # Check if using SDF file for 3D data
    if sdf_file:
        if verbose:
            print(f'Loading 3D data from SDF file: {sdf_file}')
        file_name = sdf_file
        df = load_3d_data_from_sdf(sdf_file, property_key,pdb_key)
        smiles_column = 'original_smiles'
        # Override smiles_column for SDF data
        kwargs['smiles_column'] = smiles_column
        if verbose:
            print(f'Loaded {len(df)} molecules from SDF file')
    else:
        if 'csv_file' not in kwargs:
            raise ValueError('csv file not provided, please provide valid csv with the command line argument --csv-file')
        file_name = kwargs.get('csv_file', 'file.csv')
        smiles_column = kwargs.get('smiles_column', 'SMILES')

        # Read the data
        df = pd.read_csv(file_name, sep=sep, na_values=['NAN', '?', 'NaN'])

    standard_smiles_column = kwargs.get('standardized_smiles', 'Stand_SMILES')

    # Standardize SMILES
    add_rdkit_standardized_smiles(df, smiles_column, verbose=int(verbose), outname=standard_smiles_column)

    # Check RDKit compatibility if requested
    check_rdkit_desc = kwargs.get('check_rdkit', False)
    if check_rdkit_desc:
        validate_rdkit_smiles(df, standard_smiles_column, verbose=int(verbose))

    # Get properties
    # When using SDF file, default to the property_key instead of 'Y'
    # Click's multiple=True returns a tuple, check if only default 'Y' is present
    properties_input = kwargs.get('properties', ('Y',))
    if sdf_file and list(properties_input) == ['Y']:
        properties = [property_key]
    else:
        properties = list(properties_input)

    # Validate and prepare nb_classes
    nb_classes_input = kwargs.get('nb_classes', [2])
    if len(nb_classes_input) == 1 and len(properties) > 1:
        # Single value provided, use for all properties
        nb_classes = list(nb_classes_input) * len(properties)
    else:
        nb_classes = list(nb_classes_input)

    if len(nb_classes) != len(properties):
        raise ValueError(f'Number of nb_classes values ({len(nb_classes)}) must match number of properties ({len(properties)})')

    # Prepare class_values - should be None or a list of lists
    class_values_input = kwargs.get('class_values', None)
    if class_values_input:
        # User provided class values - need to distribute across properties
        class_values_list = list(class_values_input)
        if len(properties) == 1:
            class_values = [class_values_list]
        else:
            # Multiple properties - same values for all, or need proper handling
            # For now, use same values for all properties
            class_values = [class_values_list] * len(properties)
        use_quantiles = False
    else:
        # No class values provided - use quantile-based split
        class_values = None
        use_quantiles = True

    # Get categorical flag
    categorical = kwargs.get('categorical', False)

    # When using quantiles, set up quantile boundaries based on nb_classes
    if use_quantiles and not categorical:
        # For N classes, we need N-1 quantile boundaries
        class_quantiles = []
        for nb in nb_classes:
            # Create quantile boundaries (e.g., for 2 classes: [0.5], for 3 classes: [0.33, 0.67])
            quantiles = [i / nb for i in range(1, nb)]
            class_quantiles.append(quantiles)
        class_values = class_quantiles

    # Drop rows where all properties are NA
    df.dropna(inplace=True, how='all', subset=properties)
    df = df.reset_index(drop=True)

    # Build classes
    prop_builder = ClassBuilder(
        properties=properties,
        nb_classes=nb_classes,
        class_values=class_values,
        categorical=categorical,
        use_quantiles=use_quantiles,
        prefix='Class',
        min_allowed_class_samples=30,
        verbose=int(verbose)
    )

    # Check properties exist
    prop_builder.check_properties(df)

    # Generate class labels
    df, train_properties = prop_builder.generate_train_properties(df)

    # Handle sample weights
    use_sample_weight = kwargs.get('use_sample_weight', False)
    if use_sample_weight:
        # Validate and prepare weighted_class_index
        weighted_class_index_input = kwargs.get('weighted_class_index', [0])
        if len(weighted_class_index_input) == 1 and len(train_properties) > 1:
            weighted_class_index_list = list(weighted_class_index_input) * len(train_properties)
        else:
            weighted_class_index_list = list(weighted_class_index_input)

        # Validate and prepare property_weight
        property_weight_input = kwargs.get('property_weight', [10])
        if len(property_weight_input) == 1 and len(train_properties) > 1:
            property_weight_list = list(property_weight_input) * len(train_properties)
        else:
            property_weight_list = list(property_weight_input)

        if len(weighted_class_index_list) != len(train_properties):
            raise ValueError(f'Number of weighted_class_index values ({len(weighted_class_index_list)}) must match number of train_properties ({len(train_properties)})')

        if len(property_weight_list) != len(train_properties):
            raise ValueError(f'Number of property_weight values ({len(property_weight_list)}) must match number of train_properties ({len(train_properties)})')

        # For categorical data, convert indices to labels
        if categorical:
            weighted_samples_index = {p: str(weighted_class_index_list[i]) for i, p in enumerate(train_properties)}
        else:
            weighted_samples_index = {p: weighted_class_index_list[i] for i, p in enumerate(train_properties)}

        select_sample_weights = {p: property_weight_list[i] for i, p in enumerate(train_properties)}

        df = prop_builder.generate_sample_weights(df, weighted_samples_index, select_sample_weights)

    # Create output directory if needed
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save output
    file_stem = Path(file_name).stem
    output_path = str(Path(out_dir) / f'automol_prepared_{file_stem}.csv')
    df.to_csv(output_path, index=False)

    # Build labelnames mapping for downstream use (train_classification.py)
    labelnames = {}
    for tp in train_properties:
        if tp in df.columns:
            unique_values = sorted(df[tp].dropna().unique())
            labelnames[tp] = {int(i): str(v) for i, v in enumerate(unique_values)}

    # Create preparation info JSON
    prep_info = {
        'input_file': str(Path(file_name).name),
        'dataset': file_stem,
        'properties': properties,
        'train_properties': train_properties,
        'labelnames': labelnames,
        'smiles_column': smiles_column,
        'standardized_smiles_column': standard_smiles_column,
        'nb_classes': nb_classes,
        'class_values': class_values,
        'use_quantiles': use_quantiles,
        'categorical': categorical,
        'use_sample_weight': use_sample_weight,
        'check_rdkit': check_rdkit_desc,
        'n_samples': len(df),
    }

    # Add 3D data info if SDF file was used
    if sdf_file:
        prep_info['preparation_type'] = '3d_data_prep'
        prep_info['sdf_file'] = sdf_file
        if protein_folder:
            prep_info['protein_folder'] = protein_folder
        prep_info['property_key'] = property_key

    # Add sample weight info if used
    if use_sample_weight:
        prep_info['sample_weights'] = {
            'weighted_class_index': weighted_samples_index,
            'weights': select_sample_weights
        }

    info_path = str(Path(out_dir) / f'automol_prepared_{file_stem}_info.json')
    with open(info_path, 'w') as f:
        json.dump(prep_info, f, indent=2)

    if verbose:
        print(f'Saved the prepared csv file to {output_path}')
        print(f'Saved preparation info to {info_path}')

    return f'Saved the prepared csv file to {output_path} and info to {info_path}'


if __name__ == "__main__":
    print(main())
