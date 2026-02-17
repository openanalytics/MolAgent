
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


def load_3d_data_from_sdf(sdf_file, property_key='pChEMBL',pdb_key:str='pdb'):
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
              help='Properties to be modelled (can specify multiple)')
@click.option('--use-log10', is_flag=True, default=False,
              help='Apply log10 transformation to properties')
@click.option('--use-percentages', is_flag=True, default=False,
              help='Divide property by 100')
@click.option('--use-logit', is_flag=True, default=False,
              help='Apply logit transformation')
@click.option('--use-sample-weight', is_flag=True, default=False,
              help='Define sample weight')
@click.option('--property-selection', multiple=True, default=['<1'],
              help='Identifier to select the samples with provided weight, per property, must match order of properties! e.g. can be <v1 or  >v2 with v1, v2 chosen float numbers, for example 1.0 and 4.3 ')
@click.option('--property-weight', multiple=True, default=[10],
              help='Identifier to define provided weight, per property, must match order of properties! ')
@click.option('--blender-properties', multiple=True, default=None,
              help='Additional properties to pass to blender/final estimator (used as-is, no transformations)')
def main(**kwargs):
    import numpy as np, pandas as pd
    from  matplotlib import pyplot as plt
    from automol.property_prep import validate_rdkit_smiles, add_rdkit_standardized_smiles

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

        if file_name.endswith('.gz'):
            df = pd.read_csv(file_name, sep=sep, na_values=['NAN', '?', 'NaN'], compression='gzip')
        else:
            df = pd.read_csv(file_name, sep=sep, na_values=['NAN', '?', 'NaN'])

    standard_smiles_column = kwargs.get('standardized_smiles', 'Stand_SMILES')
    add_rdkit_standardized_smiles(df, smiles_column, verbose=verbose, outname=standard_smiles_column)

    check_rdkit_desc = kwargs.get('check_rdkit',False)
    if check_rdkit_desc:
        validate_rdkit_smiles(df, standard_smiles_column,verbose=verbose)

    # When using SDF file, default to the property_key instead of 'Y'
    # Click's multiple=True returns a tuple, check if only default 'Y' is present
    properties_input = kwargs.get('properties', ('Y',))
    if sdf_file and list(properties_input) == ['Y']:
        properties = [property_key]
    else:
        properties = list(properties_input)

    use_log10=kwargs.get('use_log10',False)
    percentages=kwargs.get('use_percentages',False)
    use_logit=kwargs.get('use_logit',False)

    df.dropna(inplace=True,how='all', subset = properties)
    df = df.reset_index(drop=True)

    # Check for negative or zero values before applying log10 transformation
    if use_log10:
        for prop in properties:
            if prop in df.columns:
                min_val = df[prop].min()
                if min_val <= 0:
                    print(f'WARNING: Property "{prop}" has non-positive values (min: {min_val}).')
                    print(f'Log10 transformation requires positive values. Disabling log10 transformation for this property.')
                    print(f'The data may already be in log space (e.g., log permeability, pIC50).')
                    print(f'Proceeding without log10 transformation.')
                    use_log10 = False
                    break

    from automol.property_prep import PropertyTransformer

    prop_builder=PropertyTransformer(properties,use_log10=use_log10,use_logit=use_logit,percentages=percentages)

    prop_builder.check_properties(df)        
    df,train_properties=prop_builder.generate_train_properties(df)

    use_sample_weight=kwargs.get('use_sample_weight',False)
    weighted_samples_index={p:v for p,v in zip(train_properties, kwargs.get('property_selection',['<1']))}
    select_sample_weights={p:v for p,v in zip(train_properties, kwargs.get('property_weight',['<1']))}
    if use_sample_weight:
        df=prop_builder.generate_sample_weights(df,weighted_samples_index,select_sample_weights)

    # Handle blender properties - additional columns to pass to final estimator
    blender_properties = list(kwargs.get('blender_properties', [])) if kwargs.get('blender_properties') else []
    if blender_properties and verbose:
        print(f'Blender properties specified: {blender_properties}')
    # Validate blender properties exist and are numeric (with warning only)
    if blender_properties:
        for bp in blender_properties:
            if bp not in df.columns:
                print(f'Warning: Blender property "{bp}" not found in CSV columns. Available columns: {list(df.columns)}')
            else:
                # Check if numeric (with warning)
                if not pd.api.types.is_numeric_dtype(df[bp]):
                    print(f'Warning: Blender property "{bp}" is not numeric (dtype: {df[bp].dtype}). This may cause issues during training.')
                else:
                    # Check for missing values
                    if df[bp].isna().any():
                        na_count = df[bp].isna().sum()
                        print(f'Warning: Blender property "{bp}" has {na_count} missing values.')

    # Create output directory if needed
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save output
    file_stem=Path(file_name).stem  # Remove extension
    output_path=str(Path(out_dir) / f'automol_prepared_{file_stem}.csv')
    df.to_csv(output_path,index=False)

    # Create preparation info JSON
    prep_info = {
        'input_file': str(Path(file_name).name),
        'dataset': file_stem,
        'properties': list(train_properties),
        'smiles_column': smiles_column,
        'standardized_smiles_column': standard_smiles_column,
        'use_log10': use_log10,
        'use_percentages': percentages,
        'use_logit': use_logit,
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

    # Add blender properties info if specified
    if blender_properties:
        prep_info['blender_properties'] = blender_properties

    # Add sample weight info if used
    if use_sample_weight:
        prep_info['sample_weights'] = {
            'selection_criteria': weighted_samples_index,
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