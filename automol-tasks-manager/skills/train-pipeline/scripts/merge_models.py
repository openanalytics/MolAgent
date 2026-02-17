#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "torch"]
# ///
"""
Merge per-property AutoMol .pt model files into a single multi-property file.

Each per-property file contains the full model (encoder + all estimators).
This script uses upstream merge_model()/delete_properties() to produce one
combined file, eliminating encoder duplication and enabling single-call
multi-property prediction.
"""

import json
import re
import shutil
import sys
from pathlib import Path

import click


def infer_property_name(model_path):
    """Infer property name from model filename.

    Strips suffixes: _stackingregmodel.pt, _stackingclfmodel.pt, _refitted
    """
    name = Path(model_path).stem  # e.g. gamma1_refitted_stackingregmodel
    # Remove model type suffix
    name = re.sub(r'_stacking(reg|clf)model$', '', name)
    # Remove _refitted suffix
    name = re.sub(r'_refitted$', '', name)
    return name


@click.command()
@click.option('--model-files', multiple=True, required=True,
              help='Per-property .pt files to merge (repeat for each file)')
@click.option('--properties', multiple=True, default=None,
              help='Property names in same order as model-files (default: infer from filenames)')
@click.option('--output-file', required=True,
              help='Output merged .pt file path')
@click.option('--verify-encoder/--no-verify-encoder', default=True,
              help='Assert encoder state_dicts match across all models')
@click.option('--cleanup/--no-cleanup', default=False,
              help='Delete individual files after successful merge')
@click.option('--verbose/--no-verbose', default=False,
              help='Verbose output')
def main(**kwargs):
    """Merge per-property AutoMol model files into a single multi-property file.

    Takes N per-property .pt files (each containing the full model with
    identical pretrained encoder) and merges them into one file using
    upstream merge_model() and delete_properties().

    Single-property short-circuit: if only one --model-files is provided,
    copies it directly to --output-file.

    Examples:
        # Merge 3 property models
        uv run python merge_models.py \\
            --model-files MolagentFiles/gamma1_stackingregmodel.pt \\
            --model-files MolagentFiles/gamma2_stackingregmodel.pt \\
            --model-files MolagentFiles/gamma3_stackingregmodel.pt \\
            --output-file MolagentFiles/merged_stackingregmodel.pt \\
            --verify-encoder --verbose

        # Merge refitted models with cleanup
        uv run python merge_models.py \\
            --model-files MolagentFiles/gamma1_refitted_stackingregmodel.pt \\
            --model-files MolagentFiles/gamma2_refitted_stackingregmodel.pt \\
            --output-file MolagentFiles/merged_refitted_stackingregmodel.pt \\
            --cleanup --verbose
    """
    import torch
    from automol.stacking import load_model, save_model

    model_files = list(kwargs['model_files'])
    properties_cli = list(kwargs['properties']) if kwargs['properties'] else None
    output_file = kwargs['output_file']
    verify_encoder = kwargs['verify_encoder']
    cleanup = kwargs['cleanup']
    verbose = kwargs['verbose']

    # Validate inputs
    if not model_files:
        print("Error: at least one --model-files required", file=sys.stderr)
        sys.exit(1)

    for mf in model_files:
        if not Path(mf).exists():
            print(f"Error: model file not found: {mf}", file=sys.stderr)
            sys.exit(1)

    # Infer property names if not provided
    if properties_cli and len(properties_cli) != len(model_files):
        print(f"Error: --properties count ({len(properties_cli)}) must match "
              f"--model-files count ({len(model_files)})", file=sys.stderr)
        sys.exit(1)

    properties = properties_cli or [infer_property_name(mf) for mf in model_files]

    if verbose:
        print(f"Model files: {model_files}")
        print(f"Properties:  {properties}")
        print(f"Output:      {output_file}")

    # Single-property short-circuit
    if len(model_files) == 1:
        if verbose:
            print("Single model file — copying directly (no merge needed)")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_files[0], output_file)
        result = {
            "merged_model": output_file,
            "properties": properties,
            "source_files": model_files,
            "merged": False,
        }
        print(json.dumps(result))
        return

    # Load all models
    if verbose:
        print(f"\nLoading {len(model_files)} models...")

    models = []
    for mf in model_files:
        if verbose:
            print(f"  Loading {mf}...")
        m = load_model(mf, use_gpu=False)
        models.append(m)

    # Verify encoder identity
    if verify_encoder and len(models) > 1:
        if verbose:
            print("\nVerifying encoder identity across models...")
        base_fg = models[0].feature_generators
        for i, m in enumerate(models[1:], 1):
            for key in base_fg:
                if key not in m.feature_generators:
                    continue
                gen_base = base_fg[key]
                gen_other = m.feature_generators[key]
                # Compare state_dicts if they are torch modules
                if hasattr(gen_base, 'state_dict') and hasattr(gen_other, 'state_dict'):
                    sd_base = gen_base.state_dict()
                    sd_other = gen_other.state_dict()
                    for param_name in sd_base:
                        if param_name not in sd_other:
                            print(f"Warning: encoder param '{param_name}' missing in model {i} ({model_files[i]})",
                                  file=sys.stderr)
                            continue
                        if not torch.equal(sd_base[param_name], sd_other[param_name]):
                            print(f"Error: encoder param '{param_name}' differs between "
                                  f"model 0 ({model_files[0]}) and model {i} ({model_files[i]})",
                                  file=sys.stderr)
                            sys.exit(1)
        if verbose:
            print("  Encoders verified identical.")

    # Build merged model
    if verbose:
        print(f"\nMerging {len(models)} models...")

    base_model = models[0]
    base_prop = properties[0]

    # Keep only the base property in the base model
    props_to_remove = [p for p in list(base_model.models.keys()) if p != base_prop]
    if props_to_remove:
        if verbose:
            print(f"  Base model: keeping '{base_prop}', removing {props_to_remove}")
        base_model.delete_properties(props_to_remove)

    # Merge each subsequent model's target property
    for i in range(1, len(models)):
        donor = models[i]
        prop = properties[i]

        # Isolate the target property in the donor
        donor_props_to_remove = [p for p in list(donor.models.keys()) if p != prop]
        if donor_props_to_remove:
            if verbose:
                print(f"  Donor model {i}: keeping '{prop}', removing {donor_props_to_remove}")
            donor.delete_properties(donor_props_to_remove)

        if verbose:
            print(f"  Merging '{prop}' from {model_files[i]}...")
        base_model.merge_model(donor, other_props=[prop])

    # Verify merged model has all properties
    merged_props = sorted(base_model.models.keys())
    expected_props = sorted(properties)
    if merged_props != expected_props:
        print(f"Error: merged model has properties {merged_props} "
              f"but expected {expected_props}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"  Merged model properties: {merged_props}")

    # Clean and save
    if verbose:
        print(f"\nCleaning and saving to {output_file}...")
    base_model.deep_clean()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_model(base_model, output_file)

    if verbose:
        size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        individual_size = sum(Path(mf).stat().st_size for mf in model_files) / (1024 * 1024)
        print(f"  Merged file size: {size_mb:.1f} MB")
        print(f"  Individual files total: {individual_size:.1f} MB")
        print(f"  Saved: {individual_size - size_mb:.1f} MB ({(1 - size_mb/individual_size)*100:.0f}%)")

    # Optional cleanup
    if cleanup:
        if verbose:
            print("\nCleaning up individual files...")
        for mf in model_files:
            Path(mf).unlink()
            if verbose:
                print(f"  Deleted {mf}")

    result = {
        "merged_model": output_file,
        "properties": properties,
        "source_files": model_files,
        "merged": True,
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
