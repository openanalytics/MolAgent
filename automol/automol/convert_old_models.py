"""
AutoMol: Pipeline for automated machine learning for drug design.

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""
import os
import sys
import torch
from argparse import ArgumentParser

from . import model
from . import stacking_util
from . import dataset
from . import stacking

sys.modules['model'] = sys.modules['automol.model']
sys.modules['stacking'] = sys.modules['automol.stacking']
sys.modules['stacking_util'] = sys.modules['automol.stacking_util']
sys.modules['dataset'] = sys.modules['automol.dataset']

def convert(input_file, output_file, overwrite = False):
    if not overwrite and input_file == output_file:
        raise ValueError("input_file and output_file should be different")
    pt_contents = torch.load(input_file, map_location='cpu')
    torch.save(pt_contents, output_file)
    print(f"Successfully saved file {output_file}")

if __name__ == "__main__":
    """
     example:
     python -m automol.convert_old_models \
     --input_file automol/trained_models_old/JNJ_ENUM_SMILES_STEREO_ENCODER.pt \
     --output_file automol/trained_models/JNJ_ENUM_SMILES_STEREO_ENCODER.pt
    """
    parser = ArgumentParser("Convert files created with old (non-packaged) transformers code such that they are compatible with new jnj_auto_ml package")
    parser.add_argument("--input_file", dest="input_file", help="original pt file")
    parser.add_argument("--output_file", dest="output_file", help="converted pt file")

    args = parser.parse_args()
    convert(input_file = args.input_file, output_file = args.output_file)
