# Predict Skill — Examples

## Example 1: Single Model + CSV File

**User**: "Predict properties for my new molecules in `new_compounds.csv`"

**Flow**:
1. Registry has 1 model: `raw_data-gamma1_gamma2_gamma3-20260207-1430`
2. Auto-selects it, shows summary
3. User provided CSV path — no need to ask
4. Runs: `predict.py --model-file MolagentFiles/raw_data-gamma1_gamma2_gamma3-20260207_1430/merged_refitted_stackingregmodel.pt --smiles-file new_compounds.csv --verbose`
5. Shows sample predictions table for all 3 gamma properties

## Example 2: Merged Model — Subset of Properties

**User**: "Predict just gamma1 for CCO and c1ccccc1"

**Flow**:
1. Registry has merged model with gamma1, gamma2, gamma3
2. Auto-selects it
3. SMILES provided inline
4. Runs: `predict.py --model-file .../merged_refitted_stackingregmodel.pt --smiles-list "CCO" --smiles-list "c1ccccc1" --properties gamma1 --verbose`
5. Shows predictions for gamma1 only

## Example 3: Multiple Models + Selection

**User**: "Predict for CCO"

**Flow**:
1. Registry has 3 models — AskUserQuestion: which model?
2. User picks `ChEMBL_SMILES-prop5-20260207-2314` (classification)
3. Runs: `predict.py --model-file MolagentFiles/ChEMBL_SMILES-prop5-20260207_2300/prop5_refitted_stackingclfmodel.pt --smiles-list "CCO" --verbose`
4. Shows classification prediction for prop5

## Example 4: Model With Blender Properties

**User**: "Use the solubility model to predict from `test.csv`"

**Flow**:
1. Registry entry has `blender_properties: ["LogP_exp"]`
2. Warns user: "This model uses blender properties (LogP_exp). Make sure your CSV has this column."
3. Runs with `--blender-properties LogP_exp`
4. predict.py reads LogP_exp values from the CSV automatically

## Example 5: No Models Trained

**User**: "Predict toxicity for aspirin"

**Flow**:
1. Registry is empty or doesn't exist
2. Response: "No trained models found. Use the `train-pipeline` skill to train a model first."
3. STOP
