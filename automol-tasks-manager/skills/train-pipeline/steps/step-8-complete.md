# Step 8: Complete Pipeline

**Mode**: Inline (displays summary to user)

## Purpose

Generate the model card, register the model, display final summary, and mark pipeline as complete.

## Generate Model Card

Run the model card generation script:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/generate_model_card.py \
    --pipeline-state {config.output_folder}/pipeline_state.json
```

This creates `{config.output_folder}/model_card.md` with:
- Model file path (refitted or original)
- Training configuration
- Evaluation metrics
- Usage instructions (predict skill)
- Deployment instructions

## Register Model

Register the trained model in the model registry for discoverability by the predict skill:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/model_registry.py register \
    --pipeline-state {config.output_folder}/pipeline_state.json
```

Capture the output JSON — it contains the `id` assigned to this model and the registry path.

Note: The registry file stays at `MolagentFiles/model_registry.json` (global, not inside the run folder).

## Display Final Summary

Present to the user:

```
Pipeline Complete!

Target: {properties}
Task type: {task_type}
Model: {best_model_path}

Metrics:
  {prop1}: {formatted_metrics}
  {prop2}: {formatted_metrics}

Generated Files:
  - Prepared data: {prepared_csv}
  - Split data: {split_csv}
  - Model: {model_file}
  - Refitted model: {refitted_model or "skipped"}
  - Model card: {model_card}

Run folder: {config.output_folder}
Model ID: {model_id}
Registry: MolagentFiles/model_registry.json

Next Steps:
  - Visualize results: use the 'visualize' skill to explore evaluation charts
  - Make predictions: use the 'predict' skill (auto-discovers this model)
  - Deploy as API: see `deploy/DEPLOYMENT.md` in the repo root for REST API, Docker, and cloud options (if available)
  - View model card: {model_card}
```

Note: The `.pt` model file is self-contained — encoder weights are baked in during training. You only need `automol` + `torch` + `sklearn` + `rdkit` at inference time, not `automol-resources`.

If multiple properties were trained, models are merged into a single `.pt` file (~10MB saved per extra property by eliminating encoder duplication). The predict skill handles both merged and individual model files transparently.

## Update State

```json
{
  "pipeline_complete": true,
  "steps_completed": [0, 1, 2, 3, 4, 5, 6, 7, 8],
  "current_step": 8,
  "last_updated": "{iso_timestamp}"
}
```

## Mark All Tasks Completed

Use TaskUpdate to mark all remaining pipeline tasks as completed.

## Ask About Visualization

After displaying the summary, ask the user if they want to visualize the results now:

```
AskUserQuestion:
  header: "Visualize"
  question: "Would you like to explore the evaluation results in an interactive dashboard?"
  options:
    - "Yes, open dashboard (Recommended) - Generate and open the HTML dashboard now"
    - "No, I'm done - Skip visualization"
```

If the user selects **"Yes"**: invoke the `visualize` skill inline, passing the current run's `pipeline_state.json` path so it skips discovery and goes straight to dashboard generation.

If the user selects **"No"**: display the closing message and stop.

## Done

The pipeline is complete. The user can now:
1. Use the `visualize` skill to explore evaluation charts in an interactive dashboard
2. Use the `predict` skill to make predictions (it auto-discovers models from the registry)
3. Deploy as a REST API — see `deploy/DEPLOYMENT.md` in the repo root (if available) for: direct server, Docker, Docker Compose (GPU/nginx/Redis), and Python client
4. Read the model card for a quick-reference summary
5. Re-run the pipeline with different settings on the same data
