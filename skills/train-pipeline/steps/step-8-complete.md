# Step 8: Complete Pipeline

**Mode**: Inline (displays summary to user)

## Purpose

Generate the model card, register the model, display final summary, and mark pipeline as complete.

## Generate Model Card

Run the model card generation script:

```bash
uv run $MOLAGENT_PLUGIN_ROOT/skills/train-pipeline/scripts/generate_model_card.py \
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
uv run $MOLAGENT_PLUGIN_ROOT/skills/train-pipeline/scripts/model_registry.py register \
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
  - Make predictions: use the 'predict' skill (auto-discovers this model)
  - Deploy as API: see the model card for usage and deployment instructions
  - View model card: {model_card}
```

Note: The `.pt` model file is self-contained — encoder weights are baked in during training. You only need `automol` + `sklearn` + `rdkit` at inference time, not `automol-resources`.

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

## Done

The pipeline is complete. The user can now:
1. Use the `predict` skill to make predictions (it auto-discovers models from the registry)
2. Deploy as a REST API — see AutoMol deployment documentation for server, Docker, and cloud options
3. Read the model card for a quick-reference summary
4. Re-run the pipeline with different settings on the same data
