<script lang="ts">
	import ModelPicker from '$lib/components/predict/ModelPicker.svelte';
	import SmilesInput from '$lib/components/predict/SmilesInput.svelte';
	import PredictResults from '$lib/components/predict/PredictResults.svelte';
	import JobSpinner from '$lib/components/shared/JobSpinner.svelte';
	import { runPredict, downloadPredictions, type ModelEntry, type JobStatus } from '$lib/api/client';

	let selectedModel = $state<ModelEntry | null>(null);
	let predictJobId = $state<string | null>(null);
	let csvText = $state<string | null>(null);
	let error = $state<string | null>(null);
	let lastJobId = $state<string | null>(null);
	let inputKey = $state(0);
	let convertBack = $state(true);

	function onModelSelect(entry: ModelEntry) {
		selectedModel = entry;
		csvText = null;
		predictJobId = null;
		error = null;
		lastJobId = null;
		inputKey++;
	}

	async function onInputReady(input: { type: string; path?: string; smiles?: string[]; smiles_column?: string; blender_properties?: string[] }) {
		if (!selectedModel) return;
		error = null;
		csvText = null;
		try {
			const { job_id } = await runPredict({
				model_id: selectedModel.id,
				smiles_file: input.type === 'file' ? input.path : undefined,
				smiles_list: input.type === 'list' ? input.smiles : undefined,
				smiles_column: input.smiles_column,
				blender_properties: input.blender_properties,
				compute_sd: true,
				convert_log10: convertBack ? undefined : false,
			});
			predictJobId = job_id;
		} catch (e) { error = e instanceof Error ? e.message : 'Prediction failed'; }
	}

	async function onPredictDone(job: JobStatus) {
		predictJobId = null;
		lastJobId = job.id;
		if (job.status === 'success') {
			try { csvText = await downloadPredictions(job.id); }
			catch (e) { error = e instanceof Error ? e.message : 'Failed to load results'; }
		} else {
			error = 'Prediction failed';
		}
	}

	function newPrediction() {
		csvText = null;
		error = null;
		lastJobId = null;
		inputKey++;
	}
</script>
<div class="pp">
	<h2>Predict</h2>
	<ModelPicker onSelect={onModelSelect} />
	{#if selectedModel}
		<div class="ms"><span class="mono">{selectedModel.id}</span><span class="tag">{selectedModel.task_type}</span><span>{selectedModel.target_properties?.join(', ')}</span></div>
		{#if selectedModel.classification?.class_values}
			<div class="class-info">
				<span class="class-info-label">Class thresholds:</span>
				{#each selectedModel.target_properties as prop}
					{@const labelnames = selectedModel.classification?.labelnames?.[`Class_${prop}`] ?? selectedModel.classification?.labelnames?.[prop]}
					{@const values = selectedModel.classification?.class_values}
					{#if values}
						<span class="class-info-prop">{prop}:</span>
						{#each Array.from({ length: (values.length ?? 0) + 1 }) as _, i}
							{@const lo = i === 0 ? null : values[i - 1]}
							{@const hi = i === values.length ? null : values[i]}
							{@const name = labelnames?.[String(i)] ?? `Class ${i}`}
							<span class="class-chip">{name}{#if lo == null} &lt; {hi}{:else if hi == null} &gt; {lo}{:else} {lo}–{hi}{/if}</span>
						{/each}
					{/if}
				{/each}
			</div>
		{/if}
		{#if !csvText && !predictJobId}
			<label class="toggle-row">
				<input type="checkbox" bind:checked={convertBack} />
				<span>Convert to original scale</span>
				<span class="toggle-hint">{convertBack ? 'predictions in original units' : 'predictions in training space (log₁₀ / logit)'}</span>
			</label>
			{#key inputKey}
				<SmilesInput onReady={onInputReady} blenderProperties={selectedModel.blender_properties ?? []} />
			{/key}
		{/if}
	{/if}
	{#if predictJobId}<JobSpinner jobId={predictJobId} label="Running predictions..." onDone={onPredictDone} />{/if}
	{#if csvText}
		<PredictResults {csvText} />
		<div class="result-actions">
			<button class="btn-secondary" onclick={newPrediction}>New Prediction</button>
			{#if lastJobId}
				<a class="btn-download" href="/api/predict/{lastJobId}/download" download>Download CSV</a>
			{/if}
		</div>
	{/if}
	{#if error}<p class="err">{error}</p>{/if}
</div>
<style>
	.pp { max-width: 900px; display: flex; flex-direction: column; gap: 20px; }
	h2 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
	.ms { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--text-secondary); padding: 10px 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); }
	.tag { font-size: 10px; padding: 1px 6px; border-radius: 4px; background: var(--bg-tertiary); color: var(--text-muted); }
	.class-info { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 10px 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); font-size: 12px; color: var(--text-secondary); }
	.class-info-label { font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; }
	.class-info-prop { font-weight: 600; font-family: var(--font-mono); }
	.class-chip { padding: 2px 8px; border-radius: 4px; background: rgba(168, 85, 247, 0.08); color: var(--text-secondary); font-family: var(--font-mono); font-size: 11px; }
	.err { font-size: 13px; color: var(--error); padding: 12px; border-radius: var(--radius); background: rgba(239, 68, 68, 0.1); }
	.result-actions { display: flex; gap: 10px; align-items: center; }
	.btn-secondary { padding: 8px 16px; border: 1px solid var(--border); border-radius: var(--radius); background: none; color: var(--text-secondary); font-size: 13px; font-weight: 500; cursor: pointer; }
	.btn-secondary:hover { background: var(--bg-tertiary); }
	.btn-download { padding: 8px 16px; border-radius: var(--radius); background: var(--accent); color: white; font-size: 13px; font-weight: 600; text-decoration: none; }
	.btn-download:hover { background: var(--accent-hover); }
	.toggle-row { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-secondary); cursor: pointer; }
	.toggle-row input { accent-color: var(--accent); }
	.toggle-hint { font-size: 11px; color: var(--text-muted); font-style: italic; }
</style>