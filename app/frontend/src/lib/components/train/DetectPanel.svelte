<script lang="ts">
	import { detectDataset } from '$lib/api/client';
	import { pipeline, applyDetectDefaults } from '$lib/stores/pipeline.svelte';

	let detecting = $state(false);

	async function startDetect() {
		if (!pipeline.csvPath && !pipeline.datasetId) return;
		detecting = true;
		pipeline.error = null;
		try {
			const opts = pipeline.datasetId
				? { dataset_id: pipeline.datasetId }
				: { csv_path: pipeline.csvPath! };
			const result = await detectDataset(opts);
			pipeline.detectResult = result;
			pipeline.config.csv_file = result.detected.csv_file ?? pipeline.csvPath ?? '';
			applyDetectDefaults(result);
			pipeline.step = 'configure';
		} catch (e) {
			pipeline.error = e instanceof Error ? e.message : 'Detection failed';
		} finally {
			detecting = false;
		}
	}

	$effect(() => {
		if ((pipeline.csvPath || pipeline.datasetId) && !pipeline.detectResult) {
			startDetect();
		}
	});

	const warnings = $derived(() => {
		const q = pipeline.detectResult?.question ?? '';
		const idx = q.indexOf('Warnings:\n');
		if (idx < 0) return [];
		return q.slice(idx + 'Warnings:\n'.length).split('\n').filter(l => l.startsWith('- ')).map(l => l.slice(2));
	});
</script>

{#if detecting}
	<div class="detect-loading">
		<span class="dot"></span>
		<span>Analyzing dataset...</span>
	</div>
{/if}

{#if pipeline.detectResult}
	{@const d = pipeline.detectResult.detected}
	{@const opts = pipeline.detectResult.options}
	{@const chars = pipeline.detectResult.characteristics}
	{@const targets = pipeline.detectResult.targets}
	<div class="detect-summary">
		<h3>Dataset Analysis</h3>
		<div class="detect-grid">
			<div class="detect-item">
				<span class="detect-label">SMILES Column</span>
				<span class="detect-value mono">{d.smiles_column || 'Not found'}</span>
			</div>
			<div class="detect-item">
				<span class="detect-label">Task Type</span>
				<span class="detect-value">{d.task}</span>
			</div>
			<div class="detect-item">
				<span class="detect-label">Targets</span>
				<span class="detect-value">{d.properties.length} properties</span>
			</div>
			<div class="detect-item">
				<span class="detect-label">Features</span>
				<span class="detect-value">{d.feature_keys.join(', ')}</span>
			</div>
			{#if chars}
				<div class="detect-item">
					<span class="detect-label">SMILES Validity</span>
					<span class="detect-value" class:warn={chars.smiles_validity_rate < 90}>
						{chars.valid_smiles}/{chars.total_smiles} ({chars.smiles_validity_rate}%)
					</span>
				</div>
			{/if}
		</div>

		{#if warnings().length > 0}
			<div class="detect-warnings">
				{#each warnings() as w}
					<p class="warning-item">{w}</p>
				{/each}
			</div>
		{/if}

		{#if targets && targets.length > 0}
			<div class="detect-targets">
				<h4>Detected Targets</h4>
				<div class="targets-chips">
					{#each targets as t}
						<div class="target-chip">
							<span class="mono">{t.column}</span>
							<span class="tag" class:tag-reg={t.task_type === 'regression'} class:tag-clf={t.task_type === 'classification'}>{t.task_type === 'regression' ? 'reg' : 'clf'}</span>
							{#if t.task_type === 'regression'}
								<span class="stat">{t.min?.toFixed(2)} – {t.max?.toFixed(2)}</span>
							{:else}
								<span class="stat">{t.suggested_nb_classes} classes</span>
							{/if}
							{#if t.null_count > 0}
								<span class="stat nan-warn">{t.null_count} NaN</span>
							{/if}
						</div>
					{/each}
				</div>
			</div>
		{:else}
			<div class="detect-targets">
				<h4>Available Columns</h4>
				<p class="cols-list">{opts.smiles_column.join(', ')}</p>
			</div>
		{/if}
	</div>
{/if}

<style>
	.detect-loading {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 16px;
		font-size: 14px;
		color: var(--text-secondary);
	}
	.dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--accent);
		animation: pulse 1.2s ease-in-out infinite;
	}
	.detect-summary {
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 20px;
		background: var(--bg-secondary);
	}
	h3 {
		font-size: 15px;
		font-weight: 600;
		margin-bottom: 16px;
		color: var(--text-primary);
	}
	h4 {
		font-size: 13px;
		font-weight: 600;
		margin-bottom: 8px;
		color: var(--text-secondary);
	}
	.detect-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 12px;
		margin-bottom: 16px;
	}
	.detect-item {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.detect-label {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-muted);
	}
	.detect-value {
		font-size: 14px;
		font-weight: 500;
		color: var(--text-primary);
	}
	.detect-value.warn {
		color: var(--error);
	}
	.detect-warnings {
		margin-bottom: 12px;
		padding: 10px 14px;
		border-radius: 6px;
		background: rgba(234, 179, 8, 0.08);
		border: 1px solid rgba(234, 179, 8, 0.3);
	}
	.warning-item {
		font-size: 12px;
		color: var(--text-secondary);
		margin: 4px 0;
	}
	.warning-item::before {
		content: '\26A0  ';
	}
	.detect-targets { margin-top: 16px; }
	.targets-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.target-chip {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 5px 10px;
		border-radius: 6px;
		border: 1px solid var(--border);
		background: var(--bg-primary);
		font-size: 12px;
	}
	.tag {
		font-size: 10px;
		padding: 1px 5px;
		border-radius: 3px;
		font-weight: 600;
		text-transform: uppercase;
	}
	.tag-reg { background: rgba(59, 130, 246, 0.12); color: #3b82f6; }
	.tag-clf { background: rgba(168, 85, 247, 0.12); color: #a855f7; }
	.stat {
		font-size: 11px;
		color: var(--text-muted);
		font-family: var(--font-mono);
	}
	.nan-warn {
		color: #d97706;
	}
	.cols-list {
		font-size: 12px;
		color: var(--text-muted);
		font-family: var(--font-mono);
		word-break: break-all;
	}
	@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
