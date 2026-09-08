<script lang="ts">
	import { uploadFile, type UploadResult, type DatasetEntry } from '$lib/api/client';
	import { datasets, refreshDatasets } from '$lib/stores/datasets.svelte';
	import FileUpload from '$lib/components/shared/FileUpload.svelte';
	let {
		onReady,
		blenderProperties = [],
	}: {
		onReady: (input: { type: 'file'; path: string; smiles_column?: string; blender_properties?: string[] } | { type: 'list'; smiles: string[] }) => void;
		blenderProperties?: string[];
	} = $props();

	const hasBlenders = $derived(blenderProperties.length > 0);
	let mode = $state<'file' | 'paste'>('file');
	let smilesText = $state('');
	let uploadedPath = $state<string | null>(null);
	let columns = $state<string[]>([]);
	let selectedColumn = $state('');
	let selectedBlenderCols = $state<Record<string, string>>({});

	$effect(() => { refreshDatasets(); });

	function onFileUploaded(result: UploadResult) {
		uploadedPath = result.dataset_id || result.path;
		columns = result.columns ?? [];
		applyColumnDefaults();
		if (columns.length === 0) {
			onReady({ type: 'file', path: uploadedPath! });
		}
	}

	function selectDataset(ds: DatasetEntry) {
		uploadedPath = ds.id;
		columns = ds.columns ?? [];
		applyColumnDefaults();
		if (columns.length === 0) {
			onReady({ type: 'file', path: ds.id });
		}
	}

	function applyColumnDefaults() {
		const guess = columns.find(c => c.toLowerCase().includes('smiles'))
			?? columns.find(c => c.toLowerCase().includes('molecule'))
			?? columns[0] ?? '';
		selectedColumn = guess;
		for (const bp of blenderProperties) {
			const match = columns.find(c => c === bp) ?? columns.find(c => c.toLowerCase() === bp.toLowerCase());
			if (match) selectedBlenderCols[bp] = match;
		}
		selectedBlenderCols = { ...selectedBlenderCols };
	}

	function confirmFile() {
		if (!uploadedPath) return;
		const bpCols = hasBlenders ? Object.values(selectedBlenderCols).filter(Boolean) : undefined;
		onReady({ type: 'file', path: uploadedPath, smiles_column: selectedColumn || undefined, blender_properties: bpCols });
	}

	function submitSmiles() { const lines = smilesText.split('\n').map(s => s.trim()).filter(Boolean); if (lines.length > 0) onReady({ type: 'list', smiles: lines }); }
</script>
<div class="si">
	<div class="mt">
		<button class:active={mode === 'file'} onclick={() => { mode = 'file'; }}>CSV File</button>
		<button class:active={mode === 'paste'} class:disabled-tab={hasBlenders} onclick={() => { if (!hasBlenders) mode = 'paste'; }} disabled={hasBlenders} title={hasBlenders ? 'This model requires blender property columns — use CSV input' : ''}>Paste SMILES</button>
	</div>
	{#if hasBlenders && mode === 'file'}
		<p class="blender-note">This model uses blender properties ({blenderProperties.join(', ')}). Your CSV must contain these columns.</p>
	{/if}
	{#if mode === 'file'}
		{#if uploadedPath}
			<div class="file-config">
				<p class="info">File: <span class="mono">{columns.length > 0 ? (uploadedPath.startsWith('ds_') ? uploadedPath : uploadedPath.split(/[\\/]/).pop()) : uploadedPath}</span></p>
				{#if columns.length > 0}
					<div class="col-picker">
						<label>SMILES Column</label>
						<select bind:value={selectedColumn}>
							{#each columns as col}
								<option value={col}>{col}</option>
							{/each}
						</select>
					</div>
					{#if hasBlenders}
						{#each blenderProperties as bp}
							<div class="col-picker">
								<label>Blender: {bp}</label>
								<select bind:value={selectedBlenderCols[bp]}>
									<option value="">— select —</option>
									{#each columns.filter(c => c !== selectedColumn) as col}
										<option value={col}>{col}</option>
									{/each}
								</select>
							</div>
						{/each}
					{/if}
					<button class="btn" onclick={confirmFile} disabled={hasBlenders && Object.values(selectedBlenderCols).some(v => !v)}>Run Predictions</button>
				{/if}
			</div>
		{:else}
			<FileUpload label="Upload CSV with SMILES" onUploaded={onFileUploaded} />
			{#if datasets.entries.length > 0}
				<div class="existing-ds">
					<span class="existing-label">Or select an existing dataset</span>
					<div class="ds-chips">
						{#each datasets.entries as ds}
							<button class="ds-chip" onclick={() => selectDataset(ds)}>
								<span class="ds-chip-name">{ds.filename}</span>
								<span class="ds-chip-meta">{ds.row_count} rows</span>
							</button>
						{/each}
					</div>
				</div>
			{/if}
		{/if}
	{:else}
		<textarea bind:value={smilesText} placeholder="Enter SMILES strings (one per line)" rows="6"></textarea>
		<button class="btn" onclick={submitSmiles} disabled={!smilesText.trim()}>Use these SMILES</button>
	{/if}
</div>
<style>
	.si { display: flex; flex-direction: column; gap: 12px; }
	.mt { display: flex; gap: 4px; }
	.mt button { padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px; background: none; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
	.mt button:hover:not(:disabled) { background: var(--bg-tertiary); }
	.mt button.active { background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }
	.mt button.disabled-tab { opacity: 0.4; cursor: not-allowed; }
	.blender-note { font-size: 12px; color: var(--text-muted); padding: 8px 12px; border-radius: 6px; background: rgba(168, 85, 247, 0.06); border: 1px solid rgba(168, 85, 247, 0.2); }
	textarea { padding: 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-family: var(--font-mono); font-size: 13px; resize: vertical; }
	textarea:focus { outline: none; border-color: var(--accent); }
	.info { font-size: 13px; color: var(--text-secondary); }
	.file-config { display: flex; flex-direction: column; gap: 10px; padding: 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); }
	.col-picker { display: flex; align-items: center; gap: 10px; }
	.col-picker label { font-size: 12px; font-weight: 500; color: var(--text-secondary); }
	.col-picker select { padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 13px; }
	.col-picker select:focus { outline: none; border-color: var(--accent); }
	.btn { padding: 8px 18px; border: none; border-radius: var(--radius); background: var(--accent); color: white; font-size: 13px; font-weight: 600; cursor: pointer; }
	.btn:hover { background: var(--accent-hover); }
	.btn:disabled { opacity: 0.5; cursor: not-allowed; }
	.existing-ds { border-top: 1px solid var(--border); padding-top: 12px; display: flex; flex-direction: column; gap: 8px; }
	.existing-label { font-size: 12px; color: var(--text-muted); }
	.ds-chips { display: flex; flex-wrap: wrap; gap: 6px; }
	.ds-chip { display: flex; flex-direction: column; gap: 1px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-secondary); cursor: pointer; text-align: left; transition: border-color 120ms, background 120ms; }
	.ds-chip:hover { border-color: var(--accent); background: var(--accent-dim); }
	.ds-chip-name { font-size: 11px; font-weight: 500; color: var(--text-primary); font-family: var(--font-mono); }
	.ds-chip-meta { font-size: 10px; color: var(--text-muted); }
</style>