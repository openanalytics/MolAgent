<script lang="ts">
	import { pipeline, resetPipeline } from '$lib/stores/pipeline.svelte';
	import { datasets, refreshDatasets } from '$lib/stores/datasets.svelte';
	import FileUpload from '$lib/components/shared/FileUpload.svelte';
	import DetectPanel from '$lib/components/train/DetectPanel.svelte';
	import ConfigPanel from '$lib/components/train/ConfigPanel.svelte';
	import PipelineSteps from '$lib/components/train/PipelineSteps.svelte';
	import FilePickerBox from '$lib/components/shared/FilePickerBox.svelte';
	import { uploadStructure } from '$lib/api/client';
	import type { UploadResult, DatasetEntry } from '$lib/api/client';

	let uploadMode = $state<'csv' | 'structure'>('csv');
	let targetName = $state('');
	let sdfFiles = $state<File[]>([]);
	let pdbFiles = $state<File[]>([]);
	let ligandsCsvFiles = $state<File[]>([]);
	let sdfFile = $derived(sdfFiles[0] ?? null);
	let ligandsCsv = $derived(ligandsCsvFiles[0] ?? null);
	let structureUploading = $state(false);
	let structureError = $state<string | null>(null);

	$effect(() => { if (pipeline.step === 'upload') refreshDatasets(); });

	function onFileUploaded(result: UploadResult) {
		pipeline.datasetId = result.dataset_id || null;
		pipeline.csvPath = result.path;
		pipeline.csvFilename = result.filename;
		pipeline.step = 'detect';
	}

	function selectDataset(ds: DatasetEntry) {
		pipeline.datasetId = ds.id;
		pipeline.csvPath = null;
		pipeline.csvFilename = ds.filename;
		pipeline.step = 'detect';
	}


	async function submitStructure() {
		if (!targetName || !sdfFile || pdbFiles.length === 0) return;
		structureUploading = true;
		structureError = null;
		try {
			const result = await uploadStructure(targetName, sdfFile, pdbFiles, ligandsCsv ?? undefined);
			pipeline.datasetId = result.structure_id;
			pipeline.csvPath = null;
			pipeline.csvFilename = result.target_name;
			pipeline.step = 'detect';
		} catch (e) {
			structureError = e instanceof Error ? e.message : 'Structure upload failed';
		} finally {
			structureUploading = false;
		}
	}
</script>
<div class="tp">
	<div class="ph"><h2>Train Model</h2>{#if pipeline.step !== 'upload'}<button class="btn-reset" onclick={resetPipeline}>New Pipeline</button>{/if}</div>
	{#if pipeline.step === 'upload'}
		<div class="upload-mode-toggle">
			<button class="mode-btn" class:active={uploadMode === 'csv'} onclick={() => uploadMode = 'csv'}>Upload CSV</button>
			<button class="mode-btn" class:active={uploadMode === 'structure'} onclick={() => uploadMode = 'structure'}>Upload 3D Structure</button>
		</div>
		{#if uploadMode === 'csv'}
			<FileUpload label="Upload training dataset (CSV)" onUploaded={onFileUploaded} />
			{#if datasets.entries.length > 0}
				<div class="existing-datasets">
					<h3>Or select an existing dataset</h3>
					<div class="ds-list">
						{#each datasets.entries as ds}
							<button class="ds-card" onclick={() => selectDataset(ds)}>
								<span class="ds-name">{ds.filename}</span>
								<span class="ds-meta">{ds.row_count.toLocaleString()} rows &middot; {ds.columns.length} cols</span>
							</button>
						{/each}
					</div>
				</div>
			{/if}
		{:else}
			<div class="structure-form">
				<div class="ff"><label>Target Name</label><input type="text" bind:value={targetName} placeholder="e.g. ABL" /></div>
				<div class="pickers">
					<div class="picker-col">
						<span class="picker-label">SDF File <span class="req">*</span></span>
						<FilePickerBox label="SDF (docked ligands)" accept=".sdf" bind:files={sdfFiles} />
					</div>
					<div class="picker-col">
						<span class="picker-label">PDB Files <span class="req">*</span></span>
						<FilePickerBox label="PDB (protein structures)" hint="select all at once" accept=".pdb" multiple bind:files={pdbFiles} />
					</div>
					<div class="picker-col">
						<span class="picker-label">Ligands CSV <span class="opt">(optional)</span></span>
						<FilePickerBox label="Ligands CSV" accept=".csv" bind:files={ligandsCsvFiles} />
					</div>
				</div>
				{#if structureError}<p class="err">{structureError}</p>{/if}
				<button class="btn-primary" onclick={submitStructure} disabled={structureUploading || !targetName || !sdfFile || pdbFiles.length === 0}>
					{structureUploading ? 'Uploading...' : 'Upload Structure'}
				</button>
			</div>
		{/if}
	{/if}
	{#if pipeline.step === 'detect' || pipeline.step === 'configure' || pipeline.step === 'training' || pipeline.step === 'done'}
		{#if pipeline.csvFilename}<p class="fi">Dataset: <span class="mono">{pipeline.csvFilename}</span></p>{/if}
		<DetectPanel />
	{/if}
	{#if pipeline.step === 'configure'}<ConfigPanel />{/if}
	{#if pipeline.step === 'training' || pipeline.step === 'done'}<PipelineSteps />{/if}
	{#if pipeline.error}<p class="err">{pipeline.error}</p>{/if}
</div>
<style>
	.tp { max-width: 900px; display: flex; flex-direction: column; gap: 20px; }
	.ph { display: flex; align-items: center; justify-content: space-between; }
	h2 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
	.btn-reset { padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px; background: none; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
	.btn-reset:hover { background: var(--bg-tertiary); }
	.fi { font-size: 13px; color: var(--text-secondary); }
	.err { font-size: 13px; color: var(--error); padding: 12px; border-radius: var(--radius); background: rgba(239, 68, 68, 0.1); }
	.existing-datasets { border-top: 1px solid var(--border); padding-top: 16px; }
	.existing-datasets h3 { font-size: 13px; font-weight: 500; color: var(--text-muted); margin-bottom: 10px; }
	.ds-list { display: flex; flex-wrap: wrap; gap: 8px; }
	.ds-card { display: flex; flex-direction: column; gap: 2px; padding: 10px 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); cursor: pointer; text-align: left; transition: border-color 120ms, background 120ms; }
	.ds-card:hover { border-color: var(--accent); background: var(--accent-dim); }
	.ds-name { font-size: 12px; font-weight: 500; color: var(--text-primary); font-family: var(--font-mono); }
	.ds-meta { font-size: 11px; color: var(--text-muted); }
	.upload-mode-toggle { display: flex; gap: 8px; }
	.mode-btn { padding: 8px 16px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-secondary); color: var(--text-secondary); font-size: 13px; font-weight: 500; cursor: pointer; }
	.mode-btn.active { border-color: var(--accent); background: var(--accent-dim); color: var(--text-primary); }
	.structure-form { display: flex; flex-direction: column; gap: 16px; border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; background: var(--bg-secondary); }
	.structure-form .ff { display: flex; flex-direction: column; gap: 4px; }
	.structure-form label { font-size: 12px; font-weight: 500; color: var(--text-secondary); }
	.structure-form input[type="text"] { padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 13px; }
	.pickers { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
	.picker-col { display: flex; flex-direction: column; gap: 6px; }
	.picker-label { font-size: 12px; font-weight: 500; color: var(--text-secondary); }
	.req { color: var(--error); }
	.opt { font-weight: 400; color: var(--text-muted); }
	.structure-form .btn-primary { align-self: flex-start; padding: 10px 24px; border: none; border-radius: var(--radius); background: var(--accent); color: white; font-size: 14px; font-weight: 600; cursor: pointer; }
	.structure-form .btn-primary:hover { background: var(--accent-hover); }
	.structure-form .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
