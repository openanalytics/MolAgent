<script lang="ts">
	import { datasets, refreshDatasets } from '$lib/stores/datasets.svelte';
	import { deleteDataset, uploadDatasetFile, type DatasetEntry } from '$lib/api/client';

	let selectedIds = $state<Set<string>>(new Set());
	let error = $state<string | null>(null);
	let uploading = $state(false);
	let dragover = $state(false);

	$effect(() => { refreshDatasets(); });

	function toggleSelect(id: string) {
		const next = new Set(selectedIds);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		selectedIds = next;
	}

	function selectAll() {
		if (selectedIds.size === datasets.entries.length) {
			selectedIds = new Set();
		} else {
			selectedIds = new Set(datasets.entries.map(e => e.id));
		}
	}

	async function handleDelete(id: string) {
		if (!confirm(`Delete this dataset? The file will be removed from disk.`)) return;
		error = null;
		try {
			await deleteDataset(id);
			selectedIds.delete(id);
			selectedIds = new Set(selectedIds);
			await refreshDatasets();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Delete failed';
		}
	}

	async function handleBulkDelete() {
		if (!confirm(`Delete ${selectedIds.size} dataset(s)? Files will be removed from disk.`)) return;
		error = null;
		try {
			for (const id of selectedIds) {
				await deleteDataset(id);
			}
			selectedIds = new Set();
			await refreshDatasets();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Bulk delete failed';
		}
	}

	async function handleFile(file: File) {
		uploading = true;
		error = null;
		try {
			await uploadDatasetFile(file);
			await refreshDatasets();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Upload failed';
		} finally {
			uploading = false;
		}
	}

	function onInput(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files?.[0]) handleFile(input.files[0]);
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragover = false;
		if (e.dataTransfer?.files?.[0]) handleFile(e.dataTransfer.files[0]);
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function formatDate(iso?: string): string {
		if (!iso) return '-';
		const d = new Date(iso);
		return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
	}
</script>

<div class="dp">
	<div class="ph">
		<h2>Datasets</h2>
		<button class="btn-refresh" onclick={() => refreshDatasets()}>Refresh</button>
	</div>

	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="upload-zone"
		class:dragover
		class:uploading
		ondragover={(e) => { e.preventDefault(); dragover = true; }}
		ondragleave={() => { dragover = false; }}
		ondrop={onDrop}
	>
		<label class="upload-label">
			<input type="file" accept=".csv" oninput={onInput} hidden />
			{#if uploading}
				<span class="upload-text">Uploading...</span>
			{:else}
				<span class="upload-icon">+</span>
				<span class="upload-text">Upload CSV</span>
				<span class="upload-hint">or drag and drop</span>
			{/if}
		</label>
	</div>

	{#if selectedIds.size > 0}
		<div class="action-bar">
			<span class="action-info">{selectedIds.size} selected</span>
			<button class="btn-delete-bulk" onclick={handleBulkDelete}>Delete Selected</button>
		</div>
	{/if}

	{#if error}<p class="err">{error}</p>{/if}

	{#if datasets.loading}
		<p class="loading">Loading datasets...</p>
	{:else if datasets.entries.length === 0}
		<p class="empty">No datasets uploaded yet. Upload a CSV to get started.</p>
	{:else}
		<div class="table-wrap">
			<table class="ds-table">
				<thead>
					<tr>
						<th class="col-cb"><input type="checkbox" checked={selectedIds.size === datasets.entries.length && datasets.entries.length > 0} onchange={selectAll} /></th>
						<th>Filename</th>
						<th>Type</th>
						<th>Rows</th>
						<th>Columns</th>
						<th>Size</th>
						<th>Uploaded</th>
						<th>Last Used</th>
						<th></th>
					</tr>
				</thead>
				<tbody>
					{#each datasets.entries as entry}
						<tr class:selected={selectedIds.has(entry.id)}>
							<td class="col-cb"><input type="checkbox" checked={selectedIds.has(entry.id)} onchange={() => toggleSelect(entry.id)} /></td>
							<td class="col-name" title={entry.filename}>{entry.filename}</td>
							<td><span class="type-badge" class:structure={entry.type === '3d_structure'}>{entry.type === '3d_structure' ? '3D Structure' : 'CSV'}</span></td>
							<td class="col-num">{entry.row_count.toLocaleString()}</td>
							<td class="col-cols" title={entry.columns.join(', ')}>{entry.columns.length}</td>
							<td class="col-size">{formatSize(entry.size_bytes)}</td>
							<td class="col-date">{formatDate(entry.uploaded_at)}</td>
							<td class="col-date">{formatDate(entry.last_used)}</td>
							<td><button class="btn-del" title="Delete dataset" onclick={() => handleDelete(entry.id)}>&times;</button></td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	.dp { max-width: 1000px; display: flex; flex-direction: column; gap: 16px; }
	.ph { display: flex; align-items: center; justify-content: space-between; }
	h2 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
	.btn-refresh { padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px; background: none; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
	.btn-refresh:hover { background: var(--bg-tertiary); }

	.upload-zone {
		border: 2px dashed var(--border);
		border-radius: var(--radius-lg);
		padding: 24px;
		text-align: center;
		transition: border-color 120ms, background 120ms;
		cursor: pointer;
	}
	.upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: var(--accent-dim); }
	.upload-zone.uploading { opacity: 0.6; pointer-events: none; }
	.upload-label { display: flex; flex-direction: column; align-items: center; gap: 4px; cursor: pointer; }
	.upload-icon { font-size: 24px; font-weight: 300; color: var(--text-muted); line-height: 1; }
	.upload-text { font-size: 13px; font-weight: 500; color: var(--text-primary); }
	.upload-hint { font-size: 11px; color: var(--text-muted); }

	.action-bar {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 14px;
		border-radius: var(--radius);
		background: rgba(239, 68, 68, 0.08);
		border: 1px solid var(--error);
	}
	.action-info { font-size: 13px; font-weight: 500; color: var(--error); }
	.btn-delete-bulk { padding: 6px 14px; border: none; border-radius: 6px; background: var(--error); color: white; font-size: 12px; font-weight: 600; cursor: pointer; }
	.btn-delete-bulk:hover { opacity: 0.9; }

	.err { font-size: 13px; color: var(--error); padding: 12px; border-radius: var(--radius); background: rgba(239, 68, 68, 0.1); }
	.loading, .empty { font-size: 13px; color: var(--text-muted); padding: 20px 0; }

	.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius); }
	.ds-table { width: 100%; border-collapse: collapse; font-size: 12px; }
	.ds-table th { text-align: left; padding: 10px 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-muted); background: var(--bg-secondary); border-bottom: 1px solid var(--border); }
	.ds-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text-secondary); vertical-align: middle; }
	.ds-table tr:last-child td { border-bottom: none; }
	.ds-table tr.selected { background: var(--accent-dim); }
	.ds-table tr:hover { background: var(--bg-tertiary); }

	.col-cb { width: 36px; text-align: center; }
	.col-cb input { accent-color: var(--accent); }
	.type-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; background: var(--bg-tertiary); color: var(--text-muted); }
	.type-badge.structure { background: rgba(168, 85, 247, 0.12); color: #a855f7; }
	.col-name { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: 11px; }
	.col-num { text-align: right; white-space: nowrap; }
	.col-cols { text-align: right; white-space: nowrap; }
	.col-size { white-space: nowrap; }
	.col-date { white-space: nowrap; }

	.btn-del { border: none; background: none; color: var(--text-muted); font-size: 18px; cursor: pointer; padding: 2px 6px; border-radius: 4px; }
	.btn-del:hover { background: rgba(239, 68, 68, 0.1); color: var(--error); }
</style>
