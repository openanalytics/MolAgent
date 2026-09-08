<script lang="ts">
	import { registry, refreshRegistry } from '$lib/stores/registry.svelte';
	import { deleteModel, mergeModels, downloadModel, type ModelEntry, type JobStatus } from '$lib/api/client';
	import JobSpinner from '$lib/components/shared/JobSpinner.svelte';

	let selectedIds = $state<Set<string>>(new Set());
	let mergeName = $state('');
	let mergeJobId = $state<string | null>(null);
	let error = $state<string | null>(null);
	let downloadingId = $state<string | null>(null);
	let downloadMsg = $state<string | null>(null);

	$effect(() => { refreshRegistry(); });

	async function handleDownload(id: string) {
		downloadingId = id;
		downloadMsg = null;
		error = null;
		try {
			const result = await downloadModel(id);
			downloadMsg = result.saved_path ? `Saved to ${result.saved_path}` : 'Downloaded';
			setTimeout(() => { downloadMsg = null; }, 4000);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Download failed';
		} finally {
			downloadingId = null;
		}
	}

	function toggleSelect(id: string) {
		const next = new Set(selectedIds);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		selectedIds = next;
	}

	function selectAll() {
		if (selectedIds.size === registry.entries.length) {
			selectedIds = new Set();
		} else {
			selectedIds = new Set(registry.entries.map(e => e.id));
		}
	}

	async function handleDelete(id: string) {
		if (!confirm(`Delete model "${id}"? This removes the model files from disk.`)) return;
		error = null;
		try {
			await deleteModel(id);
			selectedIds.delete(id);
			selectedIds = new Set(selectedIds);
			await refreshRegistry();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Delete failed';
		}
	}

	async function handleMerge() {
		if (selectedIds.size < 2) return;
		error = null;
		try {
			const { job_id } = await mergeModels({
				model_ids: [...selectedIds],
				output_name: mergeName || undefined,
			});
			mergeJobId = job_id;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Merge failed';
		}
	}

	function onMergeDone(job: JobStatus) {
		mergeJobId = null;
		if (job.status === 'success') {
			selectedIds = new Set();
			mergeName = '';
			refreshRegistry();
		} else {
			error = 'Merge failed';
		}
	}

	function formatMetrics(metrics: Record<string, Record<string, number>>): string {
		const parts: string[] = [];
		for (const [prop, m] of Object.entries(metrics)) {
			const key = m.r2 != null ? 'R2' : m.balanced_accuracy != null ? 'BA' : Object.keys(m)[0];
			const val = m.r2 ?? m.balanced_accuracy ?? Object.values(m)[0];
			if (val != null) parts.push(`${prop}: ${key}=${val.toFixed(3)}`);
		}
		return parts.join(', ') || '-';
	}

	function formatDate(iso?: string): string {
		if (!iso) return '-';
		const d = new Date(iso);
		return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
	}
</script>

<div class="mp">
	<div class="ph">
		<h2>Models</h2>
		<button class="btn-refresh" onclick={() => refreshRegistry()}>Refresh</button>
	</div>

	{#if selectedIds.size >= 2}
		<div class="merge-bar">
			<span class="merge-info">{selectedIds.size} models selected</span>
			<input type="text" class="merge-name" placeholder="Merged model name (optional)" bind:value={mergeName} />
			<button class="btn-merge" onclick={handleMerge} disabled={mergeJobId != null}>Merge Selected</button>
		</div>
	{/if}

	{#if mergeJobId}
		<JobSpinner jobId={mergeJobId} label="Merging models..." onDone={onMergeDone} />
	{/if}

	{#if downloadMsg}<p class="dl-msg">{downloadMsg}</p>{/if}
	{#if error}<p class="err">{error}</p>{/if}

	{#if registry.loading}
		<p class="loading">Loading models...</p>
	{:else if registry.entries.length === 0}
		<p class="empty">No models in the registry. Train a model to get started.</p>
	{:else}
		<div class="table-wrap">
			<table class="model-table">
				<thead>
					<tr>
						<th class="col-cb"><input type="checkbox" checked={selectedIds.size === registry.entries.length && registry.entries.length > 0} onchange={selectAll} /></th>
						<th>ID</th>
						<th>Properties</th>
						<th>Task</th>
						<th>Metrics</th>
						<th>Features</th>
						<th>Created</th>
						<th></th>
					</tr>
				</thead>
				<tbody>
					{#each registry.entries as entry}
						<tr class:selected={selectedIds.has(entry.id)}>
							<td class="col-cb"><input type="checkbox" checked={selectedIds.has(entry.id)} onchange={() => toggleSelect(entry.id)} /></td>
							<td class="col-id" title={entry.id}>
								<span class="mono">{entry.id.length > 28 ? entry.id.slice(0, 28) + '...' : entry.id}</span>
								{#if entry.is_refitted}<span class="badge badge-refit">refit</span>{/if}
								{#if entry.model_format === 'merged'}<span class="badge badge-merged">merged</span>{/if}
							</td>
							<td>{entry.target_properties.join(', ')}</td>
							<td><span class="tag-task">{entry.task_type}</span></td>
							<td class="col-metrics" title={formatMetrics(entry.metrics)}>{formatMetrics(entry.metrics)}</td>
							<td class="col-feat">{entry.feature_keys.join(', ')}</td>
							<td class="col-date">{formatDate(entry.created_at)}</td>
							<td class="col-actions">
								<button class="btn-dl" title="Download model" onclick={() => handleDownload(entry.id)} disabled={downloadingId === entry.id}>{downloadingId === entry.id ? '...' : '↓'}</button>
								<button class="btn-del" title="Delete model" onclick={() => handleDelete(entry.id)}>&times;</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	.mp { max-width: 1100px; display: flex; flex-direction: column; gap: 16px; }
	.ph { display: flex; align-items: center; justify-content: space-between; }
	h2 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
	.btn-refresh { padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px; background: none; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
	.btn-refresh:hover { background: var(--bg-tertiary); }

	.merge-bar {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 14px;
		border-radius: var(--radius);
		background: var(--accent-dim);
		border: 1px solid var(--accent);
	}
	.merge-info { font-size: 13px; font-weight: 500; color: var(--accent); }
	.merge-name { flex: 1; max-width: 240px; padding: 5px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 12px; }
	.btn-merge { padding: 6px 16px; border: none; border-radius: 6px; background: var(--accent); color: white; font-size: 12px; font-weight: 600; cursor: pointer; }
	.btn-merge:hover { background: var(--accent-hover); }
	.btn-merge:disabled { opacity: 0.5; cursor: not-allowed; }

	.err { font-size: 13px; color: var(--error); padding: 12px; border-radius: var(--radius); background: rgba(239, 68, 68, 0.1); }
	.loading, .empty { font-size: 13px; color: var(--text-muted); padding: 20px 0; }

	.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius); }
	.model-table { width: 100%; border-collapse: collapse; font-size: 12px; }
	.model-table th { text-align: left; padding: 10px 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-muted); background: var(--bg-secondary); border-bottom: 1px solid var(--border); }
	.model-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text-secondary); vertical-align: middle; }
	.model-table tr:last-child td { border-bottom: none; }
	.model-table tr.selected { background: var(--accent-dim); }
	.model-table tr:hover { background: var(--bg-tertiary); }

	.col-cb { width: 36px; text-align: center; }
	.col-cb input { accent-color: var(--accent); }
	.col-id { max-width: 220px; }
	.col-metrics { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.col-feat { max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.col-date { white-space: nowrap; }

	.tag-task { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: var(--bg-tertiary); color: var(--text-muted); text-transform: lowercase; }
	.badge { font-size: 9px; padding: 1px 5px; border-radius: 3px; margin-left: 4px; font-weight: 600; text-transform: uppercase; }
	.badge-refit { background: rgba(16, 185, 129, 0.12); color: #10b981; }
	.badge-merged { background: rgba(99, 102, 241, 0.12); color: #6366f1; }

	.col-actions { display: flex; gap: 4px; align-items: center; }
	.btn-dl { border: none; background: none; color: var(--text-muted); font-size: 16px; cursor: pointer; padding: 2px 6px; border-radius: 4px; }
	.btn-dl:hover { background: rgba(59, 130, 246, 0.1); color: var(--accent); }
	.btn-dl:disabled { opacity: 0.4; cursor: not-allowed; }
	.btn-del { border: none; background: none; color: var(--text-muted); font-size: 18px; cursor: pointer; padding: 2px 6px; border-radius: 4px; }
	.btn-del:hover { background: rgba(239, 68, 68, 0.1); color: var(--error); }
	.dl-msg { font-size: 12px; color: var(--success); padding: 8px 12px; border-radius: var(--radius); background: rgba(34, 197, 94, 0.08); }
</style>
