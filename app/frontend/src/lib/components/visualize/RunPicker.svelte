<script lang="ts">
	import { listRuns, getSettings, type PipelineState } from '$lib/api/client';
	let { onSelect }: { onSelect: (run: PipelineState) => void } = $props();
	let runs = $state<PipelineState[]>([]);
	let selectedIdx = $state(-1);
	let loading = $state(true);
	let mcpMode = $state<string>('local');
	$effect(() => {
		Promise.all([listRuns(), getSettings()]).then(([r, s]) => {
			runs = r.filter(run => run.steps_completed?.includes(5));
			mcpMode = s.mode;
			loading = false;
		}).catch(() => { loading = false; });
	});
	let selectedRun = $derived(selectedIdx >= 0 ? runs[selectedIdx] : null);
	let mismatch = $derived(selectedRun != null && selectedRun.source !== mcpMode);
</script>
<div class="rp">
	<h4>Select Run</h4>
	{#if loading}<p class="muted">Loading runs...</p>
	{:else if runs.length === 0}<p class="muted">No completed evaluation runs found.</p>
	{:else}<select bind:value={selectedIdx} onchange={() => { if (selectedIdx >= 0) onSelect(runs[selectedIdx]); }}>
		<option value={-1}>-- Choose a run --</option>
		{#each runs as run, idx}<option value={idx} class:other-source={run.source !== mcpMode}>{run.source === 'remote' ? '☁ ' : ''}{run.run_id} - {(run.config as any)?.target_properties?.join(', ')} ({(run.config as any)?.task_type})</option>{/each}
	</select>{/if}
	{#if mismatch}
		<p class="warning">This run was trained in {selectedRun?.source} mode. Predict requires switching MCP to {selectedRun?.source} in Settings.</p>
	{/if}
</div>
<style>
	.rp { display: flex; flex-direction: column; gap: 8px; }
	h4 { font-size: 13px; font-weight: 600; color: var(--text-secondary); }
	select { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 13px; width: 100%; }
	select:focus { outline: none; border-color: var(--accent); }
	.muted { font-size: 13px; color: var(--text-muted); }
	option.other-source { color: var(--warning); }
	.warning { font-size: 12px; color: var(--warning); padding: 8px 12px; border-radius: 6px; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); }
</style>
