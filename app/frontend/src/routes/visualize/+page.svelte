<script lang="ts">
	import RunPicker from '$lib/components/visualize/RunPicker.svelte';
	import DashboardFrame from '$lib/components/visualize/DashboardFrame.svelte';
	import { getDashboardHtml, type PipelineState } from '$lib/api/client';
	let selectedRun = $state<PipelineState | null>(null);
	let dashboardHtml = $state<string | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(false);
	async function onRunSelect(run: PipelineState) {
		selectedRun = run;
		dashboardHtml = null;
		error = null;
		loading = true;
		try { dashboardHtml = await getDashboardHtml(run.run_id); }
		catch { error = 'Dashboard not available for this run. Train with the pipeline to generate one.'; }
		finally { loading = false; }
	}
</script>
<div class="vp">
	<h2>Visualize</h2>
	<RunPicker onSelect={onRunSelect} />
	{#if loading}<p class="loading">Loading dashboard...</p>{/if}
	{#if dashboardHtml}<DashboardFrame html={dashboardHtml} />{/if}
	{#if error}<p class="info">{error}</p>{/if}
</div>
<style>
	.vp { display: flex; flex-direction: column; gap: 20px; }
	h2 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
	.loading { font-size: 13px; color: var(--text-muted); }
	.info { font-size: 13px; color: var(--text-secondary); padding: 12px; border-radius: var(--radius); background: var(--bg-secondary); border: 1px solid var(--border); }
</style>
