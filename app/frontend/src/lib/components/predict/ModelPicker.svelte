<script lang="ts">
	import { registry, refreshRegistry } from '$lib/stores/registry.svelte';
	import type { ModelEntry } from '$lib/api/client';
	let { onSelect }: { onSelect: (entry: ModelEntry) => void } = $props();
	let selectedIdx = $state(-1);
	$effect(() => { refreshRegistry(); });
</script>
<div class="mp">
	<h4>Select Model</h4>
	{#if registry.loading}<p class="muted">Loading registry...</p>
	{:else if registry.entries.length === 0}<p class="muted">No models found. Train a model first.</p>
	{:else}<select bind:value={selectedIdx} onchange={() => { if (selectedIdx >= 0) onSelect(registry.entries[selectedIdx]); }}>
		<option value={-1}>-- Choose a model --</option>
		{#each registry.entries as entry, idx}<option value={idx}>{entry.id} - {entry.target_properties?.join(', ')} ({entry.task_type})</option>{/each}
	</select>{/if}
</div>
<style>
	.mp { display: flex; flex-direction: column; gap: 8px; }
	h4 { font-size: 13px; font-weight: 600; color: var(--text-secondary); }
	select { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 13px; width: 100%; }
	select:focus { outline: none; border-color: var(--accent); }
	.muted { font-size: 13px; color: var(--text-muted); }
</style>