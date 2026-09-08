<script lang="ts">
	let { csvText }: { csvText: string } = $props();
	const rows = $derived.by(() => {
		const lines = csvText.trim().split('\n');
		if (lines.length < 2) return { headers: [] as string[], data: [] as string[][], total: 0 };
		return { headers: lines[0].split(','), data: lines.slice(1, 11).map(l => l.split(',')), total: lines.length - 1 };
	});
</script>
{#if rows.headers.length > 0}
<div class="rp"><h4>Predictions ({rows.total} rows)</h4><div class="tw"><table><thead><tr>{#each rows.headers as h}<th>{h}</th>{/each}</tr></thead><tbody>{#each rows.data as row}<tr>{#each row as cell}<td>{cell}</td>{/each}</tr>{/each}</tbody></table></div></div>
{/if}
<style>
	.rp { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); overflow: hidden; }
	h4 { font-size: 13px; font-weight: 600; color: var(--text-secondary); padding: 12px 16px; border-bottom: 1px solid var(--border); }
	.tw { overflow-x: auto; }
	table { width: 100%; font-size: 12px; border-collapse: collapse; }
	th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
	th { font-weight: 600; color: var(--text-muted); text-transform: uppercase; font-size: 10px; background: var(--bg-primary); }
	td { color: var(--text-secondary); font-family: var(--font-mono); }
</style>