<script lang="ts">
	let {
		label,
		hint = '',
		accept = '*',
		multiple = false,
		files = $bindable<File[]>([]),
	}: {
		label: string;
		hint?: string;
		accept?: string;
		multiple?: boolean;
		files?: File[];
	} = $props();

	let dragover = $state(false);

	function applyFiles(list: FileList | null) {
		if (!list) return;
		files = multiple ? Array.from(list) : [list[0]];
	}

	function onInput(e: Event) {
		applyFiles((e.target as HTMLInputElement).files);
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragover = false;
		applyFiles(e.dataTransfer?.files ?? null);
	}

	const summary = $derived(
		files.length === 0 ? null
		: files.length === 1 ? files[0].name
		: `${files.length} files selected`
	);
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	class="box"
	class:dragover
	class:has-file={files.length > 0}
	ondragover={(e) => { e.preventDefault(); dragover = true; }}
	ondragleave={() => { dragover = false; }}
	ondrop={onDrop}
>
	<label class="inner">
		<input type="file" {accept} {multiple} oninput={onInput} hidden />
		{#if summary}
			<span class="icon done">✓</span>
			<span class="text selected">{summary}</span>
			<span class="hint">click to replace</span>
		{:else}
			<span class="icon">+</span>
			<span class="text">{label}</span>
			{#if hint}<span class="hint">{hint}</span>{/if}
		{/if}
	</label>
</div>

<style>
	.box {
		border: 2px dashed var(--border);
		border-radius: var(--radius-lg);
		padding: 20px 16px;
		text-align: center;
		transition: border-color 120ms, background 120ms;
		cursor: pointer;
	}
	.box:hover, .box.dragover {
		border-color: var(--accent);
		background: var(--accent-dim);
	}
	.box.has-file {
		border-style: solid;
		border-color: var(--accent);
		background: var(--accent-dim);
	}
	.inner {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		cursor: pointer;
	}
	.icon {
		font-size: 24px;
		font-weight: 300;
		color: var(--text-muted);
		line-height: 1;
	}
	.icon.done {
		font-size: 20px;
		color: var(--accent);
		font-weight: 600;
	}
	.text {
		font-size: 13px;
		font-weight: 500;
		color: var(--text-primary);
	}
	.text.selected {
		font-family: var(--font-mono);
		font-size: 12px;
	}
	.hint {
		font-size: 11px;
		color: var(--text-muted);
	}
</style>
