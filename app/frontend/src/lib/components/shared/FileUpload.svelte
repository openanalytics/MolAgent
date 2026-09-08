<script lang="ts">
	import { uploadFile, type UploadResult } from '$lib/api/client';

	let {
		accept = '.csv',
		label = 'Upload CSV',
		onUploaded,
	}: {
		accept?: string;
		label?: string;
		onUploaded: (result: UploadResult) => void;
	} = $props();

	let uploading = $state(false);
	let dragover = $state(false);
	let error = $state<string | null>(null);

	async function handleFile(file: File) {
		uploading = true;
		error = null;
		try {
			const result = await uploadFile(file);
			onUploaded(result);
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
</script>

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
		<input type="file" {accept} oninput={onInput} hidden />
		{#if uploading}
			<span class="upload-text">Uploading...</span>
		{:else}
			<span class="upload-icon">+</span>
			<span class="upload-text">{label}</span>
			<span class="upload-hint">or drag and drop</span>
		{/if}
	</label>
	{#if error}
		<p class="upload-error">{error}</p>
	{/if}
</div>

<style>
	.upload-zone {
		border: 2px dashed var(--border);
		border-radius: var(--radius-lg);
		padding: 32px;
		text-align: center;
		transition: border-color 120ms, background 120ms;
		cursor: pointer;
	}
	.upload-zone:hover, .upload-zone.dragover {
		border-color: var(--accent);
		background: var(--accent-dim);
	}
	.upload-zone.uploading {
		opacity: 0.6;
		pointer-events: none;
	}
	.upload-label {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
		cursor: pointer;
	}
	.upload-icon {
		font-size: 28px;
		font-weight: 300;
		color: var(--text-muted);
		line-height: 1;
	}
	.upload-text {
		font-size: 14px;
		font-weight: 500;
		color: var(--text-primary);
	}
	.upload-hint {
		font-size: 12px;
		color: var(--text-muted);
	}
	.upload-error {
		margin-top: 8px;
		font-size: 12px;
		color: var(--error);
	}
</style>
