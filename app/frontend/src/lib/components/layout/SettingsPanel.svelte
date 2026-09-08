<script lang="ts">
	import { getSettings, updateSettings, type MCPSettingsResponse } from '$lib/api/client';

	let settings = $state<MCPSettingsResponse | null>(null);
	let mode = $state<'local' | 'remote'>('local');
	let url = $state('');
	let serverPath = $state('');
	let authToken = $state('');
	let outputFolder = $state('');
	let saving = $state(false);
	let message = $state('');

	async function load() {
		try {
			settings = await getSettings();
			mode = settings.mode as 'local' | 'remote';
			url = settings.url ?? '';
			serverPath = settings.server_path ?? '';
			outputFolder = settings.output_folder ?? '';
		} catch {}
	}

	async function save() {
		saving = true;
		message = '';
		try {
			settings = await updateSettings({
				mode,
				url: mode === 'remote' ? url : undefined,
				server_path: mode === 'local' ? serverPath : undefined,
				auth_token: authToken || undefined,
				output_folder: outputFolder || undefined,
			});
			message = 'Saved';
			setTimeout(() => { message = ''; }, 2000);
		} catch (e) {
			message = e instanceof Error ? e.message : 'Save failed';
		} finally {
			saving = false;
		}
	}

	$effect(() => { load(); });
</script>

<div class="sp">
	<h3>MCP Connection</h3>
	<div class="mode-toggle">
		<label class="mode-opt" class:active={mode === 'local'}>
			<input type="radio" bind:group={mode} value="local" /> Local Server
		</label>
		<label class="mode-opt" class:active={mode === 'remote'}>
			<input type="radio" bind:group={mode} value="remote" /> Remote URL
		</label>
	</div>

	{#if mode === 'local'}
		<div class="ff">
			<label>Server script path (leave empty for default)</label>
			<input type="text" bind:value={serverPath} placeholder="mcp/server.py" />
		</div>
	{:else}
		<div class="ff">
			<label>MCP Server URL</label>
			<input type="text" bind:value={url} placeholder="https://mcp.example.com/mcp" />
		</div>
		<div class="ff">
			<label>Auth Token (optional)</label>
			<input type="password" bind:value={authToken} placeholder="Bearer token" />
		</div>
	{/if}

	<h3>Output</h3>
	<div class="ff">
		<label>Output folder (local path for downloaded models and dashboards)</label>
		<input type="text" bind:value={outputFolder} placeholder="/mnt/c/Users/.../output" />
		{#if settings?.warnings?.length}
			{#each settings.warnings as w}
				<p class="warn">{w}</p>
			{/each}
		{/if}
	</div>

	<div class="actions">
		{#if message}<span class="msg">{message}</span>{/if}
		<button class="btn-save" onclick={save} disabled={saving}>{saving ? 'Saving...' : 'Save'}</button>
	</div>
</div>

<style>
	.sp { display: flex; flex-direction: column; gap: 12px; padding: 16px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); }
	h3 { font-size: 14px; font-weight: 600; color: var(--text-primary); margin: 0; }
	.mode-toggle { display: flex; gap: 12px; }
	.mode-opt { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-secondary); cursor: pointer; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border); }
	.mode-opt.active { border-color: var(--accent); background: var(--accent-dim); color: var(--text-primary); }
	.mode-opt input { display: none; }
	.ff { display: flex; flex-direction: column; gap: 4px; }
	.ff label { font-size: 11px; font-weight: 500; color: var(--text-secondary); }
	.ff input { padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 13px; }
	.ff input:focus { outline: none; border-color: var(--accent); }
	.actions { display: flex; align-items: center; gap: 12px; justify-content: flex-end; }
	.msg { font-size: 12px; color: var(--accent); }
	.btn-save { padding: 6px 16px; border: none; border-radius: 6px; background: var(--accent); color: white; font-size: 13px; font-weight: 600; cursor: pointer; }
	.btn-save:hover { background: var(--accent-hover); }
	.btn-save:disabled { opacity: 0.5; cursor: not-allowed; }
	.warn {
		margin: 6px 0 0;
		padding: 8px 10px;
		font-size: 12px;
		line-height: 1.45;
		color: var(--text-primary);
		background: color-mix(in srgb, var(--warning, #d4a056) 12%, transparent);
		border-left: 3px solid var(--warning, #d4a056);
		border-radius: 4px;
	}
</style>
