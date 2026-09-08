<script lang="ts">
	import { getJobStatus, getJobLogs, getDashboardFromJob, downloadModel, type JobStatus as JobStatusType } from '$lib/api/client';
	import { pipeline } from '$lib/stores/pipeline.svelte';
	import type { TrainingResult } from '$lib/api/client';

	let dlStatus = $state<'idle' | 'downloading' | 'done' | 'error'>('idle');
	let dlPath = $state<string | null>(null);

	async function handleDownloadModel() {
		const modelId = pipeline.result?.model_id;
		if (!modelId) return;
		dlStatus = 'downloading';
		try {
			const result = await downloadModel(modelId);
			dlPath = result.saved_path;
			dlStatus = 'done';
		} catch {
			dlStatus = 'error';
		}
	}

	const STEPS = [
		'Preparing data',
		'Splitting data',
		'Training model',
		'Merging models',
		'Evaluating model',
		'Refitting model',
		'Merging refitted models',
		'Generating dashboard',
	];

	let status = $state<'pending' | 'running' | 'success' | 'failed'>('running');
	let progress = $state(0);
	let progressTotal = $state(8);
	let progressLabel = $state('');
	let logs = $state<string[]>([]);
	let error = $state<string | null>(null);
	let dashboardHtml = $state<string | null>(null);
	$effect(() => {
		if (!pipeline.jobId) return;
		status = 'running';
		let errorCount = 0;
		const id = setInterval(async () => {
			try {
				const jobStatus = await getJobStatus(pipeline.jobId!);
				status = jobStatus.status;
				progress = jobStatus.progress;
				progressTotal = jobStatus.progress_total || 8;
				progressLabel = jobStatus.progress_label;
				errorCount = 0;

				if (jobStatus.status === 'success' || jobStatus.status === 'failed') {
					clearInterval(id);
					if (jobStatus.status === 'success') {
						pipeline.result = jobStatus.result as TrainingResult;
						pipeline.step = 'done';
						try {
							dashboardHtml = await getDashboardFromJob(pipeline.jobId!);
						} catch {}
					} else {
						const jobLogs = await getJobLogs(pipeline.jobId!);
						logs = jobLogs.lines;
						error = 'Pipeline failed — check logs below';
					}
				}
			} catch {
				errorCount++;
				if (errorCount >= 20) {
					clearInterval(id);
					status = 'failed';
					logs = [...logs, 'Polling stopped after repeated errors — is the backend running?'];
				}
			}
		}, 2000);
		return () => clearInterval(id);
	});

	const metrics = $derived(pipeline.result?.metrics ?? {});
</script>

<div class="ps">
	<h3>Pipeline Execution</h3>

	<div class="steps-list">
		{#each STEPS as label, idx}
			{@const stepNum = idx + 1}
			{@const isDone = stepNum < progress || status === 'success'}
			{@const isActive = stepNum === progress && status === 'running'}
			<div class="step-row" class:done={isDone} class:active={isActive} class:future={!isDone && !isActive}>
				<div class="step-ind">
					{#if isDone}<span class="chk">&#10003;</span>
					{:else if isActive}<span class="dot"></span>
					{:else}<span class="num">{stepNum}</span>{/if}
				</div>
				<span>{label}</span>
			</div>
		{/each}
	</div>

	{#if status === 'running'}
		<div class="progress-bar">
			<div class="progress-fill" style="width: {(progress / progressTotal) * 100}%"></div>
		</div>
		{#if progressLabel}<p class="progress-label">{progressLabel}</p>{/if}
	{/if}

	{#if error}
		<div class="step-err"><p>{error}</p></div>
	{/if}

	{#if logs.length > 0}
		<details class="logs-section">
			<summary>Logs ({logs.length} lines)</summary>
			<pre class="logs">{logs.slice(-30).join('\n')}</pre>
		</details>
	{/if}

	{#if Object.keys(metrics).length > 0}
		<div class="metrics"><h4>Model Metrics</h4>
			{#each Object.entries(metrics) as [prop, vals]}
				<div class="mrow"><span class="mono mprop">{prop}</span>{#each Object.entries(vals) as [key, val]}<span class="mval">{key}: {typeof val === 'number' ? val.toFixed(4) : val}</span>{/each}</div>
			{/each}
		</div>
	{/if}

	{#if pipeline.result?.model_id}
		<div class="download-section">
			{#if dlStatus === 'idle'}
				<button class="btn-download" onclick={handleDownloadModel}>Download Model ({pipeline.result.model_filename})</button>
			{:else if dlStatus === 'downloading'}
				<button class="btn-download" disabled>Downloading...</button>
			{:else if dlStatus === 'done'}
				<span class="dl-done">{dlPath ? `Saved to ${dlPath}` : 'Downloaded'}</span>
			{:else}
				<span class="dl-err">Download failed</span>
				<button class="btn-download" onclick={handleDownloadModel}>Retry</button>
			{/if}
		</div>
	{/if}

	{#if dashboardHtml}
		<div class="dashboard-frame">
			<h4>Dashboard</h4>
			<iframe srcdoc={dashboardHtml} sandbox="allow-scripts" title="Dashboard"></iframe>
		</div>
	{/if}

	{#if pipeline.step === 'done' && !dashboardHtml && !error}
		<div class="done-msg">Pipeline complete!</div>
	{/if}
</div>

<style>
	.ps { display: flex; flex-direction: column; gap: 16px; }
	h3 { font-size: 15px; font-weight: 600; color: var(--text-primary); }
	h4 { font-size: 13px; font-weight: 600; color: var(--text-secondary); margin-bottom: 8px; }
	.steps-list { display: flex; flex-wrap: wrap; gap: 4px; }
	.step-row { display: flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: var(--radius); background: var(--bg-secondary); border: 1px solid var(--border); font-size: 13px; color: var(--text-primary); }
	.step-row.active { border-color: var(--accent); background: var(--accent-dim); }
	.step-row.done { border-color: var(--success); }
	.step-row.future { opacity: 0.5; }
	.step-ind { width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; background: var(--bg-tertiary); color: var(--text-muted); }
	.step-row.done .step-ind { background: var(--success); color: white; }
	.step-row.active .step-ind { background: var(--accent); color: white; }
	.chk { font-size: 12px; }
	.dot { width: 8px; height: 8px; border-radius: 50%; background: white; animation: pulse 1.2s ease-in-out infinite; }
	.num { font-size: 10px; }
	.progress-bar { height: 6px; border-radius: 3px; background: var(--bg-tertiary); overflow: hidden; }
	.progress-fill { height: 100%; background: var(--accent); transition: width 300ms ease; border-radius: 3px; }
	.progress-label { font-size: 12px; color: var(--text-muted); }
	.step-err { padding: 12px 16px; border-radius: var(--radius); background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }
	.step-err p { font-size: 13px; color: var(--error); }
	.logs-section { margin-top: 8px; }
	.logs-section summary { font-size: 12px; color: var(--text-muted); cursor: pointer; }
	.logs { font-size: 11px; max-height: 200px; overflow-y: auto; padding: 10px; border-radius: var(--radius); background: var(--bg-tertiary); color: var(--text-secondary); white-space: pre-wrap; word-break: break-all; }
	.metrics { padding: 16px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); }
	.mrow { display: flex; gap: 16px; align-items: center; padding: 4px 0; border-bottom: 1px solid var(--border); }
	.mrow:last-child { border-bottom: none; }
	.mprop { font-weight: 600; min-width: 100px; color: var(--text-primary); font-size: 13px; }
	.mval { font-size: 12px; color: var(--text-secondary); }
	.download-section { display: flex; align-items: center; gap: 10px; }
	.btn-download { padding: 8px 18px; border: none; border-radius: var(--radius); background: var(--accent); color: white; font-size: 13px; font-weight: 600; cursor: pointer; }
	.btn-download:hover { background: var(--accent-hover); }
	.btn-download:disabled { opacity: 0.5; cursor: not-allowed; }
	.dl-done { font-size: 12px; color: var(--success); }
	.dl-err { font-size: 12px; color: var(--error); }
	.dashboard-frame { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
	.dashboard-frame iframe { width: 100%; height: 600px; border: none; }
	.done-msg { padding: 16px; border-radius: var(--radius); background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); color: var(--success); font-weight: 600; font-size: 14px; text-align: center; }
	@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
