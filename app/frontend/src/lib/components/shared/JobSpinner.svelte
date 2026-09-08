<script lang="ts">
	import { getJobStatus, getJobLogs, type JobStatus } from '$lib/api/client';

	let {
		jobId,
		label = 'Running...',
		onDone,
	}: {
		jobId: string;
		label?: string;
		onDone: (job: JobStatus) => void;
	} = $props();

	let logs = $state<string[]>([]);
	let status = $state<string>('pending');

	$effect(() => {
		if (!jobId) return;
		status = 'running';
		let errorCount = 0;
		const id = setInterval(async () => {
			try {
				const [jobStatus, jobLogs] = await Promise.all([
					getJobStatus(jobId),
					getJobLogs(jobId),
				]);
				status = jobStatus.status;
				logs = jobLogs.lines;
				errorCount = 0;
				if (jobStatus.status === 'success' || jobStatus.status === 'failed') {
					clearInterval(id);
					onDone(jobStatus);
				}
			} catch {
				errorCount++;
				if (errorCount >= 20) {
					clearInterval(id);
					status = 'failed';
					logs = [...logs, 'Polling stopped after repeated errors'];
				}
			}
		}, 1500);
		return () => clearInterval(id);
	});
</script>

<div class="spinner-container">
	<div class="spinner-header">
		{#if status === 'running' || status === 'pending'}
			<div class="spinner-dot"></div>
		{/if}
		<span class="spinner-label">{label}</span>
		<span class="spinner-status" class:running={status === 'running'} class:failed={status === 'failed'}>
			{status}
		</span>
	</div>
	{#if logs.length > 0}
		<pre class="spinner-logs">{logs.slice(-20).join('\n')}</pre>
	{/if}
</div>

<style>
	.spinner-container {
		border: 1px solid var(--border);
		border-radius: var(--radius);
		background: var(--bg-secondary);
		overflow: hidden;
	}
	.spinner-header {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 12px 16px;
		border-bottom: 1px solid var(--border);
	}
	.spinner-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--accent);
		animation: pulse 1.2s ease-in-out infinite;
	}
	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.3; }
	}
	.spinner-label {
		font-size: 13px;
		font-weight: 500;
		color: var(--text-primary);
		flex: 1;
	}
	.spinner-status {
		font-size: 11px;
		font-family: var(--font-mono);
		color: var(--text-muted);
		text-transform: uppercase;
	}
	.spinner-status.running { color: var(--accent); }
	.spinner-status.failed { color: var(--error); }
	.spinner-logs {
		padding: 12px 16px;
		font-size: 11px;
		font-family: var(--font-mono);
		color: var(--text-secondary);
		background: var(--bg-primary);
		max-height: 200px;
		overflow-y: auto;
		white-space: pre-wrap;
		word-break: break-all;
		margin: 0;
	}
</style>
