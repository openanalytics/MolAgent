<script lang="ts">
	import { page } from '$app/state';

	const tabs = [
		{ href: '/train', label: 'Train' },
		{ href: '/predict', label: 'Predict' },
		{ href: '/datasets', label: 'Datasets' },
		{ href: '/models', label: 'Models' },
		{ href: '/visualize', label: 'Visualize' },
		{ href: '/settings', label: 'Settings' },
	];
</script>

<nav class="tab-bar">
	<span class="tab-logo">AutoMol</span>
	<div class="tab-links">
		{#each tabs as tab}
			<a
				href={tab.href}
				class="tab-link"
				class:active={page.url.pathname.startsWith(tab.href)}
			>
				{tab.label}
			</a>
		{/each}
	</div>
</nav>

<style>
	.tab-bar {
		display: flex;
		align-items: center;
		gap: 24px;
		padding: 0 20px;
		height: 52px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-secondary);
		flex-shrink: 0;
	}
	.tab-logo {
		font-weight: 700;
		font-size: 16px;
		color: var(--accent);
		letter-spacing: -0.02em;
	}
	.tab-links {
		display: flex;
		gap: 4px;
	}
	.tab-link {
		padding: 6px 14px;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 500;
		color: var(--text-secondary);
		text-decoration: none;
		transition: background 120ms, color 120ms;
	}
	.tab-link:hover {
		background: var(--bg-tertiary);
		color: var(--text-primary);
	}
	.tab-link.active {
		background: var(--accent-dim);
		color: var(--accent);
	}
</style>
