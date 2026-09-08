<script lang="ts">
	import SettingsPanel from '$lib/components/layout/SettingsPanel.svelte';
</script>
<div class="sp-page">
	<h2>Settings</h2>
	<SettingsPanel />
</div>
<style>
	.sp-page { max-width: 600px; display: flex; flex-direction: column; gap: 20px; }
	h2 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
</style>
