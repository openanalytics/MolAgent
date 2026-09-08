<script lang="ts">
	import '../app.css';
	import TabBar from '$lib/components/layout/TabBar.svelte';
	let { children } = $props();
</script>
<div class="shell"><TabBar /><main class="main">{@render children()}</main></div>
<style>
	.shell { display: flex; flex-direction: column; height: 100vh; width: 100vw; overflow: hidden; }
	.main { flex: 1; overflow-y: auto; padding: 24px; }
</style>