<script lang="ts">
	let { html }: { html: string } = $props();
</script>
<div class="df"><iframe srcdoc={html} title="AutoMol Dashboard" sandbox="allow-scripts"></iframe></div>
<style>
	.df { width: 100%; height: 85vh; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
	iframe { width: 100%; height: 100%; border: none; }
</style>