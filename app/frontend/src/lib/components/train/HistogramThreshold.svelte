<script lang="ts">
	import type { TargetHistogram } from '$lib/api/client';

	interface Props {
		histogram: TargetHistogram;
		min: number;
		max: number;
		thresholds: number[];
		onchange: (thresholds: number[]) => void;
	}

	let { histogram, min, max, thresholds, onchange }: Props = $props();

	const W = 460;
	const H = 120;
	const PAD = 2;
	const MAX_THRESHOLDS = 5;

	const maxCount = $derived(Math.max(...histogram.counts, 1));
	const range = $derived(max - min || 1);

	function xForValue(val: number): number {
		return ((val - min) / range) * W;
	}

	function valueForX(x: number): number {
		return min + (x / W) * range;
	}

	const classColors = ['#3b82f620', '#8b5cf620', '#f59e0b20', '#10b98120', '#ef444420', '#6366f120'];

	function addThreshold() {
		if (thresholds.length >= MAX_THRESHOLDS) return;
		const mid = min + range * ((thresholds.length + 1) / (thresholds.length + 2));
		const next = [...thresholds, Math.round(mid * 100) / 100].sort((a, b) => a - b);
		onchange(next);
	}

	function removeThreshold(idx: number) {
		const next = thresholds.filter((_, i) => i !== idx);
		onchange(next);
	}

	function onSliderInput(idx: number, e: Event) {
		const val = parseFloat((e.target as HTMLInputElement).value);
		const next = [...thresholds];
		next[idx] = val;
		next.sort((a, b) => a - b);
		onchange(next);
	}

	const classLabels = $derived(() => {
		const labels: string[] = [];
		const sorted = [...thresholds].sort((a, b) => a - b);
		for (let i = 0; i <= sorted.length; i++) {
			const lo = i === 0 ? '' : sorted[i - 1].toFixed(2);
			const hi = i === sorted.length ? '' : sorted[i].toFixed(2);
			if (!lo) labels.push(`< ${hi}`);
			else if (!hi) labels.push(`> ${lo}`);
			else labels.push(`${lo} – ${hi}`);
		}
		return labels;
	});
</script>

<div class="hist-container">
	<div class="hist-chart">
		<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="hist-svg">
			<!-- Class region backgrounds -->
			{#each classLabels() as _, i}
				{@const x0 = i === 0 ? 0 : xForValue(thresholds[i - 1])}
				{@const x1 = i === thresholds.length ? W : xForValue(thresholds[i])}
				<rect x={x0} y="0" width={x1 - x0} height={H} fill={classColors[i % classColors.length]} />
			{/each}
			<!-- Bars -->
			{#each histogram.counts as count, i}
				{@const x = xForValue(histogram.edges[i])}
				{@const w = xForValue(histogram.edges[i + 1]) - x}
				{@const h = (count / maxCount) * (H - 4)}
				<rect
					x={x + PAD / 2}
					y={H - h}
					width={Math.max(w - PAD, 1)}
					height={h}
					fill="var(--accent)"
					opacity="0.6"
					rx="1"
				/>
			{/each}
			<!-- Threshold lines -->
			{#each thresholds as t}
				{@const x = xForValue(t)}
				<line x1={x} y1="0" x2={x} y2={H} stroke="var(--error)" stroke-width="2" stroke-dasharray="4 2" />
			{/each}
		</svg>
		<!-- Range sliders overlaid on the SVG -->
		<div class="sliders-overlay">
			{#each thresholds as t, i}
				<input
					type="range"
					class="threshold-slider"
					min={min}
					max={max}
					step={(max - min) / 200}
					value={t}
					oninput={(e) => onSliderInput(i, e)}
				/>
			{/each}
		</div>
	</div>

	<!-- Class labels -->
	<div class="class-labels">
		{#each classLabels() as label, i}
			<span class="class-label" style:background={classColors[i % classColors.length].replace('20', '30')}>Class {i}: {label}</span>
		{/each}
	</div>

	<!-- Controls -->
	<div class="hist-controls">
		<button class="btn-add" onclick={addThreshold} disabled={thresholds.length >= MAX_THRESHOLDS}>+ Add threshold</button>
		{#each thresholds as t, i}
			<span class="threshold-tag">
				{t.toFixed(2)}
				<button class="btn-x" onclick={() => removeThreshold(i)}>&times;</button>
			</span>
		{/each}
	</div>
</div>

<style>
	.hist-container {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.hist-chart {
		position: relative;
		width: 100%;
		height: 120px;
		border: 1px solid var(--border);
		border-radius: 6px;
		overflow: hidden;
		background: var(--bg-primary);
	}
	.hist-svg {
		width: 100%;
		height: 100%;
	}
	.sliders-overlay {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		display: flex;
		flex-direction: column;
		justify-content: center;
		pointer-events: none;
	}
	.threshold-slider {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		margin: 0;
		opacity: 0;
		cursor: col-resize;
		pointer-events: all;
	}
	.class-labels {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.class-label {
		font-size: 11px;
		padding: 2px 8px;
		border-radius: 4px;
		color: var(--text-secondary);
		font-family: var(--font-mono);
	}
	.hist-controls {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 6px;
	}
	.btn-add {
		font-size: 11px;
		padding: 3px 10px;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: none;
		color: var(--text-secondary);
		cursor: pointer;
	}
	.btn-add:hover { background: var(--bg-tertiary); }
	.btn-add:disabled { opacity: 0.4; cursor: not-allowed; }
	.threshold-tag {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		font-family: var(--font-mono);
		padding: 2px 8px;
		border-radius: 4px;
		background: rgba(239, 68, 68, 0.1);
		color: var(--error);
	}
	.btn-x {
		border: none;
		background: none;
		color: var(--error);
		font-size: 14px;
		line-height: 1;
		cursor: pointer;
		padding: 0 2px;
	}
</style>
