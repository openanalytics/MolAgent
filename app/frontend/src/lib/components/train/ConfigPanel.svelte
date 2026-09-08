<script lang="ts">
	import { pipeline } from '$lib/stores/pipeline.svelte';
	import { runPipeline, type TargetInfo } from '$lib/api/client';
	import HistogramThreshold from './HistogramThreshold.svelte';

	let showAdvanced = $state(false);
	let starting = $state(false);
	let targetThresholds = $state<Record<string, number[]>>({});

	const cfg = pipeline.config;
	const detect = $derived(pipeline.detectResult);
	const opts = $derived(detect?.options);

	const continuousTargets = $derived(
		(detect?.targets ?? []).filter(
			(t): t is TargetInfo & { histogram: NonNullable<TargetInfo['histogram']>; min: number; max: number } =>
				t.task_type === 'regression' && t.histogram != null && t.min != null && t.max != null
		)
	);

	$effect(() => {
		if (cfg.task !== 'Regression') {
			cfg.use_log10 = false;
			cfg.use_logit = false;
		}
	});

	function updateThresholds(column: string, values: number[]) {
		targetThresholds[column] = values;
		targetThresholds = { ...targetThresholds };
		syncClassValues();
	}

	function syncClassValues() {
		const allThresholds = cfg.properties.flatMap(p => targetThresholds[p] ?? []);
		if (allThresholds.length > 0) {
			cfg.class_values = allThresholds.sort((a, b) => a - b);
			cfg.nb_classes = cfg.properties.map(p => (targetThresholds[p]?.length ?? 0) + 1);
		} else {
			cfg.class_values = undefined;
			cfg.nb_classes = [2];
		}
	}

	async function startTraining() {
		starting = true;
		pipeline.error = null;
		try {
			const { job_id } = await runPipeline(cfg);
			pipeline.jobId = job_id;
			pipeline.step = 'training';
		} catch (e) {
			pipeline.error = e instanceof Error ? e.message : 'Failed to start pipeline';
		} finally {
			starting = false;
		}
	}

	function toggleProperty(col: string) {
		const idx = cfg.properties.indexOf(col);
		if (idx >= 0) cfg.properties.splice(idx, 1);
		else cfg.properties.push(col);
		cfg.properties = [...cfg.properties];
	}

	function toggleFeature(key: string) {
		const fk = cfg.feature_keys!;
		const idx = fk.indexOf(key);
		if (idx >= 0) fk.splice(idx, 1);
		else fk.push(key);
		cfg.feature_keys = [...fk];
	}

	function toggleBlender(col: string) {
		const arr = cfg.blender_properties ?? [];
		const idx = arr.indexOf(col);
		if (idx >= 0) arr.splice(idx, 1);
		else arr.push(col);
		cfg.blender_properties = arr.length > 0 ? [...arr] : undefined;
	}

	function toggleInList(field: 'base_list' | 'blender_list' | 'red_dim_list', key: string) {
		const arr = cfg[field] ?? [];
		const idx = arr.indexOf(key);
		if (idx >= 0) arr.splice(idx, 1);
		else arr.push(key);
		cfg[field] = arr.length > 0 ? [...arr] : undefined;
	}

	function multiSelectLabel(field: 'base_list' | 'blender_list' | 'red_dim_list'): string {
		const arr = cfg[field];
		if (!arr || arr.length === 0) return 'Auto (from budget)';
		if (arr.length <= 3) return arr.join(', ');
		return `${arr.length} selected`;
	}
</script>

<div class="config-panel">
	<h3>Pipeline Configuration</h3>
	{#if detect}<p class="hint">Defaults from dataset analysis. Adjust as needed.</p>{/if}

	<fieldset class="section"><legend>Target Properties</legend>
		<div class="cg">{#each opts?.properties ?? cfg.properties as col}<label class="ci"><input type="checkbox" checked={cfg.properties.includes(col)} onchange={() => toggleProperty(col)} /><span class="mono">{col}</span></label>{/each}</div>
	</fieldset>

	<fieldset class="section"><legend>General</legend>
		<div class="fg">
			<div class="ff"><label>SMILES Column</label><select bind:value={cfg.smiles_column}>{#each opts?.smiles_column ?? [cfg.smiles_column] as col}<option value={col}>{col}</option>{/each}</select></div>
			<div class="ff"><label>Task Type</label><select bind:value={cfg.task}><option value="Regression">Regression</option><option value="Classification">Classification</option><option value="RegressionClassification">Regression on Binary (prob.)</option></select></div>
			<div class="ff"><label>Computational Load</label><select bind:value={cfg.computational_load}><option value="free">Free (0-2 min)</option><option value="cheap">Cheap (2-10 min)</option><option value="moderate">Moderate (10-360 min)</option><option value="expensive">Expensive (1-48 hrs)</option></select></div>
			<div class="ff"><label>Split Strategy</label><select bind:value={cfg.split_strategy}><option value="mixed">Mixed</option><option value="stratified">Stratified</option><option value="leave_group_out">Leave-Group-Out</option></select></div>
			<div class="ff"><label>Test Size</label><input type="number" min="0.05" max="0.5" step="0.05" bind:value={cfg.test_size} /></div>
			<div class="ff"><label>Random State</label><input type="number" bind:value={cfg.random_state} /></div>
		</div>
	</fieldset>

	<fieldset class="section"><legend>Feature Generators</legend>
		<div class="cg">{#each opts?.feature_keys ?? ['Bottleneck', 'Bottleneck_chembl37_base', 'Bottleneck_chembl27', 'rdkit', 'fps_2048_2'] as feat}<label class="ci"><input type="checkbox" checked={cfg.feature_keys!.includes(feat)} onchange={() => toggleFeature(feat)} />{feat}</label>{/each}</div>
	</fieldset>

	{#if detect?.blender_properties && detect.blender_properties.length > 0}
		<fieldset class="section"><legend>Blender Properties</legend>
			<p class="hint">Auxiliary columns that provide extra signal during training. The model will require these values at prediction time.</p>
			<div class="cg">
				{#each detect.blender_properties as bp}
					<label class="ci" title={bp.reasons.join('; ')}>
						<input type="checkbox" checked={(cfg.blender_properties ?? []).includes(bp.column)} onchange={() => toggleBlender(bp.column)} />
						<span class="mono">{bp.column}</span>
						<span class="blender-conf" class:strong={bp.confidence === 'strong'} class:moderate={bp.confidence === 'moderate'}>{bp.confidence}</span>
					</label>
				{/each}
			</div>
		</fieldset>
	{/if}

	{#if cfg.task === 'Regression'}
		<fieldset class="section"><legend>Regression Options</legend>
			<div class="cg">
				<label class="ci"><input type="checkbox" bind:checked={cfg.use_log10} /> Apply log10 transform</label>
				<label class="ci"><input type="checkbox" bind:checked={cfg.use_logit} /> Apply logit transform</label>
			</div>
		</fieldset>
	{/if}

	{#if cfg.task === 'Classification' || cfg.task === 'RegressionClassification'}
		<fieldset class="section"><legend>Classification Options</legend>
			<div class="cg"><label class="ci"><input type="checkbox" bind:checked={cfg.categorical} /> Targets are already categorical (0/1/2...)</label></div>

			{#if cfg.categorical}
				<div class="fg" style="margin-top: 10px;"><div class="ff"><label>Number of Classes</label><input type="number" min="2" max="20" value={cfg.nb_classes?.[0] ?? 2} oninput={(e) => { cfg.nb_classes = [parseInt((e.target as HTMLInputElement).value) || 2]; }} /></div></div>
			{:else}
				<p class="hint" style="margin-top: 8px;">Define class boundaries on the distribution. Drag sliders or add thresholds to partition continuous values into classes.</p>
				{#each continuousTargets.filter(t => cfg.properties.includes(t.column)) as target}
					<div class="threshold-section">
						<h5 class="threshold-title">{target.column}</h5>
						<HistogramThreshold
							histogram={target.histogram}
							min={target.min}
							max={target.max}
							thresholds={targetThresholds[target.column] ?? []}
							onchange={(vals) => updateThresholds(target.column, vals)}
						/>
					</div>
				{/each}
				{#if continuousTargets.filter(t => cfg.properties.includes(t.column)).length === 0}
					<div class="fg" style="margin-top: 10px;"><div class="ff"><label>Number of Classes</label><input type="number" min="2" max="20" value={cfg.nb_classes?.[0] ?? 2} oninput={(e) => { cfg.nb_classes = [parseInt((e.target as HTMLInputElement).value) || 2]; }} /></div></div>
				{/if}
			{/if}
		</fieldset>
	{/if}

	<button class="adv-toggle" onclick={() => { showAdvanced = !showAdvanced; }}>{showAdvanced ? 'Hide' : 'Show'} Advanced Options</button>

	{#if showAdvanced}
		<fieldset class="section"><legend>Training Options</legend>
			<div class="fg">
				<div class="ff"><label>Outer CV Folds</label><input type="number" min="2" max="20" bind:value={cfg.outer_folds} /></div>
				<div class="ff"><label>Jobs (inner)</label><input type="number" min="-1" max="32" bind:value={cfg.n_jobs_inner} /></div>
				<div class="ff"><label>Clustering Method</label><select bind:value={cfg.clustering_method}><option value="Bottleneck">Bottleneck</option><option value="Butina">Butina</option><option value="Scaffold">Scaffold</option></select></div>
				<div class="ff"><label>N Clusters</label><input type="number" min="2" max="200" bind:value={cfg.n_clusters} /></div>
				<div class="ff"><label>Butina Cutoff</label><input type="number" min="0.1" max="0.9" step="0.05" bind:value={cfg.butina_cutoff} /></div>
				<div class="ff"><label>CSV Separator</label><select bind:value={cfg.sep}><option value=",">,</option><option value=";">;</option><option value="&#9;">tab</option></select></div>
			</div>
			<div class="cg" style="margin-top: 8px;">
				<label class="ci"><input type="checkbox" bind:checked={cfg.refit} /> Refit on full data</label>
				<label class="ci"><input type="checkbox" bind:checked={cfg.include_test_in_refit} /> Include test set in refit</label>
				<label class="ci"><input type="checkbox" bind:checked={cfg.use_sample_weight} /> Use sample weights</label>
			</div>
			{#if cfg.use_sample_weight}
				<div class="fg" style="margin-top: 8px;">
					<div class="ff"><label>Weight Selection (e.g. &lt;1, &gt;5)</label><input type="text" bind:value={cfg.sample_weight_selection} placeholder="e.g. <1 or >5" /></div>
					<div class="ff"><label>Weight Multiplier</label><input type="number" min="1" max="1000" step="1" bind:value={cfg.sample_weight_multiplier} placeholder="10" /></div>
				</div>
			{/if}
		</fieldset>

		<fieldset class="section"><legend>Model Configuration</legend>
			<div class="fg">
				<div class="ff"><label>Architecture</label>
					<select value={cfg.ensemble_config ?? ''} onchange={(e) => { cfg.ensemble_config = (e.target as HTMLSelectElement).value || undefined; }}>
						<option value="">Auto (from budget)</option>
						{#each Object.entries(opts?.model_configs ?? {}) as [key, desc]}
							<option value={key} title={desc}>{key}</option>
						{/each}
					</select>
				</div>
				<div class="ff"><label>Search Type</label>
					<select value={cfg.search_type ?? ''} onchange={(e) => { cfg.search_type = (e.target as HTMLSelectElement).value || undefined; }}>
						<option value="">Auto</option>
						{#each Object.entries(opts?.search_types ?? {}) as [key, desc]}
							<option value={key} title={desc}>{key}</option>
						{/each}
					</select>
				</div>
				{#if cfg.search_type === 'randomized' || cfg.search_type === 'hyperopt'}
					<div class="ff"><label>Search Iterations</label>
						<input type="number" min="10" max="5000" step="10" bind:value={cfg.randomized_iterations} />
					</div>
				{/if}
				<div class="ff"><label>Scorer</label>
					<select value={cfg.scorer ?? ''} onchange={(e) => { cfg.scorer = (e.target as HTMLSelectElement).value || undefined; }}>
						<option value="">Auto</option>
						{#each Object.entries(opts?.scorers ?? {}) as [key, desc]}
							<option value={key} title={desc}>{key}</option>
						{/each}
					</select>
				</div>
			</div>

			<div class="ms-group">
				<div class="ff">
					<label>Base Estimators <span class="hint-inline">(empty = auto from budget)</span></label>
					<details class="multi-select">
						<summary>{multiSelectLabel('base_list')}</summary>
						<div class="ms-options">
							{#each Object.entries(opts?.base_estimators ?? {}) as [key, desc]}
								<label class="ci" title={desc}>
									<input type="checkbox" checked={(cfg.base_list ?? []).includes(key)} onchange={() => toggleInList('base_list', key)} />
									{key}
								</label>
							{/each}
						</div>
					</details>
				</div>
				<div class="ff">
					<label>Blender Estimators</label>
					<details class="multi-select">
						<summary>{multiSelectLabel('blender_list')}</summary>
						<div class="ms-options">
							{#each Object.entries(opts?.blender_estimators ?? {}) as [key, desc]}
								<label class="ci" title={desc}>
									<input type="checkbox" checked={(cfg.blender_list ?? []).includes(key)} onchange={() => toggleInList('blender_list', key)} />
									{key}
								</label>
							{/each}
						</div>
					</details>
				</div>
				<div class="ff">
					<label>Dimensionality Reduction</label>
					<details class="multi-select">
						<summary>{multiSelectLabel('red_dim_list')}</summary>
						<div class="ms-options">
							{#each Object.entries(opts?.dim_reduction ?? {}) as [key, desc]}
								<label class="ci" title={desc}>
									<input type="checkbox" checked={(cfg.red_dim_list ?? []).includes(key)} onchange={() => toggleInList('red_dim_list', key)} />
									{key}
								</label>
							{/each}
						</div>
					</details>
				</div>
			</div>
		</fieldset>
	{/if}

	<div class="actions">
		{#if pipeline.error}<p class="err">{pipeline.error}</p>{/if}
		<button class="btn-primary" onclick={startTraining} disabled={starting || cfg.properties.length === 0}>{starting ? 'Starting...' : 'Start Training'}</button>
	</div>
</div>

<style>
	.config-panel { display: flex; flex-direction: column; gap: 16px; }
	h3 { font-size: 15px; font-weight: 600; color: var(--text-primary); }
	.hint { font-size: 12px; color: var(--text-muted); margin-top: -8px; }
	.section { border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; background: var(--bg-secondary); }
	legend { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); padding: 0 6px; }
	.fg { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
	.ff { display: flex; flex-direction: column; gap: 4px; }
	.ff label { font-size: 11px; font-weight: 500; color: var(--text-secondary); }
	.ff input, .ff select { padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); font-size: 13px; }
	.ff input:focus, .ff select:focus { outline: none; border-color: var(--accent); }
	.cg { display: flex; flex-wrap: wrap; gap: 8px 16px; }
	.ci { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-secondary); cursor: pointer; }
	.ci input[type="checkbox"] { accent-color: var(--accent); }
	.adv-toggle { align-self: flex-start; padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px; background: none; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
	.adv-toggle:hover { background: var(--bg-tertiary); }
	.actions { display: flex; flex-direction: column; align-items: flex-end; gap: 8px; padding-top: 8px; }
	.err { font-size: 12px; color: var(--error); }
	.btn-primary { padding: 10px 24px; border: none; border-radius: var(--radius); background: var(--accent); color: white; font-size: 14px; font-weight: 600; cursor: pointer; }
	.btn-primary:hover { background: var(--accent-hover); }
	.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
	.hint-inline { font-weight: 400; color: var(--text-muted); }
	.blender-conf { font-size: 9px; padding: 1px 5px; border-radius: 3px; background: var(--bg-tertiary); color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
	.blender-conf.strong { background: rgba(16, 185, 129, 0.12); color: #10b981; }
	.blender-conf.moderate { background: rgba(234, 179, 8, 0.12); color: #b45309; }
	.threshold-section { margin-top: 12px; }
	.threshold-title { font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; }
	.ms-group { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-top: 12px; }
	.multi-select { position: relative; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-primary); }
	.multi-select summary { padding: 6px 10px; font-size: 13px; color: var(--text-secondary); cursor: pointer; list-style: none; }
	.multi-select summary::after { content: '▾'; float: right; color: var(--text-muted); }
	.multi-select[open] summary::after { content: '▴'; }
	.ms-options { display: flex; flex-direction: column; gap: 4px; padding: 8px 10px; border-top: 1px solid var(--border); max-height: 200px; overflow-y: auto; }
	.ms-options .ci { font-size: 12px; }
</style>
