import type { DetectSessionResult, TrainingConfig, TrainingResult } from '$lib/api/client';

export function defaultConfig(): TrainingConfig {
	return {
		csv_file: '',
		smiles_column: '',
		properties: [],
		task: 'Regression',
		sep: ',',
		computational_load: 'cheap',
		feature_keys: ['Bottleneck'],
		use_log10: false,
		use_logit: false,
		categorical: false,
		nb_classes: [2],
		split_strategy: 'mixed',
		test_size: 0.25,
		outer_folds: 4,
		random_state: 42,
		n_jobs_inner: 2,
		refit: true,
		include_test_in_refit: true,
		use_sample_weight: false,
		sample_weight_selection: undefined,
		sample_weight_multiplier: undefined,
		clustering_method: 'Bottleneck',
		n_clusters: 20,
		butina_cutoff: 0.6,
	};
}

export const pipeline = $state({
	step: 'upload' as 'upload' | 'detect' | 'configure' | 'training' | 'done',
	csvPath: null as string | null,
	csvFilename: null as string | null,
	datasetId: null as string | null,
	detectResult: null as DetectSessionResult | null,
	config: defaultConfig(),
	jobId: null as string | null,
	result: null as TrainingResult | null,
	error: null as string | null,
});

export function resetPipeline() {
	pipeline.step = 'upload';
	pipeline.csvPath = null;
	pipeline.csvFilename = null;
	pipeline.datasetId = null;
	pipeline.detectResult = null;
	pipeline.config = defaultConfig();
	pipeline.jobId = null;
	pipeline.result = null;
	pipeline.error = null;
}

export function applyDetectDefaults(detect: DetectSessionResult) {
	const cfg = pipeline.config;
	const d = detect.detected;
	cfg.smiles_column = d.smiles_column;
	cfg.properties = d.properties.length > 0 ? [d.properties[0]] : [];
	cfg.task = d.task as TrainingConfig['task'];
	cfg.use_log10 = d.use_log10;
	cfg.feature_keys = [...d.feature_keys];
	cfg.split_strategy = d.split_strategy;
	cfg.computational_load = d.computational_load;
	cfg.sdf_file = d.sdf_file ?? undefined;
	cfg.protein_folder = d.protein_folder ?? undefined;
	if (cfg.task === 'Classification') {
		cfg.scorer = 'balanced_accuracy';
		if (d.categorical != null) cfg.categorical = d.categorical;
		if (d.nb_classes) cfg.nb_classes = d.nb_classes;
	}
}
