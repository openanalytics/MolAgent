const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		headers: { 'Content-Type': 'application/json', ...init?.headers },
		...init,
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(err.detail || res.statusText);
	}
	return res.json();
}

// Settings
export const getSettings = () => request<MCPSettingsResponse>('/settings');
export const updateSettings = (s: MCPSettingsUpdate) =>
	request<MCPSettingsResponse>('/settings', { method: 'PUT', body: JSON.stringify(s) });

// Jobs
export const getJobStatus = (id: string) => request<JobStatus>(`/jobs/${id}/status`);
export const getJobLogs = (id: string) => request<JobLogs>(`/jobs/${id}/logs`);
export const cancelJob = (id: string) => request<{ status: string; job_id: string }>(`/jobs/${id}/cancel`, { method: 'POST' });

// Models (via MCP list_models)
export const listModels = () => request<ModelsResponse>('/models');

// Train
export async function uploadFile(file: File): Promise<UploadResult> {
	const form = new FormData();
	form.append('file', file);
	const res = await fetch(`${BASE}/train/upload`, { method: 'POST', body: form });
	if (!res.ok) throw new Error('Upload failed');
	return res.json();
}

export const detectDataset = (opts: { csv_path?: string; dataset_id?: string }) =>
	request<DetectSessionResult>('/train/detect', {
		method: 'POST',
		body: JSON.stringify(opts),
	});

export const configureSession = (req: ConfigureRequest) =>
	request<ConfigureResponse>('/train/configure', { method: 'POST', body: JSON.stringify(req) });

export const runPipeline = (config: TrainingConfig) =>
	request<{ job_id: string }>('/train/run', {
		method: 'POST',
		body: JSON.stringify({ config }),
	});

export const listRuns = () => request<PipelineState[]>('/train/runs');
export const getRun = (runId: string) => request<PipelineState>(`/train/runs/${runId}`);

// Predict
export const runPredict = (req: PredictRequest) =>
	request<{ job_id: string }>('/predict/run', { method: 'POST', body: JSON.stringify(req) });

export const getPredictResult = (jobId: string) =>
	request<{ csv_path: string; filename: string }>(`/predict/${jobId}/result`);

export async function downloadPredictions(jobId: string): Promise<string> {
	const res = await fetch(`${BASE}/predict/${jobId}/download`);
	if (!res.ok) throw new Error('Download failed');
	return res.text();
}

// Visualize
export const getDashboardHtml = async (runId: string): Promise<string> => {
	const res = await fetch(`${BASE}/visualize/${runId}/html`);
	if (!res.ok) throw new Error('Dashboard not available');
	return res.text();
};

export const getDashboardFromJob = async (jobId: string): Promise<string> => {
	const res = await fetch(`${BASE}/visualize/job/${jobId}/html`);
	if (!res.ok) throw new Error('Dashboard not available');
	return res.text();
};

// Datasets
export const listDatasets = () => request<DatasetEntry[]>('/datasets');
export const deleteDataset = (id: string) =>
	request<{ status: string; dataset_id: string; filename: string }>(`/datasets/${id}`, { method: 'DELETE' });
export async function uploadDatasetFile(file: File): Promise<DatasetUploadResult> {
	const form = new FormData();
	form.append('file', file);
	const res = await fetch(`${BASE}/datasets/upload`, { method: 'POST', body: form });
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(err.detail || 'Upload failed');
	}
	return res.json();
}

export async function uploadStructure(
	targetName: string,
	sdfFile: File,
	pdbFiles: File[],
	ligandsCsv?: File
): Promise<StructureUploadResult> {
	const form = new FormData();
	form.append('target_name', targetName);
	form.append('sdf_file', sdfFile);
	for (const f of pdbFiles) form.append('pdb_files', f);
	if (ligandsCsv) form.append('ligands_csv', ligandsCsv);
	const res = await fetch(`${BASE}/structures/upload`, { method: 'POST', body: form });
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(err.detail || 'Upload failed');
	}
	return res.json();
}

// Registry / Delete / Merge
export const deleteModel = (modelId: string) =>
	request<{ status: string; id: string; files_removed: number }>(`/registry/${modelId}`, { method: 'DELETE' });

export const mergeModels = (req: MergeRequest) =>
	request<{ job_id: string }>('/registry/merge', { method: 'POST', body: JSON.stringify(req) });

// Admin
export const adminManage = (action: string, user_id?: string, owner_id?: string) =>
	request<Record<string, unknown>>('/admin/manage', {
		method: 'POST',
		body: JSON.stringify({ action, user_id, owner_id }),
	});

// Health
export const getHealth = () => request<{ status: string }>('/health');

// ── Types ───────────────────────────────────────────────────────────────────

export interface MCPSettingsResponse {
	mode: string;
	url: string | null;
	server_path: string | null;
	has_auth: boolean;
	output_folder: string | null;
	warnings?: string[];
}

export interface MCPSettingsUpdate {
	mode?: 'local' | 'remote';
	url?: string;
	server_path?: string;
	auth_token?: string;
	output_folder?: string;
}

export interface JobStatus {
	id: string;
	description: string;
	status: 'pending' | 'running' | 'success' | 'failed';
	exit_code: number | null;
	created_at: string;
	finished_at: string | null;
	result: unknown;
	progress: number;
	progress_total: number;
	progress_label: string;
}

export interface JobLogs {
	id: string;
	lines: string[];
}

export interface UploadResult {
	path: string;
	filename: string;
	size: number;
	columns?: string[];
	dataset_id?: string;
}

export interface TargetHistogram {
	counts: number[];
	edges: number[];
}

export interface TargetInfo {
	column: string;
	task_type: 'regression' | 'classification';
	unique_values: number;
	null_count: number;
	mean?: number;
	std?: number;
	min?: number;
	max?: number;
	skewness?: number;
	kurtosis?: number;
	is_skewed?: boolean;
	suggest_log_transform?: boolean;
	all_positive?: boolean;
	histogram?: TargetHistogram;
	suggested_nb_classes?: number;
	suggested_categorical?: boolean;
	class_values_detected?: number[];
	class_distribution?: Record<string, number>;
}

export interface BlenderProperty {
	column: string;
	reasons: string[];
	correlation: number;
	confidence: 'strong' | 'moderate' | 'low';
}

export interface DataCharacteristics {
	valid_smiles: number;
	total_smiles: number;
	smiles_validity_rate: number;
	class_balance: Record<string, unknown>;
	high_correlations: Array<{ target_a: string; target_b: string; correlation: number }>;
	null_rates: Record<string, number>;
}

export interface DetectSessionResult {
	session_id: string;
	detected: {
		csv_file?: string;
		smiles_column: string;
		properties: string[];
		task: string;
		use_log10: boolean;
		feature_keys: string[];
		split_strategy: string;
		computational_load: string;
		categorical?: boolean;
		nb_classes?: number[] | null;
		sdf_file?: string | null;
		protein_folder?: string | null;
	};
	options: {
		task: string[];
		computational_load: Record<string, string>;
		feature_keys: string[];
		split_strategy: Record<string, string>;
		smiles_column: string[];
		properties: string[];
		base_estimators?: Record<string, string>;
		blender_estimators?: Record<string, string>;
		dim_reduction?: Record<string, string>;
		model_configs?: Record<string, string>;
		search_types?: Record<string, string>;
		scorers?: Record<string, string>;
	};
	question: string;
	targets?: TargetInfo[];
	blender_properties?: BlenderProperty[];
	characteristics?: DataCharacteristics;
}

export interface ConfigureRequest {
	session_id: string;
	confirm?: boolean;
	smiles_column?: string;
	properties?: string[];
	task?: string;
	computational_load?: string;
	feature_keys?: string[];
	use_log10?: boolean;
	use_logit?: boolean;
	split_strategy?: string;
	base_list?: string[];
	blender_list?: string[];
	red_dim_list?: string[];
	model_config?: string;
	search_type?: string;
	randomized_iterations?: number;
	scorer?: string;
	// Classification options
	categorical?: boolean;
	nb_classes?: number[];
	class_values?: number[];
	class_quantiles?: number[];
	// Refit control
	refit?: boolean;
	include_test_in_refit?: boolean;
	// Sample weights
	use_sample_weight?: boolean;
	sample_weight_selection?: string;
	sample_weight_multiplier?: number;
}

export interface ConfigureResponse {
	session_id: string;
	validation_error: boolean;
	question: string | null;
	config: TrainingConfig | null;
}

export interface TrainingConfig {
	csv_file: string;
	smiles_column: string;
	properties: string[];
	task: 'Regression' | 'Classification' | 'RegressionClassification';
	sep?: string;
	computational_load?: string;
	feature_keys?: string[];
	use_log10?: boolean;
	use_logit?: boolean;
	categorical?: boolean;
	nb_classes?: number[];
	class_values?: number[];
	class_quantiles?: number[];
	split_strategy?: string;
	test_size?: number;
	outer_folds?: number;
	random_state?: number;
	n_jobs_inner?: number;
	n_jobs_outer?: number;
	sdf_file?: string;
	protein_folder?: string;
	refit?: boolean;
	include_test_in_refit?: boolean;
	// Tier 1 extensions
	blender_properties?: string[];
	scorer?: string;
	use_advanced?: boolean;
	use_sample_weight?: boolean;
	sample_weight_selection?: string;
	sample_weight_multiplier?: number;
	clustering_method?: string;
	n_clusters?: number;
	butina_cutoff?: number;
	// Tier 2 extensions — advanced model configuration
	base_list?: string[];
	blender_list?: string[];
	red_dim_list?: string[];
	ensemble_config?: string;
	search_type?: string;
	randomized_iterations?: number;
}

export interface TrainingResult {
	run_id: string;
	output_folder: string;
	model_id: string | null;
	model_filename: string;
	dashboard_html: string;
	metrics: Record<string, Record<string, number>>;
	properties: string[];
	task: string;
	model_path: string;
	dashboard_path: string;
	pipeline_state_path: string;
	train_info: Record<string, unknown>;
}

export interface DownloadModelResult {
	model_id: string;
	model_filename: string;
	saved_path: string | null;
	size_bytes: number;
}

export const downloadModel = (modelId: string) =>
	request<DownloadModelResult>(`/registry/${modelId}/download`, { method: 'POST' });

export interface PipelineState {
	pipeline_version: string;
	run_id: string;
	current_step: number;
	steps_completed: number[];
	pipeline_complete: boolean;
	config: Record<string, unknown>;
	outputs: Record<string, unknown>;
	metrics: Record<string, Record<string, number>>;
	source?: 'local' | 'remote';
}

export interface ModelsResponse {
	models: ModelEntry[];
	registry_path?: string;
	error?: string;
}

export interface ClassificationInfo {
	categorical?: boolean;
	nb_classes?: number[] | null;
	class_values?: number[] | null;
	labelnames?: Record<string, Record<string, string>> | null;
}

export interface ModelEntry {
	id: string;
	target_properties: string[];
	task_type: string;
	metrics: Record<string, Record<string, number>>;
	feature_keys: string[];
	source_dataset?: string;
	is_refitted?: boolean;
	computational_load?: string;
	model_format?: string;
	created_at?: string;
	blender_properties?: string[];
	classification?: ClassificationInfo;
}


export interface PredictRequest {
	model_id?: string;
	model_file?: string;
	smiles_file?: string;
	smiles_list?: string[];
	smiles_column?: string;
	properties?: string[];
	compute_sd?: boolean;
	blender_properties?: string[];
	blender_values?: Record<string, number>;
	convert_log10?: boolean;
}

export interface MergeRequest {
	model_ids: string[];
	output_name?: string;
	verify_encoder?: boolean;
}

export interface DatasetEntry {
	id: string;
	filename: string;
	size_bytes: number;
	columns: string[];
	row_count: number;
	uploaded_at: string;
	last_used: string;
	owner?: string;
	type?: string;
}

export interface DatasetUploadResult {
	dataset_id: string;
	filename: string;
	columns: string[];
	row_count: number;
	size_bytes: number;
}

export interface StructureUploadResult {
	structure_id: string;
	target_name: string;
	sdf_path: string;
	pdb_folder: string;
	sdf_record_count: number;
	pdb_count: number;
}
