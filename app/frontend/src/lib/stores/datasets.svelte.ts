import { listDatasets, type DatasetEntry } from '$lib/api/client';

export const datasets = $state({
	entries: [] as DatasetEntry[],
	loading: false,
	error: null as string | null,
});

export async function refreshDatasets() {
	datasets.loading = true;
	datasets.error = null;
	try {
		const result = await listDatasets();
		datasets.entries = result ?? [];
	} catch (e) {
		datasets.error = e instanceof Error ? e.message : 'Failed to load datasets';
		datasets.entries = [];
	} finally {
		datasets.loading = false;
	}
}
