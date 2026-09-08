import { listModels, type ModelEntry } from '$lib/api/client';

export const registry = $state({
	entries: [] as ModelEntry[],
	loading: false,
	error: null as string | null,
});

export async function refreshRegistry() {
	registry.loading = true;
	registry.error = null;
	try {
		const result = await listModels();
		registry.entries = result.models ?? [];
		if (result.error) {
			registry.error = result.error;
		}
	} catch (e) {
		registry.error = e instanceof Error ? e.message : 'Failed to load models';
		registry.entries = [];
	} finally {
		registry.loading = false;
	}
}
