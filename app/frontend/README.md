# AutoMol Frontend

SvelteKit 5 + Tailwind CSS 4 app. Communicates with the FastAPI backend at port 8000 via Vite proxy.

## Development

```bash
npm install
npm run dev        # http://localhost:5173
```

Requires the backend running at port 8000 (see `../README.md`).

## Build

```bash
npm run build      # Static site output via @sveltejs/adapter-static
npm run preview    # Preview the build
```

## Type Checking

```bash
npm run check      # svelte-check + TypeScript
```
