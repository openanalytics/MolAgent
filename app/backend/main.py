"""AutoMol Web App -- FastAPI backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .mcp_client import call_tool, disconnect, MCPAuthError
from .routes import admin, datasets, jobs, predict, registry, settings, structures, train, visualize


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await disconnect()


app = FastAPI(title="AutoMol", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(datasets.router)
app.include_router(registry.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(visualize.router)
app.include_router(settings.router)
app.include_router(admin.router)
app.include_router(structures.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models")
async def list_models():
    """List trained models via MCP list_models tool."""
    try:
        result = await call_tool("list_models", {})
        return result
    except MCPAuthError as exc:
        return {"models": [], "error": f"Access denied: {exc}"}
    except RuntimeError as exc:
        return {"models": [], "error": str(exc)}
