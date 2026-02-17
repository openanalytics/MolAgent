"""
FastAPI server for AutoMol model inference.

Provides REST API endpoints for molecular property prediction using trained AutoMol models.

Usage:
    # Start server directly
    python -m deploy.server.app --model model.pt --port 8000

    # Or use uvicorn
    MODEL_PATH=/path/to/model.pt uvicorn deploy.server.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class PredictRequest(BaseModel):
    """Single SMILES prediction request."""
    smiles: str = Field(..., description="SMILES string to predict")
    properties: Optional[List[str]] = Field(None, description="Properties to predict (None for all)")

    model_config = {"json_schema_extra": {"example": {"smiles": "CCO"}}}


class BatchPredictRequest(BaseModel):
    """Batch SMILES prediction request."""
    smiles_list: List[str] = Field(..., description="List of SMILES strings")
    properties: Optional[List[str]] = Field(None, description="Properties to predict (None for all)")

    model_config = {
        "json_schema_extra": {
            "example": {"smiles_list": ["CCO", "CCN", "c1ccccc1"]}
        }
    }


class PredictResponse(BaseModel):
    """Single prediction response."""
    smiles: str
    predictions: Dict[str, Optional[float]]
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictResponse]
    count: int
    processing_time_ms: float


class FeatureRequest(BaseModel):
    """Feature extraction request."""
    smiles_list: List[str]


class FeatureResponse(BaseModel):
    """Feature extraction response."""
    features: Dict[str, List[List[float]]]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: Optional[str]
    version: str
    device: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    properties: List[str]
    feature_generators: List[str]
    model_type: str
    use_gpu: bool


# ============================================================================
# Global State
# ============================================================================

class ModelState:
    """Global model state."""
    model = None
    model_path: Optional[str] = None
    load_time: Optional[float] = None
    device: str = "cpu"
    properties: List[str] = []


state = ModelState()


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_path = os.environ.get("MODEL_PATH", None)
    use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"

    if model_path:
        load_model(model_path, use_gpu=use_gpu)
    else:
        logger.warning("No MODEL_PATH set. Use /model/load endpoint to load a model.")

    yield

    # Cleanup
    state.model = None
    logger.info("Server shutdown complete")


# ============================================================================
# FastAPI App
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AutoMol API",
        description="REST API for molecular property prediction using AutoMol",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI):
    """Register all API routes."""

    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """API root - returns API information."""
        return {
            "name": "AutoMol API",
            "version": "1.0.0",
            "description": "Molecular property prediction API",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "batch_predict": "/predict/batch",
                "features": "/features",
                "model_info": "/model/info",
            },
            "model_loaded": state.model is not None,
            "properties": state.properties,
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        try:
            from automol.version import __version__
        except ImportError:
            __version__ = "unknown"

        return HealthResponse(
            status="healthy",
            model_loaded=state.model is not None,
            model_path=state.model_path,
            version=__version__,
            device=state.device,
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info():
        """Get model information."""
        if state.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        model = state.model

        # Get feature generators
        feature_gens = []
        if hasattr(model, 'feature_generators'):
            feature_gens = list(model.feature_generators.keys())

        # Get model type
        model_type = type(model).__name__

        # Check GPU usage
        use_gpu = state.device != "cpu"

        return ModelInfoResponse(
            properties=state.properties,
            feature_generators=feature_gens,
            model_type=model_type,
            use_gpu=use_gpu,
        )

    @app.post("/model/load")
    async def load_model_endpoint(model_path: str, use_gpu: bool = False):
        """Load a model from path."""
        # Validate model path to prevent path traversal
        resolved = Path(model_path).resolve()
        allowed_dir = Path(os.environ.get("MODEL_DIR", "MolagentFiles")).resolve()
        if not str(resolved).startswith(str(allowed_dir)):
            raise HTTPException(status_code=403, detail=f"Model path must be within {allowed_dir}")
        try:
            load_model(str(resolved), use_gpu=use_gpu)
            return {"status": "success", "model_path": str(resolved), "properties": state.properties}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """Predict for a single SMILES."""
        if state.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        try:
            # Get properties to predict
            props = request.properties if request.properties else state.properties

            # Make prediction
            result = state.model.predict(
                smiles=[request.smiles],
                props=props if props else None
            )

            # Format response
            predictions = {}
            for prop in props:
                pred_key = f"predicted_{prop}"
                if pred_key in result:
                    val = result[pred_key]
                    if hasattr(val, '__iter__'):
                        predictions[prop] = float(val[0]) if val[0] is not None else None
                    else:
                        predictions[prop] = float(val) if val is not None else None

            return PredictResponse(
                smiles=request.smiles,
                predictions=predictions,
            )
        except Exception as e:
            logger.error(f"Prediction error for {request.smiles}: {e}")
            return PredictResponse(
                smiles=request.smiles,
                predictions={},
                error=str(e),
            )

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(request: BatchPredictRequest):
        """Predict for multiple SMILES."""
        if state.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        start_time = time.time()
        predictions = []

        try:
            # Get properties to predict
            props = request.properties if request.properties else state.properties

            # Make batch prediction
            result = state.model.predict(
                smiles=request.smiles_list,
                props=props if props else None
            )

            # Format response for each SMILES
            for i, smiles in enumerate(request.smiles_list):
                pred_dict = {}
                for prop in props:
                    pred_key = f"predicted_{prop}"
                    if pred_key in result:
                        val = result[pred_key]
                        if hasattr(val, '__iter__') and len(val) > i:
                            pred_dict[prop] = float(val[i]) if val[i] is not None else None
                        else:
                            pred_dict[prop] = None

                predictions.append(PredictResponse(
                    smiles=smiles,
                    predictions=pred_dict,
                ))

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Return partial results with errors
            for smiles in request.smiles_list:
                predictions.append(PredictResponse(
                    smiles=smiles,
                    predictions={},
                    error=str(e),
                ))

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictResponse(
            predictions=predictions,
            count=len(predictions),
            processing_time_ms=processing_time,
        )

    @app.post("/features", response_model=FeatureResponse)
    async def get_features(request: FeatureRequest):
        """Get feature vectors for SMILES."""
        if state.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        try:
            # Generate features using the model's feature generators
            features_dict = {}

            if hasattr(state.model, 'feature_generators'):
                for name, gen in state.model.feature_generators.items():
                    if hasattr(gen, 'generate_features'):
                        feats = gen.generate_features(request.smiles_list)
                    elif hasattr(gen, '__call__'):
                        feats = gen(request.smiles_list)
                    else:
                        continue

                    if hasattr(feats, 'tolist'):
                        features_dict[name] = feats.tolist()
                    else:
                        features_dict[name] = list(feats)

            return FeatureResponse(
                features=features_dict,
                count=len(request.smiles_list),
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

def load_model(model_path: str, use_gpu: bool = False) -> None:
    """Load AutoMol model from file."""
    import torch
    from automol.stacking import load_model as automol_load_model

    logger.info(f"Loading model from {model_path}")
    start_time = time.time()

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    state.model = automol_load_model(
        model_file=model_path,
        use_gpu=use_gpu,
        retrieve_reproducability_errors=False
    )
    state.model_path = model_path
    state.load_time = time.time() - start_time

    # Set device
    if use_gpu and torch.cuda.is_available():
        state.device = "cuda"
    else:
        state.device = "cpu"

    # Get available properties
    if hasattr(state.model, 'models'):
        state.properties = list(state.model.models.keys())
    else:
        state.properties = []

    logger.info(f"Model loaded in {state.load_time:.2f}s")
    logger.info(f"Available properties: {state.properties}")
    logger.info(f"Using device: {state.device}")


def run_server(
    model_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    use_gpu: bool = False,
):
    """Run the FastAPI server."""
    import uvicorn

    if model_path:
        os.environ["MODEL_PATH"] = model_path
    if use_gpu:
        os.environ["USE_GPU"] = "true"

    uvicorn.run(
        "deploy.server.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


# Create app instance
app = create_app()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoMol API Server")
    parser.add_argument("--model", type=str, help="Path to model file (.pt)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    run_server(
        model_path=args.model,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
