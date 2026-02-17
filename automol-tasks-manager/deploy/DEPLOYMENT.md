# Deploying Trained AutoMol Models

After training a model with the `train-pipeline` skill, you have several options for using it in production.

## What's in the .pt Model File

The `.pt` file is **self-contained** — it includes:
- All feature generators with their weights (Bottleneck encoder, rdkit descriptors, etc.)
- All trained sklearn estimators (XGBoost, LightGBM, stacking meta-learners)
- Model configuration and normalization parameters

You do **NOT** need `automol-resources` at inference time. The pretrained encoder weights are baked into the `.pt` file during training.

**Required runtime dependencies**: `automol`, `torch`, `sklearn`, `rdkit`, `pandas`

**Typical file sizes**: 12-15 MB (2D features only), 150+ MB (with 3D AffGraph features)

---

## Option 1: Predict Skill (Interactive)

The simplest option — use the `predict` skill within Claude Code:

```
> Use the predict skill on my new molecules
```

The skill auto-discovers models from `MolagentFiles/model_registry.json` and runs inference. No server setup needed.

## Option 2: Direct Script (Batch)

Run predictions directly from the command line:

```bash
source ${AUTOMOL_VENV:-.venv}/bin/activate
uv run python skills/predict/scripts/predict.py \
    --model-file MolagentFiles/{run_folder}/gamma1_refitted_stackingregmodel.pt \
    --smiles-file new_molecules.csv \
    --output-folder results/ \
    --verbose
```

Output: `results/gamma1_predictions.csv` with a `predicted_gamma1` column. The script auto-detects the SMILES column — use `--smiles-column` only if your column has a non-standard name.

For merged models (multi-property), a single invocation predicts all properties at once. For individual models, run once per property.

## Option 3: REST API Server (Production)

Deploy the model as a FastAPI server for programmatic access.

### Quick Start

```bash
source ${AUTOMOL_VENV:-.venv}/bin/activate
MODEL_PATH=MolagentFiles/{run_folder}/gamma1_refitted_stackingregmodel.pt \
    python -m deploy.server.app --port 8000
```

Server starts at `http://localhost:8000` with auto-generated docs at `/docs`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — confirms model is loaded |
| `/model/info` | GET | Model metadata: properties, features, device |
| `/predict` | POST | Single SMILES prediction |
| `/predict/batch` | POST | Batch predictions (list of SMILES) |
| `/features` | POST | Extract raw feature vectors |
| `/model/load` | POST | Hot-load a different model file |

### Example Requests

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"smiles": "CCO"}'
```

**Batch prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"smiles_list": ["CCO", "CCN", "c1ccccc1"]}'
```

### Python Client

```python
from deploy.client import AutoMolClient

client = AutoMolClient("http://localhost:8000")

# Single prediction
result = client.predict("CCO")
# → {"gamma1": 1.23}

# Batch prediction
results = client.predict_batch(["CCO", "CCN", "c1ccccc1"])
# → [{"gamma1": 1.23}, {"gamma1": 2.45}, {"gamma1": 0.87}]

# Predict from CSV file
df = client.predict_file("new_data.csv", smiles_column="SMILES", output_path="predictions.csv")
```

## Option 4: Docker (Isolated / Cloud)

For containerized deployment or cloud environments.

### Build and Run

```bash
# Build from AutoMol root directory
docker build -t automol-server \
    -f plugins/automol-models/deploy/docker/Dockerfile .

# Run with model mounted
docker run -p 8000:8000 \
    -v $(pwd)/MolagentFiles/{run_folder}:/models:ro \
    -e MODEL_PATH=/models/gamma1_refitted_stackingregmodel.pt \
    automol-server
```

### Docker Compose

```bash
# CPU mode
MODELS_DIR=./MolagentFiles/{run_folder} MODEL_NAME=gamma1_refitted_stackingregmodel.pt \
    docker-compose -f deploy/docker/docker-compose.yml up -d

# GPU mode
docker-compose -f deploy/docker/docker-compose.yml --profile gpu up -d

# With nginx reverse proxy (rate limiting, load balancing)
docker-compose -f deploy/docker/docker-compose.yml --profile with-nginx up -d
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `automol-api` | 8000 | CPU inference server |
| `automol-api-gpu` | 8001 | GPU inference server (needs `--profile gpu`) |
| `nginx` | 80/443 | Reverse proxy with rate limiting (needs `--profile with-nginx`) |
| `redis` | 6379 | Prediction cache (needs `--profile with-cache`) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/model.pt` | Path to model file inside container |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Uvicorn worker count |
| `USE_GPU` | `false` | Enable CUDA inference |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `MODEL_DIR` | `MolagentFiles` | Allowed directory for `/model/load` endpoint |

---

## Choosing a Deployment Option

| Scenario | Recommended Option |
|----------|--------------------|
| Interactive exploration | Option 1: Predict skill |
| One-off batch processing | Option 2: Direct script |
| Application integration | Option 3: REST API |
| Cloud / multi-user | Option 4: Docker |
| CI/CD pipeline | Option 2 or 4 |

## Security Notes

- The `/model/load` endpoint restricts paths to the `MODEL_DIR` directory to prevent path traversal
- The Docker container runs models as read-only mounts (`:ro`)
- Nginx config includes rate limiting (10 req/s with burst of 20)
- For production, add authentication (API key via `Authorization: Bearer` header — supported by the client)
