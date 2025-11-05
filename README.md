# Scriptify

End-to-end handwriting synthesis: train → package → serve → web.

Built on PyTorch with an LSTM + attention model trained on the IAM On‑Line Handwriting Database. Based on Alex Graves' "Generating Sequences With Recurrent Neural Networks" (https://arxiv.org/abs/1308.0850).

## Features

- Train LSTM + attention with GMM output on IAM
- Multi‑GPU training (DDP) and resumable checkpoints
- Weights & Biases logging (optional)
- FastAPI inference service + React frontend
- HuggingFace Space deployment template

## Project Structure

```
scriptify/
├── ml/                      # Training pipeline (PyTorch)
│   ├── src/                 # models/, data/, training/, utils/
│   ├── config/              # config.yaml (paths, hyperparams)
│   ├── scripts/             # run.sh, run_ddp.sh
│   ├── outputs/             # runs/, checkpoints/, logs
│   └── packaged_models/     # packaged artifacts for serving
├── api/                     # FastAPI inference service
│   └── src/
│       ├── main.py          # API entry
│       └── inference_utils.py
└── frontend/                # React + TypeScript app
    └── src/                 # components/, services/, App.tsx
```

## Quickstart

### 1) Setup

- Python 3.9+, Conda, Node 18+, CUDA GPU recommended
- Obtain IAM On‑Line Handwriting (ascii/ and lineStrokes/)

### 2) Prepare data

```bash
cd ml
# Edit config/config.yaml → set:
#   paths.raw_data_root       (contains ascii/ and lineStrokes/)
#   paths.processed_data_dir  (write location for .npy files)
conda env create -f environment.yml && conda activate scriptify_ml_env
python -m src.data.dataset   # writes processed arrays to paths.processed_data_dir
```

### 3) Train

```bash
cd ml
wandb login                  # optional
bash scripts/run.sh          # single GPU (torchrun)

# Multi-node/GPU (requires ml/.env with DISTRIBUTED settings)
# Save this as ml/.env (example values):
#   SCRIPTIFY_DIST_MASTER_ADDR=10.0.0.1
#   SCRIPTIFY_DIST_MASTER_PORT=29500
#   SCRIPTIFY_DIST_NNODES=2
#   SCRIPTIFY_DIST_NPROC_PER_NODE=2
bash scripts/run_ddp.sh 0    # node rank 0
```

### 4) Package model for serving

The API loads from `ml/packaged_models` and expects these filenames:
- `handwriting_model.pt`
- `handwriting_model.scripted.pt`

Generate matching files with:

```bash
cd ml
python -m src.package_model --pkg_name handwriting_model
```

Or specify paths explicitly:

```bash
python -m src.package_model \
  --checkpoint <path/to/checkpoint> \
  --config <path/to/config.yaml> \
  --pkg_name handwriting_model
```

If you change names or locations, update `api/src/main.py` constants (`MODEL_DIR`, `SCRIPTED_MODEL_NAME`, `METADATA_MODEL_NAME`) or rename the files accordingly.

### 5) Run API

```bash
cd api
pip install -r requirements.txt
cd src
python main.py               # http://localhost:8000  (see /docs)
# Alternatively: uvicorn main:app --reload
```

### 6) Run frontend

```bash
cd frontend
npm install
npm run dev                  # http://localhost:5173
```

## Model (brief)

- 3‑layer LSTM with soft attention over characters
- GMM output for (dx, dy, pen‑up) stroke prediction
- Tunable via `ml/config/config.yaml`

## Configuration

Edit `ml/config/config.yaml` (or set `SCRIPTIFY_CONFIG_PATH`). Key fields:

- `paths.raw_data_root` — IAM dataset root (contains `ascii/` and `lineStrokes/`)
- `paths.processed_data_dir` — where processed `.npy` are saved
- `paths.outputs_dir` — training runs and checkpoints
- `dataset.max_stroke_len`, `dataset.max_text_len`, `dataset.alphabet_string`
- `wandb.enabled`, `wandb.project_name`

## Deployment

- Frontend: Vercel — https://scriptify-web.vercel.app
- Backend: HuggingFace Spaces — https://huggingface.co/spaces/h3nock/scriptify-api
