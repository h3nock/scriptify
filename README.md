# Scriptify

An end-to-end machine learning system for handwriting synthesis using LSTM neural networks trained on the IAM On-Line Handwriting Database.

## Overview

This repository contains the complete pipeline from model training to deployment, including data preprocessing, neural network implementation, API service, and web interface for handwriting generation.

This implementation is based on Alex Graves' paper ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850).

## Repository Structure

This project is organized into four main components:

- **`ml/`** - Complete ML training pipeline with PyTorch implementation
- **`api/`** - Standalone FastAPI backend service for model inference and flexible deployment
- **`frontend/`** - React TypeScript web application
- **`scriptify-hf-space/`** - HuggingFace Space specific deployment

## Tech Stack

### ML

- **PyTorch** with CUDA for GPU training
- **LSTM + Attention** architecture for handwriting generation
- **Gaussian Mixture Models** for stroke prediction
- **IAM On-Line Handwriting Database** containing 13,040 text lines from 500+ writers
- **Multi-GPU training** with distributed data parallel
- **Weights & Biases** for experiment tracking

### Frontend

- React 19 with TypeScript
- Vite for build tooling
- HTML5 Canvas for rendering handwriting

### Backend

- Python with FastAPI
- PyTorch for model inference
- Pre-trained LSTM models

## Model Architecture

### Core Components

**Three-Layer LSTM Network:**

- Layer 1: Input (3D strokes + character context) → 400 hidden units
- Layer 2: Previous layer + input + attention window → 400 hidden units
- Layer 3: Combined context from all layers → 400 hidden units

**Attention Mechanism:**

- Soft attention over character sequences using Gaussian mixture components
- 10 attention mixture components for character alignment
- Dynamic window positioning with learnable attention weights

**Gaussian Mixture Model Output:**

- 20 mixture components for stroke prediction
- Outputs (x,y) coordinates and end-of-stroke probabilities

### Training Configuration

**Multi-Stage Training:**

- Stage 1: Batch size 64, LR 1e-3, Patience 2000 steps
- Stage 2: Batch size 96, LR 5e-4, Patience 1500 steps
- Stage 3: Batch size 128, LR 1e-4, Patience 1000 steps

**Data Processing:**

- Maximum stroke length: 1200 points
- Maximum text length: 80 characters
- Train/validation split: 90/10

**Distributed Training Support:**

- Multi-GPU training with NCCL backend
- Gradient clipping (value: 5.0)
- Adam optimizer with configurable decay schedules

## Project Structure

```
scriptify/
├── ml/                      # Machine Learning Pipeline
│   ├── src/
│   │   ├── models/         # LSTM, Attention, GMM implementations
│   │   ├── data/           # Data preprocessing and loading
│   │   ├── training/       # Training loops and optimization
│   │   └── utils/          # Utilities and helper functions
│   ├── config/             # Training configurations and hyperparameters
│   ├── scripts/            # Training scripts and distributed computing
│   ├── outputs/            # Model checkpoints and training logs
│   └── environment.yml     # Conda environment specification
├── api/                     # FastAPI backend service
│   ├── src/
│   │   ├── inference_utils.py # Model loading and inference
│   │   └── main.py         # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/               # React Web Application
│   ├── src/
│       ├── components/     # UI components
│       ├── hooks/          # Custom React hooks for API integration
│       ├── services/       # API client
│       └── App.tsx         # Main application component
└── scriptify-hf-space/     # HuggingFace Space deployment
    ├── main.py            # HF Spaces FastAPI entry point
    ├── inference_utils.py # Model inference utilities
    ├── Dockerfile         # HF Spaces container configuration
    ├── requirements.txt   # HF Spaces python dependencies
    ├── packaged_models/   # Model artifacts
    └── styles/            # Pre-trained style templates
        ├── style1.npy
        └── style1.txt
```

## Getting Started

### Prerequisites

- Python 3.9+ (for API and ML)
- Node.js 18+ (for frontend)
- CUDA-compatible GPU (recommended for ML training)
- Conda (for ML env't management)
- Weights & Biases account (for experiment tracking)
- IAM On-Line Handwriting Database

### Model Training

```bash
cd ml

# Setup conda environment
conda env create -f environment.yml
conda activate scriptify_ml_env

# Login to Weights & Biases (only for the first time)
wandb login

# Configure training parameters by editing config/config.yaml

# Single GPU training
bash scripts/run.sh

# Multi-GPU distributed training
bash scripts/run_ddp.sh 0 # 1st node
bash scripts/run_ddp.sh 1 # 2nd node
```

### API Server

```bash
cd api

# Setup Python environment
pip install -r requirements.txt

# Start development server
python src/main.py
# API available at http://localhost:8000
```

### Frontend Application

```bash
cd frontend

# Setup Node.js environment
npm install

# Start development server
npm run dev
# Application available at http://localhost:5173
```

## Deployment

**Frontend**: Deployed to [Vercel](https://scriptify-web.vercel.app)
**Backend**: Deployed on [HuggingFace Spaces](https://huggingface.co/spaces/bitwise42/scriptify-api/tree/main)
