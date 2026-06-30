# ExoNova — AI-Powered Exoplanet Discovery Platform
### ISRO Hackathon 2026 · Problem Statement 7


![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

A production-grade AI pipeline for detecting and classifying exoplanet transit signals in noisy stellar light curves.

---

# Overview

ExoNova is an end-to-end machine learning platform that analyzes stellar light curves to identify exoplanet transit candidates.

The platform combines classical astronomy algorithms with deep learning to distinguish real planetary transits from common false positives such as eclipsing binaries, stellar variability, starspot activity, and instrumental noise.

Key capabilities include:

- Physics-based synthetic data generation
- Automated preprocessing and detrending
- Box Least Squares (BLS) transit search
- 1D CNN classification
- Ensemble validation
- Explainable AI using Integrated Gradients
- FastAPI REST backend
- React dashboard with interactive visualizations
- Validation against curated NASA Kepler Objects of Interest

---

#  Why ExoNova?

Traditional exoplanet detection pipelines require significant manual inspection and domain expertise. ExoNova streamlines this process by combining astronomical signal processing, deep learning, and explainable AI into a unified platform.

The goal is to accelerate candidate screening while maintaining transparency through interpretable predictions.

# Features

## AI Detection Pipeline

- Synthetic light-curve generation
- Automated preprocessing
- BLS transit detection
- CNN-based classification
- Ensemble refinement
- Confidence calibration

## Explainable AI

- Integrated Gradients
- Confidence scores
- Prediction explanations
- Saliency visualization

## Interactive Dashboard

- Upload CSV/TSV light curves
- Demo signal generator
- Analyze curated NASA targets
- Prediction history
- Interactive charts

## Backend

- FastAPI
- SQLite persistence
- OpenAPI/Swagger docs
- REST APIs

---

## Quick Start

```bash
# 1. Install Python dependencies
cd ml && pip install -r requirements.txt --break-system-packages

# 2. Train the model (generates 17,500 synthetic samples, trains 1D-CNN)
python3 train.py --n-per-class 600 --epochs 40
# ~20 min total on a CPU. A trained checkpoint (run_20260621_200934) is already included.

# 3. Start both services
cd ..
./start.sh
# Backend:  http://localhost:8000/api
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Dashboard (port 5173)             │
│  Upload CSV │ Demo Generator │ Real KOI Targets │ Results   │
└────────────────────────┬────────────────────────────────────┘
                         │ /api/* (Vite proxy)
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend (port 8000)               │
│  /upload  /analyze  /history  /candidates  /real-targets    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Detection Pipeline                        │
│  1. Clean/detrend   →   preprocessing.py                    │
│  2. BLS search      →   features.py                         │
│  3. CNN classify    →   model_cnn.py (353K params)          │
│  4. Ensemble vet    →   ensemble.py                         │
│  5. Explain         →   explain.py (Integrated Gradients)   │
└─────────────────────────────────────────────────────────────┘
```

---

# Machine Learning Pipeline

1. Load stellar light curve
2. Clean and detrend signal
3. Normalize flux
4. Extract transit features
5. Perform BLS period search
6. CNN inference
7. Ensemble validation
8. Confidence calibration
9. Explain prediction
10. Display interactive results

---

## Signal Classes

| Class | Description |
|-------|-------------|
| 🪐 PLANET_TRANSIT | Flat-bottomed periodic flux dip from exoplanet |
| ⭐ ECLIPSING_BINARY | Deep V-shaped eclipses, secondary detected at phase 0.5 |
| 〰️ STELLAR_VARIABILITY | Smooth sinusoidal pulsation, no flat bottom |
| 🌑 STARSPOT_ACTIVITY | Quasi-periodic rotational modulation, evolving amplitude |
| 📡 INSTRUMENT_NOISE | No real periodic signal; dominated by systematics |

---

| Metric | Value |
|-------|------:|
| Accuracy | **85.6%** |
| Macro F1 | **0.857** |
| Parameters | **353K** |
| Classes | **5** |

---
## Model Performance (test set, n=750)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Planet Transit | 0.843 | 0.713 | 0.773 |
| Eclipsing Binary | 0.987 | 0.980 | 0.983 |
| Stellar Variability | 0.770 | 0.847 | 0.806 |
| Starspot Activity | 0.962 | 0.853 | 0.905 |
| Instrument Noise | 0.756 | 0.887 | 0.816 |
| **Macro Average** | | | **0.857** |

**Overall test accuracy: 85.6%**

## Data Formats Supported

Upload any CSV/TSV with columns: `time, flux, [flux_err]`
- Kepler PDC-SAP light curves
- TESS SPOC light curves  
- TESS TIC CTL exports
- Generic photometric time series

## Real NASA Targets (Validation Set)

9 curated real Kepler KOIs from the NASA Exoplanet Archive:
- **Confirmed planets**: Kepler-10 b, Kepler-90 b/i, Kepler-62 e, Kepler-22 b, Kepler-100 c
- **False positives**: KOI-189.02, KOI-686.02 (eclipsing low-mass stars)  
- **Candidate**: KOI-2700.01 (disintegrating rock candidate)

Live NASA Exoplanet Archive fetch is available when `prefer_live=True` in `real_data_loader.py`.

## Directory Structure

```
exoplanet-platform/
├── ml/                     # ML pipeline (Python)
│   ├── synthetic_generator.py  # Physics-based light curve injection
│   ├── preprocessing.py        # Detrending, cleaning, phase-folding
│   ├── features.py             # BLS search + 10 vetting features
│   ├── dataset_builder.py      # Training dataset construction
│   ├── model_cnn.py            # 1D-CNN + temperature scaler
│   ├── ensemble.py             # Rule-based vetting layer
│   ├── explain.py              # Integrated gradients saliency
│   ├── real_data_loader.py     # NASA Exoplanet Archive integration
│   ├── train.py                # Reproducible training script
│   └── checkpoints/            # Versioned model artifacts
├── backend/                # FastAPI REST API
│   ├── main.py             # All endpoints
│   ├── pipeline/           # DetectionPipeline orchestrator
│   ├── models/             # Pydantic schemas
│   └── storage/            # SQLite DB layer
├── frontend/               # React + TypeScript dashboard
│   └── src/
│       ├── pages/          # Dashboard, Analyze, Results, History, ...
│       └── components/     # LightCurveViewer, PhaseFoldChart, ...
└── start.sh                # One-command launcher
```

#  Supported Data

Accepted formats:

- Kepler PDC-SAP
- TESS SPOC
- Generic CSV
- TSV

Columns:

```text
time
flux
flux_err (optional)
```

---

#  NASA Validation

Validated on curated Kepler Objects of Interest including:

- Kepler-10 b
- Kepler-22 b
- Kepler-62 e
- Kepler-90 b
- Kepler-90 i
- Kepler-100 c
- KOI-189.02
- KOI-686.02
- KOI-2700.01

---

#  REST API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /upload | POST | Upload light curve |
| /analyze | POST | Run inference |
| /history | GET | Prediction history |
| /candidates | GET | Candidate list |
| /real-targets | GET | NASA targets |
| /health | GET | Health check |

---

#  Tech Stack

**Frontend**

- React
- TypeScript
- Tailwind CSS
- Vite

**Backend**

- FastAPI
- SQLite
- Pydantic
- Uvicorn

**Machine Learning**

- PyTorch
- NumPy
- Pandas
- SciPy
- Scikit-learn

---

#  Requirements

- Python 3.11+
- Node.js 20+
- npm 10+

##  Highlights

- End-to-end AI-powered exoplanet detection platform
- 17,500+ synthetic training light curves
- 353K-parameter 1D CNN
- 85.6% test accuracy
- Explainable AI with Integrated Gradients
- FastAPI REST backend
- React + TypeScript dashboard
- NASA Kepler validation

#  Roadmap

- Docker support
- CI/CD
- Cloud deployment
- Transformer models
- Multi-planet detection
- User authentication
- Batch inference
- Kubernetes

---

#  Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit changes.
4. Push your branch.
5. Open a Pull Request.

---

## Future Work

Future improvements include support for transformer-based architectures, cloud-native deployment, automated retraining pipelines, multi-planet detection, and integration with live astronomical observation streams.

---

#  License

MIT License.

---

#  Author

**Asad Khan**

- GitHub: https://github.com/meranaamkhann
- LinkedIn: https://www.linkedin.com/in/meranaamkhann/

---

If you found this project useful, consider starring the repository.