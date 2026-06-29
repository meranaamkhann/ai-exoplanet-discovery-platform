# ExoNova — AI-Powered Exoplanet Discovery Platform
### ISRO Hackathon 2026 · Problem Statement 7

A production-grade AI pipeline for detecting and classifying exoplanet transit signals in noisy stellar light curves.

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
# Backend:  http://localhost:8000
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Dashboard (port 5173)              │
│  Upload CSV │ Demo Generator │ Real KOI Targets │ Results   │
└────────────────────────┬────────────────────────────────────┘
                         │ /api/* (Vite proxy)
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend (port 8000)                │
│  /upload  /analyze  /history  /candidates  /real-targets    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Detection Pipeline                         │
│  1. Clean/detrend   →   preprocessing.py                    │
│  2. BLS search      →   features.py                         │
│  3. CNN classify    →   model_cnn.py (353K params)          │
│  4. Ensemble vet    →   ensemble.py                         │
│  5. Explain         →   explain.py (Integrated Gradients)   │
└─────────────────────────────────────────────────────────────┘
```

## Signal Classes

| Class | Description |
|-------|-------------|
| 🪐 PLANET_TRANSIT | Flat-bottomed periodic flux dip from exoplanet |
| ⭐ ECLIPSING_BINARY | Deep V-shaped eclipses, secondary detected at phase 0.5 |
| 〰️ STELLAR_VARIABILITY | Smooth sinusoidal pulsation, no flat bottom |
| 🌑 STARSPOT_ACTIVITY | Quasi-periodic rotational modulation, evolving amplitude |
| 📡 INSTRUMENT_NOISE | No real periodic signal; dominated by systematics |

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
