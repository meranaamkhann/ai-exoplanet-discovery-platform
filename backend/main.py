"""
main.py
========
FastAPI backend for the AI-Powered Exoplanet Discovery and Analysis Platform.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations
import sys
import os
import io
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, os.path.abspath(ML_DIR))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import csv as csv_module

from models.schemas import (
    UploadResponse, LightCurvePreview, TimeSeriesPoint, AnalyzeRequest, AnalyzeResponse,
    CandidateResult, EvidenceItemSchema, TransitFeaturesSchema, HistoryResponse, HistoryItem,
    CandidateStatusUpdate, RealTargetInfo, ModelInfo, SummaryStats,
)
from storage import db
from pipeline.pipeline import DetectionPipeline, registry
from pipeline.report_generator import generate_analysis_report_pdf

from real_data_loader import REAL_KOI_CACHE, synthesize_lightcurve_for_koi, list_real_targets
from synthetic_generator import SyntheticLightCurveGenerator

def _json_safe(obj):
    """Recursively replace NaN/Infinity with None so standard JSON encoding doesn't choke.
    Some ground-truth metadata fields (e.g. planet_radius_re for non-planet classes) are
    intentionally NaN in the dataclass to mean 'not applicable' — None is the correct JSON
    representation of that."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """Default response class that tolerates NaN/Infinity in returned data by converting them
    to null, since strict JSON (unlike Python's json module default) doesn't allow them and
    several pipeline outputs (e.g. 'not applicable' physical parameters) are legitimately NaN."""

    def render(self, content) -> bytes:
        return json.dumps(_json_safe(content), allow_nan=False).encode("utf-8")


app = FastAPI(
    title="ExoNova — AI-Powered Exoplanet Discovery Platform",
    description=(
        "Production-grade pipeline for detecting and classifying exoplanet transit signals "
        "in noisy stellar light curves. Supports Kepler/TESS formats, provides calibrated "
        "confidence scores, physical parameter estimation, and explainable AI analysis."
    ),
    version="1.0.0",
    default_response_class=SafeJSONResponse,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing(request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback, logging
    logging.error(f"Unhandled exception on {request.url}: {exc}\n{traceback.format_exc()}")
    return SafeJSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {str(exc)[:200]}"},
    )

pipeline: DetectionPipeline | None = None


@app.on_event("startup")
def startup():
    db.init_db()
    global pipeline
    try:
        registry.load_latest()
        print(f"[startup] Loaded model run_id={registry.run_id}")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: {e}")
    pipeline = DetectionPipeline(registry)


# =================================================================== HEALTH
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": registry.is_loaded,
        "model_run_id": registry.run_id,
    }


@app.get("/api/model-info", response_model=ModelInfo)
def model_info():
    if not registry.is_loaded:
        raise HTTPException(503, "No model loaded.")
    m = registry.metrics
    return ModelInfo(
        run_id=registry.run_id,
        test_accuracy=m.get("test_accuracy", 0.0),
        macro_f1=m.get("macro_f1", 0.0),
        n_train=m.get("n_train", 0),
        n_val=m.get("n_val", 0),
        n_test=m.get("n_test", 0),
        calibration_temperature=m.get("calibration_temperature", 1.0),
        class_names=m.get("class_names", []),
    )


# =================================================================== UPLOAD
def _parse_lightcurve_file(filename: str, content: bytes):
    """Supports CSV/TSV (time,flux[,flux_err] columns, header optional) and
    simple two/three-column whitespace-delimited text — covers the common
    Kepler/TESS exported formats plus generic research light curves."""
    if not content or not content.strip():
        raise ValueError("File is empty.")

    text = content.decode("utf-8", errors="replace").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("File contains no readable lines.")

    first = lines[0]
    delimiter = "\t" if "\t" in first else ","
    if delimiter not in first and "," not in first:
        delimiter = None  # whitespace-separated

    rows = []
    if delimiter:
        reader = csv_module.reader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                vals = [float(x) for x in row[:3]]
                rows.append(vals)
            except ValueError:
                continue  # header row or comment line
    else:
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                vals = [float(x) for x in parts[:3]]
                rows.append(vals)
            except ValueError:
                continue

    if len(rows) < 10:
        raise ValueError(
            f"Only {len(rows)} parseable data rows found (minimum 10 required). "
            "Expected columns: time, flux, [flux_err]. Header rows are skipped automatically."
        )

    arr = np.array(rows)
    time = arr[:, 0]
    flux = arr[:, 1]
    flux_err = arr[:, 2] if arr.shape[1] >= 3 else None
    return time, flux, flux_err


@app.post("/api/upload", response_model=UploadResponse)
async def upload_lightcurve(file: UploadFile = File(...)):
    content = await file.read()
    try:
        time, flux, flux_err = _parse_lightcurve_file(file.filename, content)
    except ValueError as e:
        msg = str(e)
        # 400 = bad request (empty/totally malformed), 422 = parseable but can't process data
        status = 400 if "empty" in msg.lower() or "no readable" in msg.lower() else 422
        raise HTTPException(status, msg)

    warnings = []
    if len(time) < 200:
        warnings.append("Fewer than 200 data points — period/transit detection accuracy will be limited.")

    try:
        cleaned = pipeline.clean(time, flux, flux_err)
    except ValueError as e:
        raise HTTPException(422, f"Data quality issue: {e}")

    baseline_days = float(cleaned.time[-1] - cleaned.time[0])
    cadence_minutes = float(np.median(np.diff(cleaned.time)) * 24 * 60) if len(cleaned.time) > 1 else 0.0

    raw_payload = {"time": time.tolist(), "flux": (flux / np.nanmedian(flux)).tolist()}
    cleaned_payload = {
        "time": cleaned.time.tolist(), "flux": cleaned.flux.tolist(),
        "trend": cleaned.trend.tolist(), "flux_err": cleaned.flux_err.tolist(),
    }

    dataset_id = db.save_dataset(
        name=file.filename, source="upload", n_points_raw=len(time),
        n_points_cleaned=cleaned.n_output_points, noise_ppm=cleaned.noise_ppm,
        baseline_days=baseline_days, raw_payload=raw_payload, cleaned_payload=cleaned_payload,
    )

    if cleaned.gap_fraction > 0.3:
        warnings.append(f"High data-gap fraction ({cleaned.gap_fraction*100:.0f}%) — some transit events may be missed.")

    return UploadResponse(
        dataset_id=dataset_id, n_points_raw=len(time), n_points_after_cleaning=cleaned.n_output_points,
        n_removed_outliers=cleaned.n_removed_outliers, gap_fraction=cleaned.gap_fraction,
        noise_ppm=cleaned.noise_ppm, baseline_days=baseline_days, cadence_minutes=cadence_minutes,
        warnings=warnings,
    )


@app.get("/api/datasets/{dataset_id}/preview", response_model=LightCurvePreview)
def dataset_preview(dataset_id: str, max_points: int = Query(default=3000, le=10000)):
    ds = db.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found.")
    raw = ds["raw_payload"]
    cleaned = ds["cleaned_payload"]

    def downsample(t, f, n):
        if len(t) <= n:
            return t, f
        idx = np.linspace(0, len(t) - 1, n).astype(int)
        return [t[i] for i in idx], [f[i] for i in idx]

    rt, rf = downsample(raw["time"], raw["flux"], max_points)
    ct, cf = downsample(cleaned["time"], cleaned["flux"], max_points)
    tt, tf = downsample(cleaned["time"], cleaned["trend"], max_points)

    return LightCurvePreview(
        dataset_id=dataset_id,
        raw=[TimeSeriesPoint(t=t, f=f) for t, f in zip(rt, rf)],
        cleaned=[TimeSeriesPoint(t=t, f=f) for t, f in zip(ct, cf)],
        trend=[TimeSeriesPoint(t=t, f=f) for t, f in zip(tt, tf)],
    )


# =================================================================== ANALYZE
def _candidate_to_schema(r: dict) -> CandidateResult:
    v = r["verdict"]
    f = r["features"]
    return CandidateResult(
        candidate_id=r["candidate_id"], rank=r["rank"],
        final_label=v.final_label, final_confidence=v.final_confidence,
        cnn_probabilities=v.cnn_probabilities, rule_adjusted_probabilities=v.rule_adjusted_probabilities,
        evidence=[EvidenceItemSchema(**vars(e)) for e in v.evidence],
        false_positive_flags=v.false_positive_flags, is_likely_false_positive=v.is_likely_false_positive,
        data_quality_warning=v.data_quality_warning,
        features=TransitFeaturesSchema(**{k: val for k, val in f.to_dict().items() if k in TransitFeaturesSchema.model_fields}),
        explanation=r["explanation"],
        phase_folded_global=r["global_view"].tolist(), phase_folded_local=r["local_view"].tolist(),
        global_saliency=r["saliency"].global_saliency.tolist(), local_saliency=r["saliency"].local_saliency.tolist(),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not registry.is_loaded:
        raise HTTPException(503, "Model not loaded. Train a model first (python ml/train.py).")
    ds = db.get_dataset(req.dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found.")

    cleaned_payload = ds["cleaned_payload"]
    time = np.array(cleaned_payload["time"])
    flux = np.array(cleaned_payload["flux"])
    flux_err = np.array(cleaned_payload["flux_err"])

    cleaned, results, obs_summary, elapsed = pipeline.analyze(
        time, flux, flux_err,
        min_period_days=req.min_period_days, max_period_frac=req.max_period_frac_of_baseline,
        stellar_radius_rsun=req.stellar_radius_rsun, n_top_candidates=req.n_top_candidates,
        fast_mode=req.fast_mode, target_name=ds["name"],
    )

    candidate_schemas = [_candidate_to_schema(r) for r in results]

    analysis_id = db.save_analysis(
        dataset_id=req.dataset_id, model_version=registry.run_id, processing_time_seconds=elapsed,
        observation_summary=obs_summary,
        candidates=[c.model_dump() for c in candidate_schemas],
    )

    return AnalyzeResponse(
        dataset_id=req.dataset_id, analysis_id=analysis_id, candidates=candidate_schemas,
        observation_summary=obs_summary, processing_time_seconds=elapsed, model_version=registry.run_id or "unknown",
    )


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    a = db.get_analysis(analysis_id)
    if a is None:
        raise HTTPException(404, "Analysis not found.")
    return a


@app.get("/api/analyses/{analysis_id}/report.pdf")
def download_report(analysis_id: str):
    a = db.get_analysis(analysis_id)
    if a is None:
        raise HTTPException(404, "Analysis not found.")
    ds = db.get_dataset(a["dataset_id"])
    dataset_name = ds["name"] if ds else "Unknown target"
    pdf_bytes = generate_analysis_report_pdf(a, dataset_name)
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="exoplanet_report_{analysis_id}.pdf"'},
    )


# =================================================================== HISTORY
@app.get("/api/history", response_model=HistoryResponse)
def history(limit: int = 50, offset: int = 0):
    items, total = db.list_history(limit=limit, offset=offset)
    return HistoryResponse(
        items=[HistoryItem(
            analysis_id=i["analysis_id"], dataset_id=i["dataset_id"], dataset_name=i["dataset_name"],
            created_at=i["created_at"], top_label=i["top_label"] or "UNKNOWN",
            top_confidence=i["top_confidence"] or 0.0, n_candidates=i["n_candidates"],
            is_likely_false_positive=bool(i["any_fp"]),
        ) for i in items],
        total=total,
    )


@app.get("/api/stats", response_model=SummaryStats)
def summary_stats():
    return db.get_summary_stats()


# ================================================================ CANDIDATES
@app.get("/api/candidates")
def list_candidates(status: str | None = None, label: str | None = None, limit: int = 200):
    return db.list_all_candidates(status=status, label=label, limit=limit)


@app.post("/api/candidates/status")
def update_candidate(update: CandidateStatusUpdate):
    if db.get_candidate(update.candidate_id) is None:
        raise HTTPException(404, "Candidate not found.")
    db.update_candidate_status(update.candidate_id, update.status, update.reviewer_notes)
    return {"ok": True}


# ============================================================== REAL TARGETS
@app.get("/api/real-targets", response_model=list[RealTargetInfo])
def real_targets():
    return list_real_targets()


@app.post("/api/real-targets/{kepoi_name}/load")
def load_real_target(kepoi_name: str, baseline_days: float = 90.0, seed: int = 0):
    record = next((r for r in REAL_KOI_CACHE if r.kepoi_name == kepoi_name), None)
    if record is None:
        raise HTTPException(404, "Real target not found in curated cache.")
    sample = synthesize_lightcurve_for_koi(record, baseline_days=baseline_days, noise_realization_seed=seed)

    cleaned = pipeline.clean(sample.time, sample.flux, sample.flux_err)
    baseline = float(cleaned.time[-1] - cleaned.time[0])
    raw_payload = {"time": sample.time.tolist(), "flux": sample.flux.tolist()}
    cleaned_payload = {
        "time": cleaned.time.tolist(), "flux": cleaned.flux.tolist(),
        "trend": cleaned.trend.tolist(), "flux_err": cleaned.flux_err.tolist(),
    }
    dataset_id = db.save_dataset(
        name=f"{record.kepler_name or record.kepoi_name} (real KOI parameters)", source="real_koi",
        n_points_raw=len(sample.time), n_points_cleaned=cleaned.n_output_points,
        noise_ppm=cleaned.noise_ppm, baseline_days=baseline,
        raw_payload=raw_payload, cleaned_payload=cleaned_payload,
    )
    return {"dataset_id": dataset_id, "record": {
        "kepoi_name": record.kepoi_name, "kepler_name": record.kepler_name,
        "disposition": record.disposition, "disposition_reason": record.disposition_reason,
        "true_period_days": record.period_days, "true_depth_ppm": record.depth_ppm,
    }}


# ============================================================= DEMO / SAMPLE
@app.post("/api/demo/generate")
def generate_demo_sample(kind: str = Query(default="planet", pattern="^(planet|eclipsing_binary|stellar_variability|starspot|noise)$"),
                          seed: int = 0, baseline_days: float = 90.0):
    rng = np.random.default_rng(seed)
    gen = SyntheticLightCurveGenerator(rng=rng)
    maker = {
        "planet": gen.make_planet_transit, "eclipsing_binary": gen.make_eclipsing_binary,
        "stellar_variability": gen.make_stellar_variability, "starspot": gen.make_starspot_activity,
        "noise": gen.make_instrument_noise,
    }[kind]
    sample = maker(baseline_days=baseline_days)

    cleaned = pipeline.clean(sample.time, sample.flux, sample.flux_err)
    baseline = float(cleaned.time[-1] - cleaned.time[0])
    raw_payload = {"time": sample.time.tolist(), "flux": sample.flux.tolist()}
    cleaned_payload = {
        "time": cleaned.time.tolist(), "flux": cleaned.flux.tolist(),
        "trend": cleaned.trend.tolist(), "flux_err": cleaned.flux_err.tolist(),
    }
    dataset_id = db.save_dataset(
        name=f"Synthetic demo: {sample.label_name} (seed={seed})", source="synthetic_demo",
        n_points_raw=len(sample.time), n_points_cleaned=cleaned.n_output_points,
        noise_ppm=cleaned.noise_ppm, baseline_days=baseline,
        raw_payload=raw_payload, cleaned_payload=cleaned_payload,
    )
    return {"dataset_id": dataset_id, "ground_truth": _json_safe(sample.to_meta_dict())}


# ============================================================= LIVE MAST FETCH
@app.get("/api/fetch-tess/{tic_id}")
def fetch_tess_lightcurve(tic_id: int, sector: int | None = None, exptime: int | None = None):
    """
    Fetch a real TESS SPOC light curve from MAST via lightkurve.
    Requires open internet access to archive.stsci.edu.
    Falls back with a clear 503 when offline — use /api/real-targets as offline alternative.
    """
    try:
        import lightkurve as lk
        import warnings

        search_kw: dict = {}
        if sector is not None:
            search_kw["sector"] = sector
        if exptime is not None:
            search_kw["exptime"] = exptime

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", **search_kw)

        if len(results) == 0:
            raise HTTPException(404, f"No TESS SPOC light curve found for TIC {tic_id}.")

        lc = results[-1].download()
        if lc is None:
            raise HTTPException(502, "MAST download returned empty data — try again.")

        lc = lc.remove_nans().remove_outliers(sigma=5)
        time_arr = lc.time.value.astype(float)
        flux_arr = lc.flux.value.astype(float)
        flux_err_arr = lc.flux_err.value.astype(float) if hasattr(lc, "flux_err") and lc.flux_err is not None else None

        med = float(np.nanmedian(flux_arr))
        if med != 0 and np.isfinite(med):
            flux_arr = flux_arr / med
            if flux_err_arr is not None:
                flux_err_arr = flux_err_arr / med

        cleaned = pipeline.clean(time_arr, flux_arr, flux_err_arr)
        baseline = float(cleaned.time[-1] - cleaned.time[0])
        raw_payload = {"time": time_arr.tolist(), "flux": flux_arr.tolist()}
        cleaned_payload = {
            "time": cleaned.time.tolist(), "flux": cleaned.flux.tolist(),
            "trend": cleaned.trend.tolist(), "flux_err": cleaned.flux_err.tolist(),
        }
        dataset_id = db.save_dataset(
            name=f"TIC {tic_id} (TESS SPOC, live MAST)", source="tess_live",
            n_points_raw=len(time_arr), n_points_cleaned=cleaned.n_output_points,
            noise_ppm=cleaned.noise_ppm, baseline_days=baseline,
            raw_payload=raw_payload, cleaned_payload=cleaned_payload,
        )
        return {
            "dataset_id": dataset_id, "tic_id": tic_id, "n_points": len(time_arr),
            "baseline_days": round(baseline, 2), "noise_ppm": round(cleaned.noise_ppm, 1),
            "source": "MAST TESS SPOC (live download)",
        }

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(503, "lightkurve not installed. Run: pip install lightkurve")
    except Exception as e:
        err = str(e)
        if any(k in err.lower() for k in ("connection", "timeout", "network", "refused")):
            raise HTTPException(503,
                "Cannot reach MAST archive (archive.stsci.edu). Open internet access is required. "
                "Use /api/real-targets as a cached offline alternative.")
        raise HTTPException(500, f"TESS fetch failed: {err[:200]}")


@app.get("/api/fetch-kepler/{kic_id}")
def fetch_kepler_lightcurve(kic_id: int, quarter: int | None = None):
    """Fetch a real Kepler long-cadence light curve from MAST for a given KIC ID."""
    try:
        import lightkurve as lk
        import warnings

        search_kw: dict = {"exptime": 1800}
        if quarter is not None:
            search_kw["quarter"] = quarter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lk.search_lightcurve(f"KIC {kic_id}", mission="Kepler", author="Kepler", **search_kw)

        if len(results) == 0:
            raise HTTPException(404, f"No Kepler light curve found for KIC {kic_id}.")

        lc = results[-1].download()
        if lc is None:
            raise HTTPException(502, "MAST download returned empty data.")

        lc = lc.remove_nans().remove_outliers(sigma=5).normalize()
        time_arr = lc.time.value.astype(float)
        flux_arr = lc.flux.value.astype(float)
        flux_err_arr = lc.flux_err.value.astype(float) if hasattr(lc, "flux_err") and lc.flux_err is not None else None

        cleaned = pipeline.clean(time_arr, flux_arr, flux_err_arr)
        baseline = float(cleaned.time[-1] - cleaned.time[0])
        raw_payload = {"time": time_arr.tolist(), "flux": flux_arr.tolist()}
        cleaned_payload = {
            "time": cleaned.time.tolist(), "flux": cleaned.flux.tolist(),
            "trend": cleaned.trend.tolist(), "flux_err": cleaned.flux_err.tolist(),
        }
        dataset_id = db.save_dataset(
            name=f"KIC {kic_id} (Kepler, live MAST)", source="kepler_live",
            n_points_raw=len(time_arr), n_points_cleaned=cleaned.n_output_points,
            noise_ppm=cleaned.noise_ppm, baseline_days=baseline,
            raw_payload=raw_payload, cleaned_payload=cleaned_payload,
        )
        return {
            "dataset_id": dataset_id, "kic_id": kic_id, "n_points": len(time_arr),
            "baseline_days": round(baseline, 2), "noise_ppm": round(cleaned.noise_ppm, 1),
            "source": "MAST Kepler (live download)",
        }

    except HTTPException:
        raise
    except Exception as e:
        err = str(e)
        if any(k in err.lower() for k in ("connection", "timeout", "network", "refused")):
            raise HTTPException(503, "Cannot reach MAST archive. Internet access is required.")
        raise HTTPException(500, f"Kepler fetch failed: {err[:200]}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
