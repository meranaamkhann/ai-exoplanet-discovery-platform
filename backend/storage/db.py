"""
db.py
======
Lightweight SQLite storage for detection history, candidate management, and
uploaded dataset metadata. Chosen over a heavier DB for hackathon-scope
simplicity, zero external dependencies, and trivial reproducibility (just a
file on disk).
"""

from __future__ import annotations
import sqlite3
import json
import os
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = os.environ.get("EXOPLANET_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "data", "platform.db"))


def _ensure_dir():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            name TEXT,
            source TEXT,                 -- 'upload' | 'real_koi' | 'synthetic_demo'
            n_points_raw INTEGER,
            n_points_cleaned INTEGER,
            noise_ppm REAL,
            baseline_days REAL,
            raw_payload TEXT,             -- JSON: time/flux/flux_err arrays
            cleaned_payload TEXT,         -- JSON: cleaned time/flux/trend arrays
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            dataset_id TEXT REFERENCES datasets(dataset_id),
            model_version TEXT,
            processing_time_seconds REAL,
            observation_summary TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            analysis_id TEXT REFERENCES analyses(analysis_id),
            rank INTEGER,
            final_label TEXT,
            final_confidence REAL,
            is_likely_false_positive INTEGER,
            status TEXT DEFAULT 'pending',  -- pending | confirmed | rejected | needs_review
            reviewer_notes TEXT,
            payload TEXT,                    -- JSON: full CandidateResult
            created_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_dataset ON analyses(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_candidates_analysis ON candidates(analysis_id);
        """)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------- datasets
def save_dataset(name: str, source: str, n_points_raw: int, n_points_cleaned: int,
                  noise_ppm: float, baseline_days: float, raw_payload: dict, cleaned_payload: dict) -> str:
    dataset_id = new_id("ds")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO datasets (dataset_id, name, source, n_points_raw, n_points_cleaned, "
            "noise_ppm, baseline_days, raw_payload, cleaned_payload, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (dataset_id, name, source, n_points_raw, n_points_cleaned, noise_ppm, baseline_days,
             json.dumps(raw_payload), json.dumps(cleaned_payload), now_iso()),
        )
    return dataset_id


def get_dataset(dataset_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["raw_payload"] = json.loads(d["raw_payload"])
        d["cleaned_payload"] = json.loads(d["cleaned_payload"])
        return d


# ----------------------------------------------------------------- analyses
def save_analysis(dataset_id: str, model_version: str, processing_time_seconds: float,
                   observation_summary: str, candidates: list[dict]) -> str:
    analysis_id = new_id("an")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO analyses (analysis_id, dataset_id, model_version, processing_time_seconds, "
            "observation_summary, created_at) VALUES (?,?,?,?,?,?)",
            (analysis_id, dataset_id, model_version, processing_time_seconds, observation_summary, now_iso()),
        )
        for c in candidates:
            conn.execute(
                "INSERT INTO candidates (candidate_id, analysis_id, rank, final_label, final_confidence, "
                "is_likely_false_positive, status, payload, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (c["candidate_id"], analysis_id, c["rank"], c["final_label"], c["final_confidence"],
                 int(c["is_likely_false_positive"]), "pending", json.dumps(c), now_iso()),
            )
    return analysis_id


def get_analysis(analysis_id: str) -> dict | None:
    with get_conn() as conn:
        arow = conn.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
        if arow is None:
            return None
        crows = conn.execute(
            "SELECT * FROM candidates WHERE analysis_id = ? ORDER BY rank ASC", (analysis_id,)
        ).fetchall()
        a = dict(arow)
        a["candidates"] = [json.loads(c["payload"]) | {"status": c["status"], "reviewer_notes": c["reviewer_notes"]} for c in crows]
        return a


def list_history(limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]
        rows = conn.execute(
            """
            SELECT a.analysis_id, a.dataset_id, d.name AS dataset_name, a.created_at,
                   (SELECT final_label FROM candidates WHERE analysis_id=a.analysis_id ORDER BY rank ASC LIMIT 1) AS top_label,
                   (SELECT final_confidence FROM candidates WHERE analysis_id=a.analysis_id ORDER BY rank ASC LIMIT 1) AS top_confidence,
                   (SELECT COUNT(*) FROM candidates WHERE analysis_id=a.analysis_id) AS n_candidates,
                   (SELECT MAX(is_likely_false_positive) FROM candidates WHERE analysis_id=a.analysis_id) AS any_fp
            FROM analyses a
            JOIN datasets d ON a.dataset_id = d.dataset_id
            ORDER BY a.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total


# ---------------------------------------------------------------- candidates
def update_candidate_status(candidate_id: str, status: str, reviewer_notes: str | None = None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE candidates SET status = ?, reviewer_notes = COALESCE(?, reviewer_notes) WHERE candidate_id = ?",
            (status, reviewer_notes, candidate_id),
        )


def get_candidate(candidate_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        return d


def get_summary_stats() -> dict:
    """Aggregate statistics across all analyses/candidates — powers the dashboard's
    statistical summary panel."""
    with get_conn() as conn:
        total_analyses = conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]
        total_candidates = conn.execute("SELECT COUNT(*) AS c FROM candidates").fetchone()["c"]
        by_label = conn.execute(
            "SELECT final_label, COUNT(*) AS c FROM candidates GROUP BY final_label"
        ).fetchall()
        by_status = conn.execute(
            "SELECT status, COUNT(*) AS c FROM candidates GROUP BY status"
        ).fetchall()
        avg_confidence_row = conn.execute(
            "SELECT AVG(final_confidence) AS a FROM candidates"
        ).fetchone()
        n_fp = conn.execute(
            "SELECT COUNT(*) AS c FROM candidates WHERE is_likely_false_positive = 1"
        ).fetchone()["c"]
        n_datasets = conn.execute("SELECT COUNT(*) AS c FROM datasets").fetchone()["c"]
        return {
            "total_analyses": total_analyses,
            "total_candidates": total_candidates,
            "total_datasets": n_datasets,
            "by_label": {r["final_label"]: r["c"] for r in by_label},
            "by_status": {r["status"]: r["c"] for r in by_status},
            "avg_confidence": avg_confidence_row["a"] or 0.0,
            "likely_false_positives": n_fp,
        }


def list_all_candidates(status: str | None = None, label: str | None = None, limit: int = 200) -> list[dict]:
    query = "SELECT * FROM candidates WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if label:
        query += " AND final_label = ?"
        params.append(label)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out
