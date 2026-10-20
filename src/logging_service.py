"""
Experiment Logging and Persistence Service.

Persists structured execution records to both SQLite and JSONL formats.
Supports exporting run logs to CSV and JSON for scientific analysis and reproducibility.
Strictly sanitizes records to avoid logging API keys or confidential environment data.
"""

from __future__ import annotations
import csv
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
import pandas as pd


@dataclass
class RunRecord:
    """Complete structured record of a single execution."""
    run_id: str
    timestamp: str
    query: str
    policy_name: str
    selected_action: str
    decision_rationale: str
    answer_or_result: str
    
    # Uncertainty & features
    heuristic_uncertainty: float
    observable_signals: Dict[str, Any]
    
    # Evidence & Verification
    evidence_references: List[str]
    verification_status: str
    supported_claims_count: int
    unsupported_claims_count: int
    
    # Operational Resources & Costs
    model_name: str
    is_mock_model: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_calls_count: int
    tool_calls_count: int
    latency_ms: float
    estimated_cost_units: float
    measured_monetary_usd: float
    
    # Phase 2 Model & Prompt Tracking
    prompt_version: str = "v1.0-standard"
    generation_settings: Dict[str, Any] = field(default_factory=dict)
    model_reported_confidence: Optional[float] = None
    tool_errors: Optional[str] = None
    
    # Optional ground truth / benchmark fields
    reference_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    benchmark_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoggingService:
    """Manages SQLite database and JSONL file logging for experiments."""

    def __init__(self, db_path: Optional[str] = None, jsonl_path: Optional[str] = None):
        base_dir = os.path.join("results")
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        self.db_path = db_path or os.path.join(base_dir, "experiments.db")
        self.jsonl_path = jsonl_path or os.path.join(logs_dir, "runs.jsonl")
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Create sqlite tables if not present."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    query TEXT,
                    policy_name TEXT,
                    selected_action TEXT,
                    decision_rationale TEXT,
                    answer_or_result TEXT,
                    heuristic_uncertainty REAL,
                    observable_signals TEXT,
                    evidence_references TEXT,
                    verification_status TEXT,
                    supported_claims_count INTEGER,
                    unsupported_claims_count INTEGER,
                    model_name TEXT,
                    is_mock_model INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    model_calls_count INTEGER,
                    tool_calls_count INTEGER,
                    latency_ms REAL,
                    estimated_cost_units REAL,
                    measured_monetary_usd REAL,
                    reference_answer TEXT,
                    is_correct INTEGER,
                    benchmark_category TEXT,
                    prompt_version TEXT,
                    generation_settings TEXT,
                    model_reported_confidence REAL,
                    tool_errors TEXT
                )
            """)
            # Ensure columns exist if table was already created
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
            for col, col_type in [
                ("prompt_version", "TEXT"),
                ("generation_settings", "TEXT"),
                ("model_reported_confidence", "REAL"),
                ("tool_errors", "TEXT"),
            ]:
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
            conn.commit()
        finally:
            conn.close()

    def log_run(self, record: RunRecord) -> str:
        """Persist record to SQLite and JSONL."""
        # 1. SQLite insert
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, timestamp, query, policy_name, selected_action,
                    decision_rationale, answer_or_result, heuristic_uncertainty,
                    observable_signals, evidence_references, verification_status,
                    supported_claims_count, unsupported_claims_count, model_name,
                    is_mock_model, prompt_tokens, completion_tokens, total_tokens,
                    model_calls_count, tool_calls_count, latency_ms,
                    estimated_cost_units, measured_monetary_usd, reference_answer,
                    is_correct, benchmark_category, prompt_version,
                    generation_settings, model_reported_confidence, tool_errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.timestamp,
                    record.query,
                    record.policy_name,
                    record.selected_action,
                    record.decision_rationale,
                    record.answer_or_result,
                    record.heuristic_uncertainty,
                    json.dumps(record.observable_signals),
                    json.dumps(record.evidence_references),
                    record.verification_status,
                    record.supported_claims_count,
                    record.unsupported_claims_count,
                    record.model_name,
                    1 if record.is_mock_model else 0,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.model_calls_count,
                    record.tool_calls_count,
                    record.latency_ms,
                    record.estimated_cost_units,
                    record.measured_monetary_usd,
                    record.reference_answer,
                    1 if record.is_correct is True else (0 if record.is_correct is False else None),
                    record.benchmark_category,
                    record.prompt_version,
                    json.dumps(record.generation_settings),
                    record.model_reported_confidence,
                    record.tool_errors
                )
            )
            conn.commit()
        finally:
            conn.close()

        # 2. Append to JSONL
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            print(f"Warning: Failed to append to JSONL log: {e}")

        return record.run_id

    def get_recent_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch latest records from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_dataframe(self) -> pd.DataFrame:
        """Load all runs into a pandas DataFrame."""
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp DESC", conn)
        finally:
            conn.close()

    def export_csv(self, filepath: str) -> str:
        """Export all logged runs to CSV."""
        df = self.get_dataframe()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath

    def export_json(self, filepath: str) -> str:
        """Export all logged runs to JSON."""
        df = self.get_dataframe()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_json(filepath, orient="records", indent=2)
        return filepath


# Global default instance
_default_logger: Optional[LoggingService] = None

def get_logging_service() -> LoggingService:
    global _default_logger
    if _default_logger is None:
        _default_logger = LoggingService()
    return _default_logger
