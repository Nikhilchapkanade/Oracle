"""
ORACLE — Database Layer
SQLAlchemy-based persistence for sequences, mutations, and analysis results.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OracleDatabase:
    """SQLite database manager for ORACLE data persistence."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from config.settings import config

            db_path = str(config.db.url).replace("sqlite:///", "")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sequences (
                id TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                lineage TEXT NOT NULL,
                collection_date TEXT NOT NULL,
                country TEXT NOT NULL,
                continent TEXT DEFAULT '',
                host TEXT DEFAULT 'Human',
                clade TEXT DEFAULT '',
                quality_score REAL DEFAULT 1.0,
                mutations_json TEXT DEFAULT '[]',
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS lineages (
                name TEXT PRIMARY KEY,
                parent_lineage TEXT,
                who_label TEXT,
                risk_level TEXT DEFAULT 'Not classified',
                defining_mutations_json TEXT DEFAULT '[]',
                first_detected TEXT,
                first_detected_country TEXT DEFAULT '',
                sequence_count INTEGER DEFAULT 0,
                growth_rate REAL DEFAULT 0.0,
                relative_fitness REAL DEFAULT 1.0,
                immune_escape_score REAL DEFAULT 0.0,
                ace2_binding_score REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS mutation_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position INTEGER NOT NULL,
                current_aa TEXT NOT NULL,
                predicted_aa TEXT NOT NULL,
                probability REAL NOT NULL,
                expected_timeframe_days INTEGER DEFAULT 90,
                fitness_impact REAL DEFAULT 0.0,
                escape_impact REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS escape_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variant_id TEXT NOT NULL,
                lineage TEXT NOT NULL,
                overall_escape REAL NOT NULL,
                antibody_scores_json TEXT DEFAULT '{}',
                ace2_binding_change REAL DEFAULT 0.0,
                convalescent_escape REAL DEFAULT 0.0,
                vaccine_escape REAL DEFAULT 0.0,
                risk_assessment TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS vaccine_candidates (
                id TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                target_mutations_json TEXT DEFAULT '[]',
                immunogenicity_score REAL DEFAULT 0.0,
                breadth_score REAL DEFAULT 0.0,
                stability_score REAL DEFAULT 0.0,
                escape_resistance REAL DEFAULT 0.0,
                overall_score REAL DEFAULT 0.0,
                design_rationale TEXT DEFAULT '',
                generation_method TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_sequences INTEGER DEFAULT 0,
                novel_mutations INTEGER DEFAULT 0,
                predictions_generated INTEGER DEFAULT 0,
                vaccines_designed INTEGER DEFAULT 0,
                agent_durations_json TEXT DEFAULT '{}',
                errors_json TEXT DEFAULT '[]',
                status TEXT DEFAULT 'running',
                report_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT DEFAULT '',
                data_json TEXT DEFAULT '{}',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_sequences_lineage ON sequences(lineage);
            CREATE INDEX IF NOT EXISTS idx_sequences_country ON sequences(country);
            CREATE INDEX IF NOT EXISTS idx_sequences_date ON sequences(collection_date);
            CREATE INDEX IF NOT EXISTS idx_agent_logs_pipeline ON agent_logs(pipeline_id);
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    # ─────────────── Sequence Operations ───────────────

    def insert_sequence(self, seq) -> None:
        """Insert a ViralSequence into the database."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO sequences
               (id, sequence, lineage, collection_date, country, continent, host, clade,
                quality_score, mutations_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                seq.id,
                seq.sequence,
                seq.lineage,
                seq.collection_date.isoformat(),
                seq.country,
                seq.continent,
                seq.host,
                seq.clade,
                seq.quality_score,
                json.dumps([m.notation for m in seq.mutations]),
                json.dumps(seq.metadata),
            ),
        )
        conn.commit()
        conn.close()

    def insert_sequences_batch(self, sequences: list) -> int:
        """Batch insert sequences. Returns count inserted."""
        conn = self._get_conn()
        count = 0
        for seq in sequences:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO sequences
                       (id, sequence, lineage, collection_date, country, continent, host, clade,
                        quality_score, mutations_json, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        seq.id,
                        seq.sequence,
                        seq.lineage,
                        seq.collection_date.isoformat(),
                        seq.country,
                        seq.continent,
                        seq.host,
                        seq.clade,
                        seq.quality_score,
                        json.dumps([m.notation for m in seq.mutations]),
                        json.dumps(seq.metadata),
                    ),
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to insert sequence {seq.id}: {e}")
        conn.commit()
        conn.close()
        return count

    def get_sequences(
        self, lineage: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """Retrieve sequences, optionally filtered by lineage."""
        conn = self._get_conn()
        if lineage:
            rows = conn.execute(
                "SELECT * FROM sequences WHERE lineage = ? ORDER BY collection_date DESC LIMIT ?",
                (lineage, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sequences ORDER BY collection_date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_sequence_count(self) -> int:
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
        conn.close()
        return count

    # ─────────────── Lineage Operations ───────────────

    def insert_lineage(self, lineage) -> None:
        """Insert/update a Lineage."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO lineages
               (name, parent_lineage, who_label, risk_level, defining_mutations_json,
                first_detected, first_detected_country, sequence_count, growth_rate,
                relative_fitness, immune_escape_score, ace2_binding_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                lineage.name,
                lineage.parent_lineage,
                lineage.who_label,
                lineage.risk_level.value,
                json.dumps(lineage.defining_mutations),
                lineage.first_detected.isoformat() if lineage.first_detected else None,
                lineage.first_detected_country,
                lineage.sequence_count,
                lineage.growth_rate,
                lineage.relative_fitness,
                lineage.immune_escape_score,
                lineage.ace2_binding_score,
            ),
        )
        conn.commit()
        conn.close()

    def get_lineages(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM lineages ORDER BY sequence_count DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ─────────────── Pipeline Operations ───────────────

    def log_pipeline_run(
        self, run_id: str, start_time: datetime, status: str = "running"
    ):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO pipeline_runs (id, start_time, status) VALUES (?, ?, ?)",
            (run_id, start_time.isoformat(), status),
        )
        conn.commit()
        conn.close()

    def update_pipeline_run(self, run_id: str, **kwargs):
        conn = self._get_conn()
        sets = []
        vals = []
        for k, v in kwargs.items():
            if isinstance(v, (dict, list)):
                sets.append(f"{k}_json = ?")
                vals.append(json.dumps(v))
            elif isinstance(v, datetime):
                sets.append(f"{k} = ?")
                vals.append(v.isoformat())
            else:
                sets.append(f"{k} = ?")
                vals.append(v)
        vals.append(run_id)
        conn.execute(f"UPDATE pipeline_runs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        conn.close()

    def log_agent_activity(
        self,
        pipeline_id: str,
        agent_name: str,
        status: str,
        message: str = "",
        data: dict = None,
    ):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO agent_logs (pipeline_id, agent_name, status, message, data_json) VALUES (?, ?, ?, ?, ?)",
            (pipeline_id, agent_name, status, message, json.dumps(data or {})),
        )
        conn.commit()
        conn.close()

    def get_pipeline_runs(self, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_agent_logs(self, pipeline_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM agent_logs WHERE pipeline_id = ? ORDER BY timestamp",
            (pipeline_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ─────────────── Prediction & Vaccine Operations ───────────────

    def insert_prediction(self, pred) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO mutation_predictions
               (position, current_aa, predicted_aa, probability, expected_timeframe_days,
                fitness_impact, escape_impact)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                pred.position,
                pred.current_aa,
                pred.predicted_aa,
                pred.probability,
                pred.expected_timeframe_days,
                pred.fitness_impact,
                pred.escape_impact,
            ),
        )
        conn.commit()
        conn.close()

    def insert_vaccine_candidate(self, vc) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO vaccine_candidates
               (id, sequence, target_mutations_json, immunogenicity_score, breadth_score,
                stability_score, escape_resistance, overall_score, design_rationale, generation_method)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                vc.id,
                vc.sequence,
                json.dumps(vc.target_mutations),
                vc.immunogenicity_score,
                vc.breadth_score,
                vc.stability_score,
                vc.escape_resistance,
                vc.overall_score,
                vc.design_rationale,
                vc.generation_method,
            ),
        )
        conn.commit()
        conn.close()

    def get_vaccine_candidates(self, limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM vaccine_candidates ORDER BY overall_score DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_predictions(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM mutation_predictions ORDER BY probability DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_escape_scores(self, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM escape_scores ORDER BY overall_escape DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def insert_escape_score(self, es) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO escape_scores
               (variant_id, lineage, overall_escape, antibody_scores_json,
                ace2_binding_change, convalescent_escape, vaccine_escape, risk_assessment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                es.variant_id,
                es.lineage,
                es.overall_escape,
                json.dumps(es.antibody_class_scores),
                es.ace2_binding_change,
                es.convalescent_escape,
                es.vaccine_escape,
                es.risk_assessment,
            ),
        )
        conn.commit()
        conn.close()
