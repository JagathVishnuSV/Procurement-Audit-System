"""
backend/config.py
─────────────────────────────────────────────────────────────
Centralised, type-safe application configuration.

All values are read from environment variables (or .env file).
Use `get_settings()` everywhere—never import raw `os.getenv`.

Pattern: pydantic-settings BaseSettings with an @lru_cache
singleton so the .env file is parsed exactly once at startup.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env.

    Sections
    --------
    Application  – name, env, debug, log level
    PostgreSQL   – connection parameters + computed DSN properties
    Redis        – connection parameters + computed URL property
    Kafka        – bootstrap servers + topic names
    ML           – model path, thresholds
    LLM APIs     – Groq + Gemini keys (Sprint 4)
    Ingestion    – USAspending polling config
    Vector Store – FAISS index path + embedding model
    CORS         – allowed frontend origins
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",         # silently ignore unknown env vars
    )

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "Procurement Audit System"
    APP_ENV: str = "development"   # development | staging | production
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"        # DEBUG | INFO | WARNING | ERROR | CRITICAL

    # ── PostgreSQL ──────────────────────────────────────────────────────────────
    POSTGRES_USER: str = "procurement_user"
    POSTGRES_PASSWORD: str = "procurement_pass"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "procurement_audit"

    @property
    def database_url(self) -> str:
        """Synchronous SQLAlchemy DSN (used by Alembic and sync sessions)."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        """Async SQLAlchemy DSN (used by FastAPI request handlers)."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Redis ───────────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    @property
    def redis_url(self) -> str:
        """Full Redis connection URL."""
        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}"
                f":{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── Redpanda / Kafka ────────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:19092"

    # Topic names mirror the architecture document exactly
    KAFKA_TOPIC_RAW_TRANSACTIONS: str = "raw_transactions"
    KAFKA_TOPIC_ENRICHED_TRANSACTIONS: str = "enriched_transactions"
    KAFKA_TOPIC_ANOMALIES: str = "anomalies_topic"
    KAFKA_TOPIC_TRIAGE: str = "triage_topic"
    KAFKA_TOPIC_DEEP_AUDIT: str = "deep_audit_topic"

    @property
    def kafka_bootstrap_list(self) -> List[str]:
        """Kafka bootstrap servers as a Python list (aiokafka expects a list)."""
        return [s.strip() for s in self.KAFKA_BOOTSTRAP_SERVERS.split(",")]

    # ── Machine Learning ────────────────────────────────────────────────────────
    ML_ANOMALY_THRESHOLD: float = 0.1
    ML_MODEL_PATH: str = "models/isolation_forest.joblib"
    ML_CONTAMINATION: float = 0.05     # Expected fraction of outliers

    # ── LLM APIs (Sprint 4) ─────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.0
    GROQ_TOP_P: float = 0.85
    GROQ_MAX_TOKENS: int = 120
    GROQ_MAX_INPUT_CHARS: int = 3500
    GROQ_TIMEOUT_SECONDS: float = 20.0
    GROQ_MAX_RETRIES: int = 3
    GROQ_RATE_LIMIT_PER_MINUTE: int = 12
    GROQ_DAILY_QUOTA: int = 400
    GROQ_SOFT_ESCALATION_ML_SCORE: float = 0.60
    GROQ_HARD_ESCALATION_ML_SCORE: float = 0.80
    GROQ_MIN_SIGNAL_HITS_FOR_SOFT_ESCALATION: int = 2

    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.0
    GEMINI_TOP_P: float = 0.8
    GEMINI_MAX_OUTPUT_TOKENS: int = 220
    GEMINI_MAX_CLAUSES: int = 2
    GEMINI_MAX_CLAUSE_CHARS: int = 850
    GEMINI_MAX_INPUT_CHARS: int = 5000
    GEMINI_TIMEOUT_SECONDS: float = 30.0
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RATE_LIMIT_PER_MINUTE: int = 6
    GEMINI_DAILY_QUOTA: int = 200

    # ── Data Ingestion ──────────────────────────────────────────────────────────
    USASPENDING_BASE_URL: str = "https://api.usaspending.gov"
    USASPENDING_PULL_INTERVAL_SECONDS: int = 300   # 5-minute polling cycle
    USASPENDING_MAX_RECORDS_PER_PULL: int = 100
    CPPP_BASE_URL: str = "https://eprocure.gov.in/eprocure/app?page=FrontEndLatestActiveTenders&service=page"
    CPPP_PULL_INTERVAL_SECONDS: int = 600
    CPPP_MAX_RECORDS_PER_PULL: int = 100
    OCDS_BASE_URL: str = "https://data.open-contracting.org/en/publication/128/download?name=2026.jsonl.gz"
    OCDS_PULL_INTERVAL_SECONDS: int = 600
    OCDS_MAX_RECORDS_PER_PULL: int = 100
    INGESTION_POLL_INTERVAL_SECONDS: int = 300
    INGESTION_LOOP_ENABLED: bool = True
    INGESTION_LIMIT_PER_SOURCE: int = 100

    KAFKA_CONSUMER_ENABLED: bool = True
    KAFKA_CONSUMER_GROUP: str = "procurement_ingestion_workers"
    KAFKA_CONSUMER_AUTO_OFFSET_RESET: str = "latest"
    KAFKA_CONSUMER_TRIGGER_ORCHESTRATION: bool = True

    # ── Automated Orchestration (Coverage Uplift) ─────────────────────────────
    ORCH_ENABLED: bool = True
    ORCH_INTERVAL_SECONDS: int = 120
    ORCH_BATCH_SIZE: int = 200
    ORCH_CASE_POLICY: str = "ALL_SCORED"   # ALL_SCORED | ANOMALIES_ONLY
    ORCH_RUN_LLM: bool = True
    ORCH_MAX_LLM_PER_RUN: int = 200
    ORCH_LLM_MIN_SCORE: float = 0.0

    # ── Vector Store (Sprint 3) ─────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "data/faiss_index"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── CORS ─────────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> List[str]:
        """Allow CORS_ORIGINS to be supplied as a JSON string in .env."""
        if isinstance(value, str):
            # Handles: '["http://localhost:3000"]' or 'http://localhost:3000'
            stripped = value.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [origin.strip() for origin in stripped.split(",")]
        return value  # type: ignore[return-value]

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{value}'")
        return upper

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Warn (but do not block) if LLM keys are missing in non-dev envs."""
        if self.APP_ENV == "production":
            if not self.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY must be set in production")
            if not self.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY must be set in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    The @lru_cache ensures the .env file is read exactly once.
    Call `get_settings.cache_clear()` in tests to reset the cache.
    """
    return Settings()
