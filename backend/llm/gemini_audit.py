"""Gemini deep forensic audit service (Tier 2)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from backend.config import get_settings


class GeminiAuditResult(BaseModel):
    """Structured deep-audit output returned by Gemini."""

    verdict: Literal["FRAUD", "SUSPICIOUS", "NORMAL", "INCONCLUSIVE"]
    violated_clause: Optional[str] = Field(default=None, max_length=150)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=8, max_length=1200)
    cited_clause_text: Optional[str] = Field(default=None, max_length=4000)


class GeminiDeepAuditService:
    """Calls Gemini with contract clauses + transaction evidence."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = asyncio.Lock()
        self._minute_start = time.monotonic()
        self._calls_this_minute = 0
        self._day_start = datetime.now(timezone.utc).date()
        self._calls_today = 0

    async def audit_transaction(
        self,
        transaction_payload: Dict[str, Any],
        clause_hits: List[Dict[str, Any]],
    ) -> GeminiAuditResult:
        """Run deep forensic audit against retrieved contract clauses."""
        if not self._settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing; cannot run deep audit")

        await self._enforce_limits()

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._settings.GEMINI_MODEL}:generateContent?key={self._settings.GEMINI_API_KEY}"
        )

        compact_clauses = [
            {
                "contract_id": c.get("metadata", {}).get("contract_id"),
                "score": c.get("score"),
                "text": c.get("text", "")[: self._settings.GEMINI_MAX_CLAUSE_CHARS],
            }
            for c in clause_hits[: self._settings.GEMINI_MAX_CLAUSES]
        ]

        compact_tx = {
            "transaction_id": transaction_payload.get("transaction_id"),
            "vendor_id": transaction_payload.get("vendor_id"),
            "amount": transaction_payload.get("amount"),
            "date": transaction_payload.get("date"),
            "category": transaction_payload.get("category"),
            "description": transaction_payload.get("description"),
            "awarding_agency": transaction_payload.get("awarding_agency"),
            "ml_score": transaction_payload.get("ml_score"),
            "shap_summary": transaction_payload.get("shap_summary"),
        }

        prompt_payload = {
            "transaction": compact_tx,
            "clauses": compact_clauses,
        }
        prompt_json = json.dumps(prompt_payload, default=str, ensure_ascii=False)
        if len(prompt_json) > self._settings.GEMINI_MAX_INPUT_CHARS:
            prompt_json = prompt_json[: self._settings.GEMINI_MAX_INPUT_CHARS]

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "You are a forensic procurement auditor. Return JSON only with keys: "
                                "verdict (FRAUD|SUSPICIOUS|NORMAL|INCONCLUSIVE), violated_clause, confidence, rationale, cited_clause_text. "
                                "Use only provided evidence and clauses. If evidence is weak, return INCONCLUSIVE with lower confidence.\n"
                                f"Payload:\n{prompt_json}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self._settings.GEMINI_TEMPERATURE,
                "topP": self._settings.GEMINI_TOP_P,
                "maxOutputTokens": self._settings.GEMINI_MAX_OUTPUT_TOKENS,
                "responseMimeType": "application/json",
            },
        }

        timeout = httpx.Timeout(self._settings.GEMINI_TIMEOUT_SECONDS)
        last_error: Exception | None = None

        for attempt in range(1, self._settings.GEMINI_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=body)
                response.raise_for_status()
                data = response.json()

                text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text) if isinstance(text, str) else text
                return GeminiAuditResult.model_validate(parsed)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in (401, 403):
                    logger.error("Gemini auth failed (status={}): {}", status_code, exc)
                    raise RuntimeError(
                        "Gemini authentication failed (401/403). "
                        "Check GEMINI_API_KEY in .env and restart the API server."
                    ) from exc
                if status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 6.0
                    last_error = exc
                    if attempt < self._settings.GEMINI_MAX_RETRIES:
                        await asyncio.sleep(wait_seconds)
                    continue
                last_error = exc
                if attempt < self._settings.GEMINI_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** (attempt - 1), 6))
                continue
            except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt < self._settings.GEMINI_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** (attempt - 1), 6))
                continue

        logger.error("Gemini deep audit failed after retries: {}", last_error)
        raise RuntimeError(f"Gemini deep audit failed: {last_error}")

    async def _enforce_limits(self) -> None:
        """Simple in-process rate/quota guard (per minute + per day)."""
        async with self._lock:
            today = datetime.now(timezone.utc).date()
            if today != self._day_start:
                self._day_start = today
                self._calls_today = 0

            if self._calls_today >= self._settings.GEMINI_DAILY_QUOTA:
                raise RuntimeError("Gemini daily quota reached")

            now = time.monotonic()
            elapsed = now - self._minute_start
            if elapsed >= 60:
                self._minute_start = now
                self._calls_this_minute = 0

            if self._calls_this_minute >= self._settings.GEMINI_RATE_LIMIT_PER_MINUTE:
                sleep_for = max(0.0, 60 - elapsed)
                await asyncio.sleep(sleep_for)
                self._minute_start = time.monotonic()
                self._calls_this_minute = 0

            self._calls_this_minute += 1
            self._calls_today += 1
