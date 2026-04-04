"""Groq triage service (Tier 1) for anomaly escalation decisions."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

import httpx
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from backend.config import get_settings


class GroqTriageResult(BaseModel):
    """Structured triage result returned by Groq."""

    escalate: bool = Field(..., description="Whether deep audit is required")
    reason: str = Field(..., min_length=5, max_length=800)
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]


class GroqTriageService:
    """Calls Groq Llama model with strict JSON output and low-cost defaults."""

    _URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = asyncio.Lock()
        self._minute_start = time.monotonic()
        self._calls_this_minute = 0
        self._day_start = datetime.now(timezone.utc).date()
        self._calls_today = 0

    async def triage_transaction(self, transaction_payload: Dict[str, Any]) -> GroqTriageResult:
        """Run first-stage risk triage. Returns strict structured output."""
        if not self._settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing; cannot run triage")
        if self._settings.GROQ_API_KEY.startswith("hf_"):
            raise RuntimeError(
                "GROQ_API_KEY appears to be a Hugging Face token (hf_*). "
                "Set a real Groq key (typically gsk_*) in .env and restart the API server."
            )

        await self._enforce_limits()

        headers = {
            "Authorization": f"Bearer {self._settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        compact_payload = {
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
        signal_info = self._extract_risk_signals(compact_payload)

        policy_text = (
            f"Escalation policy: escalate MUST be true if ml_score >= {self._settings.GROQ_HARD_ESCALATION_ML_SCORE}. "
            f"Escalate SHOULD be true if ml_score >= {self._settings.GROQ_SOFT_ESCALATION_ML_SCORE} "
            f"and signal_hits >= {self._settings.GROQ_MIN_SIGNAL_HITS_FOR_SOFT_ESCALATION}."
        )

        compact_payload["risk_signals"] = signal_info
        prompt_payload = json.dumps(compact_payload, default=str, ensure_ascii=False)
        if len(prompt_payload) > self._settings.GROQ_MAX_INPUT_CHARS:
            prompt_payload = prompt_payload[: self._settings.GROQ_MAX_INPUT_CHARS]

        body = {
            "model": self._settings.GROQ_MODEL,
            "temperature": self._settings.GROQ_TEMPERATURE,
            "max_tokens": self._settings.GROQ_MAX_TOKENS,
            "top_p": self._settings.GROQ_TOP_P,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a procurement fraud triage engine. "
                        "Return JSON only with keys: escalate (bool), reason (string), risk_level (HIGH|MEDIUM|LOW). "
                        "Escalate only when contract-level forensic review is necessary. "
                        "Be strict, deterministic, and cost-aware. "
                        f"{policy_text}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Assess this flagged procurement transaction and SHAP explanation for escalation. "
                        "Use the included risk_signals and apply the escalation policy exactly.\n"
                        f"Payload:\n{prompt_payload}"
                    ),
                },
            ],
        }

        timeout = httpx.Timeout(self._settings.GROQ_TIMEOUT_SECONDS)
        last_error: Exception | None = None

        for attempt in range(1, self._settings.GROQ_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self._URL, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content) if isinstance(content, str) else content
                result = GroqTriageResult.model_validate(parsed)
                return self._apply_escalation_guardrails(result, signal_info)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in (401, 403):
                    logger.error("Groq auth failed (status={}): {}", status_code, exc)
                    raise RuntimeError(
                        "Groq authentication failed (401/403). "
                        "Check GROQ_API_KEY in .env and restart the API server."
                    ) from exc
                if status_code == 400:
                    try:
                        error_body = exc.response.json().get("error", {})
                        code = error_body.get("code")
                        message = error_body.get("message", "")
                    except Exception:
                        code = None
                        message = exc.response.text[:500]

                    if code == "model_decommissioned":
                        raise RuntimeError(
                            "Groq model is decommissioned. "
                            "Set GROQ_MODEL to an active id (e.g. llama-3.3-70b-versatile) "
                            "in .env and restart the API server."
                        ) from exc

                    raise RuntimeError(f"Groq bad request (400): {message}") from exc
                if status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 5.0
                    last_error = exc
                    if attempt < self._settings.GROQ_MAX_RETRIES:
                        await asyncio.sleep(wait_seconds)
                    continue

                last_error = exc
                if attempt < self._settings.GROQ_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                continue
            except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt < self._settings.GROQ_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                continue

        logger.error("Groq triage failed after retries: {}", last_error)
        raise RuntimeError(f"Groq triage failed: {last_error}")

    async def _enforce_limits(self) -> None:
        """Simple in-process rate/quota guard (per minute + per day)."""
        async with self._lock:
            today = datetime.now(timezone.utc).date()
            if today != self._day_start:
                self._day_start = today
                self._calls_today = 0

            if self._calls_today >= self._settings.GROQ_DAILY_QUOTA:
                raise RuntimeError("Groq daily quota reached")

            now = time.monotonic()
            elapsed = now - self._minute_start
            if elapsed >= 60:
                self._minute_start = now
                self._calls_this_minute = 0

            if self._calls_this_minute >= self._settings.GROQ_RATE_LIMIT_PER_MINUTE:
                sleep_for = max(0.0, 60 - elapsed)
                await asyncio.sleep(sleep_for)
                self._minute_start = time.monotonic()
                self._calls_this_minute = 0

            self._calls_this_minute += 1
            self._calls_today += 1

    def _extract_risk_signals(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ml_score = float(payload.get("ml_score") or 0.0)
        text = " ".join(
            str(payload.get(field) or "")
            for field in ("description", "category", "shap_summary")
        ).lower()

        keywords: List[str] = [
            "split", "split invoice", "split billing", "threshold", "round-number",
            "weekend", "burst", "duplicate", "backdated", "rapid", "cluster",
            "near approval", "sequential", "same-day",
        ]
        hits = [keyword for keyword in keywords if keyword in text]

        return {
            "ml_score": ml_score,
            "signal_hits": len(hits),
            "keywords": hits[:6],
        }

    def _apply_escalation_guardrails(
        self,
        result: GroqTriageResult,
        signal_info: Dict[str, Any],
    ) -> GroqTriageResult:
        ml_score = float(signal_info.get("ml_score", 0.0))
        signal_hits = int(signal_info.get("signal_hits", 0))

        hard = ml_score >= self._settings.GROQ_HARD_ESCALATION_ML_SCORE
        soft = (
            ml_score >= self._settings.GROQ_SOFT_ESCALATION_ML_SCORE
            and signal_hits >= self._settings.GROQ_MIN_SIGNAL_HITS_FOR_SOFT_ESCALATION
        )

        if result.escalate:
            return result

        if hard or soft:
            reason_prefix = (
                "Escalation guardrail applied by policy "
                f"(ml_score={ml_score:.3f}, signal_hits={signal_hits})."
            )
            adjusted_risk = "HIGH" if hard else "MEDIUM"
            adjusted_reason = f"{reason_prefix} {result.reason}".strip()
            return GroqTriageResult(
                escalate=True,
                reason=adjusted_reason[:800],
                risk_level=adjusted_risk,
            )

        return result
