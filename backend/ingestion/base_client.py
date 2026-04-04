"""
backend/ingestion/base_client.py
─────────────────────────────────────────────────────────────
Abstract async HTTP client.

Provides a reusable base for all external API clients with:
  • Automatic retries with exponential back-off (tenacity)
  • Circuit breaker via retry_if_exception_type
  • Structured logging (loguru)
  • Request/response timeout enforcement
  • Rate-limit aware (respects Retry-After headers)
  • Base headers + user-agent injection

Subclasses only need to implement `_build_request_params` and
`fetch_records` for their specific API.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ClientError(Exception):
    """Non-retryable client error (4xx)."""
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class ServerError(Exception):
    """Retryable server error (5xx / network failure)."""
    def __init__(self, message: str) -> None:
        super().__init__(f"Server error: {message}")


class RateLimitError(Exception):
    """API rate limit hit. Includes retry_after_seconds."""
    def __init__(self, retry_after: int = 60) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s")


# ─────────────────────────────────────────────────────────────────────────────
# Base Client
# ─────────────────────────────────────────────────────────────────────────────

class BaseAPIClient(ABC):
    """
    Abstract base for all external API clients.

    Configuration
    -------------
    base_url        – API root URL (e.g. "https://api.usaspending.gov")
    timeout_seconds – per-request timeout
    max_retries     – maximum retry attempts for server errors
    """

    DEFAULT_TIMEOUT: float = 30.0
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_BACKOFF_MIN: float = 1.0
    DEFAULT_BACKOFF_MAX: float = 60.0

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_retries = max_retries
        self._extra_headers = extra_headers or {}
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "BaseAPIClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._default_headers(),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _default_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ProcurementAuditSystem/1.0",
        }
        headers.update(self._extra_headers)
        return headers

    # ── Core HTTP methods ──────────────────────────────────────────────────────

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a GET request with retry logic."""
        return await self._request("GET", path, params=params)

    async def _post(self, path: str, json: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a POST request with retry logic."""
        return await self._request("POST", path, json=json)

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute an HTTP request with exponential back-off retry.

        Retries on:
          - httpx.TransportError  (network-level failures)
          - ServerError           (5xx responses)

        Does NOT retry on:
          - ClientError (4xx – bug in request, not transient)
          - RateLimitError (handled by caller with explicit sleep)
        """
        if self._client is None:
            raise RuntimeError(
                "Client not initialised. Use as async context manager: "
                "`async with client as c: ...`"
            )

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((httpx.TransportError, ServerError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                min=self.DEFAULT_BACKOFF_MIN,
                max=self.DEFAULT_BACKOFF_MAX,
            ),
            reraise=True,
        ):
            with attempt:
                logger.debug(
                    "HTTP {} {} params={} attempt={}",
                    method,
                    path,
                    params,
                    attempt.retry_state.attempt_number,
                )
                response = await self._client.request(
                    method=method,
                    url=path,
                    params=params,
                    json=json,
                )
                return self._handle_response(response)

    def _handle_response(self, response: httpx.Response) -> Any:
        """Parse response and raise typed exceptions for error status codes."""
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning("Rate limited by {}. Retry after {}s", self.base_url, retry_after)
            raise RateLimitError(retry_after=retry_after)

        if 400 <= response.status_code < 500:
            raise ClientError(
                response.status_code,
                response.text[:500],
            )

        if response.status_code >= 500:
            raise ServerError(
                f"[{response.status_code}] {response.text[:200]}"
            )

        response.raise_for_status()
        return response.json()

    # ── Abstract interface ──────────────────────────────────────────────────────

    @abstractmethod
    async def fetch_records(
        self,
        page: int = 1,
        limit: int = 100,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Fetch a page of records from the external API.

        Subclasses must implement this method and return a list of
        raw record dicts that `seed.py` will persist to PostgreSQL.
        """
        ...
