"""
backend/ingestion/usaspending_client.py
─────────────────────────────────────────────────────────────
USAspending.gov API client.

Fetches real U.S. federal procurement awards (contracts, purchase
orders) and maps them to the system's canonical Transaction schema.

API reference: https://api.usaspending.gov/docs/endpoints

Key endpoint used:
  POST /api/v2/search/spending_by_award/

This endpoint returns real contract awards with:
  - Recipient (vendor) name, UEI/DUNS number, country
  - Award amount + action date + fiscal year
  - Awarding agency + award type code
  - NAICS description (category)
  - Award ID (used as external_id for deduplication)

No API key is required for read-only access. Rate limit: ~1000 req/hr.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from backend.ingestion.base_client import BaseAPIClient, RateLimitError


# ─────────────────────────────────────────────────────────────────────────────
# Data Transfer Object – canonical form returned to the seed script
# ─────────────────────────────────────────────────────────────────────────────

class USASpendingRecord:
    """
    Normalised procurement record parsed from the USAspending API response.
    Decoupled from the ORM model to keep the ingestion layer independent.
    """

    __slots__ = (
        "external_id",
        "vendor_name",
        "vendor_uei",
        "vendor_country",
        "amount",
        "action_date",
        "fiscal_year",
        "award_type",
        "category",
        "description",
        "awarding_agency",
        "raw_data",
    )

    def __init__(
        self,
        external_id: str,
        vendor_name: str,
        vendor_uei: Optional[str],
        vendor_country: Optional[str],
        amount: float,
        action_date: datetime,
        fiscal_year: Optional[int],
        award_type: Optional[str],
        category: Optional[str],
        description: Optional[str],
        awarding_agency: Optional[str],
        raw_data: Dict[str, Any],
    ) -> None:
        self.external_id = external_id
        self.vendor_name = vendor_name.strip() if vendor_name else "UNKNOWN"
        self.vendor_uei = vendor_uei
        self.vendor_country = vendor_country
        self.amount = amount
        self.action_date = action_date
        self.fiscal_year = fiscal_year
        self.award_type = award_type
        self.category = category
        self.description = description
        self.awarding_agency = awarding_agency
        self.raw_data = raw_data

    def __repr__(self) -> str:
        return (
            f"<USASpendingRecord vendor={self.vendor_name!r} "
            f"amount={self.amount:.2f} date={self.action_date.date()}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# API Client
# ─────────────────────────────────────────────────────────────────────────────

class USASpendingClient(BaseAPIClient):
    """
    Async client for the USAspending.gov public API.

    Usage
    -----
    async with USASpendingClient() as client:
        records = await client.fetch_records(page=1, limit=100)
        for record in records:
            print(record.vendor_name, record.amount)

    Polling
    -------
    `fetch_recent_transactions()` is the primary method called by the
    polling loop in seed.py. It fetches transactions from the last N days.
    """

    # Award type codes that represent contracts (not grants/loans)
    CONTRACT_AWARD_TYPES: List[str] = ["A", "B", "C", "D"]

    # Minimum amount to filter out noise (e.g., $0 modifications)
    MIN_AMOUNT: int = 1_000

    def __init__(self, base_url: str = "https://api.usaspending.gov") -> None:
        super().__init__(
            base_url=base_url,
            timeout_seconds=45.0,   # USAspending can be slow ~10–30s
            max_retries=3,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    async def fetch_records(
        self,
        page: int = 1,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        naics_codes: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[USASpendingRecord]:
        """
        Fetch a paginated page of federal contract awards.

        Parameters
        ----------
        page        – 1-indexed page number
        limit       – records per page (max 100 per API limits)
        start_date  – ISO date string "YYYY-MM-DD" (default: 90 days ago)
        end_date    – ISO date string "YYYY-MM-DD" (default: today)
        naics_codes – optional list of 6-digit NAICS codes to filter by industry

        Returns
        -------
        List of normalised USASpendingRecord objects.
        Empty list if no records found or on non-fatal errors.
        """
        today = date.today()
        _start = start_date or (today - timedelta(days=90)).isoformat()
        _end = end_date or today.isoformat()

        payload = self._build_payload(
            start_date=_start,
            end_date=_end,
            page=page,
            limit=min(limit, 100),  # API hard cap
            naics_codes=naics_codes,
        )

        logger.info(
            "Fetching USAspending awards | page={} limit={} date_range=[{}, {}]",
            page, limit, _start, _end,
        )

        try:
            response = await self._post("/api/v2/search/spending_by_award/", json=payload)
        except RateLimitError as e:
            logger.warning("Rate limit hit. Sleeping {}s before retry.", e.retry_after)
            await asyncio.sleep(e.retry_after)
            response = await self._post("/api/v2/search/spending_by_award/", json=payload)

        results = response.get("results", [])
        total = response.get("page_metadata", {}).get("count", 0)

        logger.info(
            "Received {} records (total available: {}) from USAspending",
            len(results), total,
        )

        return [self._parse_record(r) for r in results if r is not None]

    async def fetch_recent_transactions(
        self,
        days_back: int = 7,
        max_records: int = 100,
    ) -> List[USASpendingRecord]:
        """
        Convenience method for the polling loop.

        Fetches transactions from the last `days_back` days.
        Automatically paginates if max_records > 100.
        """
        today = date.today()
        start = (today - timedelta(days=days_back)).isoformat()
        end = today.isoformat()

        all_records: List[USASpendingRecord] = []
        page = 1
        per_page = min(max_records, 100)

        while len(all_records) < max_records:
            records = await self.fetch_records(
                page=page,
                limit=per_page,
                start_date=start,
                end_date=end,
            )
            if not records:
                break
            all_records.extend(records)
            page += 1

            # Avoid hammering the API on consecutive pages
            await asyncio.sleep(0.5)

        return all_records[:max_records]

    async def get_total_pages(
        self,
        start_date: str,
        end_date: str,
        limit: int = 100,
    ) -> int:
        """
        Returns the total number of pages available for the given date range.
        Useful for full historical backfills.
        """
        payload = self._build_payload(start_date, end_date, page=1, limit=limit)
        response = await self._post("/api/v2/search/spending_by_award/", json=payload)
        metadata = response.get("page_metadata", {})
        total_count: int = metadata.get("count", 0)
        return max(1, -(-total_count // limit))  # ceiling division

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_payload(
        self,
        start_date: str,
        end_date: str,
        page: int,
        limit: int,
        naics_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
            "time_period": [
                {"start_date": start_date, "end_date": end_date}
            ],
            "award_type_codes": self.CONTRACT_AWARD_TYPES,
            "award_amounts": [
                {"lower_bound": self.MIN_AMOUNT}
            ],
        }
        if naics_codes:
            filters["naics_codes"] = naics_codes
        return {
            "filters": filters,
            "fields": [
                "Award ID",
                "Recipient Name",
                "Recipient DUNS Number",
                "recipient_id",
                "Last Modified Date",
                "Award Amount",
                "Total Outlays",
                "Awarding Agency",
                "Awarding Sub Agency",
                "Award Type",
                "Description",
                "NAICS Description",
                "Fiscal Year",
                "Place of Performance Country Code",
            ],
            "page": page,
            "limit": limit,
            "sort": "Last Modified Date",
            "order": "desc",
            "subawards": False,
        }

    def _parse_record(self, raw: Dict[str, Any]) -> USASpendingRecord:
        """
        Map a raw USAspending API result dict to a USASpendingRecord.

        Handles missing / null fields gracefully – the API does not
        guarantee all fields are populated for all award types.
        """
        # Parse last modified date (replaces Action Date which is not in contract award mappings)
        date_str: Optional[str] = raw.get("Last Modified Date")
        action_date = self._parse_date(date_str) if date_str else datetime.now(timezone.utc)

        # Normalise amount – use "Award Amount" (obligated), fallback to 0
        amount_raw = raw.get("Award Amount") or raw.get("Total Outlays") or 0.0
        try:
            amount = float(amount_raw)
        except (TypeError, ValueError):
            amount = 0.0

        return USASpendingRecord(
            external_id=str(raw.get("Award ID", "")),
            vendor_name=raw.get("Recipient Name") or "UNKNOWN VENDOR",
            vendor_uei=raw.get("Recipient DUNS Number"),
            vendor_country=raw.get("Place of Performance Country Code"),
            amount=amount,
            action_date=action_date,
            fiscal_year=raw.get("Fiscal Year"),
            award_type=raw.get("Award Type"),
            category=raw.get("NAICS Description"),
            description=raw.get("Description"),
            awarding_agency=raw.get("Awarding Agency"),
            raw_data=raw,
        )

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse ISO date string to timezone-aware datetime."""
        try:
            return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback: try common format
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
                return d.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.warning("Could not parse date string: {}", date_str)
                return datetime.now(timezone.utc)
