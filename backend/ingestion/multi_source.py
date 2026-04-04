"""Multi-source procurement ingestion adapters and canonical schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import httpx
from loguru import logger

from backend.config import get_settings
from backend.ingestion.usaspending_client import USASpendingClient


@dataclass(slots=True)
class CanonicalProcurementRecord:
    transaction_id: str
    buyer: str
    vendor: str
    amount: Decimal
    currency: str
    timestamp: datetime
    source: str  # CPPP | USA | OCDS
    description: Optional[str] = None
    category: Optional[str] = None
    country: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "buyer": self.buyer,
            "vendor": self.vendor,
            "amount": float(self.amount),
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "description": self.description,
            "category": self.category,
            "country": self.country,
            "raw": self.raw,
        }


class BaseIngestionAdapter(ABC):
    """Abstract adapter interface for procurement sources."""

    source_name: str = "UNKNOWN"

    @abstractmethod
    async def fetch(self, limit: int = 100, **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch raw records from the underlying source."""

    @abstractmethod
    def normalize(self, raw_records: Iterable[Dict[str, Any]]) -> List[CanonicalProcurementRecord]:
        """Normalize source records to canonical procurement schema."""


class USASpendingAdapter(BaseIngestionAdapter):
    source_name = "USA"

    def __init__(self) -> None:
        self._client = USASpendingClient()

    async def fetch(self, limit: int = 100, days_back: int = 7, **kwargs: Any) -> List[Dict[str, Any]]:
        async with self._client as client:
            records = await client.fetch_recent_transactions(days_back=days_back, max_records=limit)
        return [record.raw_data for record in records]

    def normalize(self, raw_records: Iterable[Dict[str, Any]]) -> List[CanonicalProcurementRecord]:
        results: List[CanonicalProcurementRecord] = []
        for raw in raw_records:
            tx_id = str(raw.get("Award ID") or raw.get("award_id") or "").strip()
            if not tx_id:
                continue

            amount_raw = raw.get("Award Amount") or raw.get("award_amount") or 0
            try:
                amount = Decimal(str(amount_raw))
            except Exception:
                amount = Decimal("0")

            date_raw = raw.get("Last Modified Date") or raw.get("action_date")
            timestamp = _safe_datetime(date_raw)
            buyer = str(raw.get("Awarding Agency") or raw.get("awarding_agency") or "UNKNOWN_BUYER").strip()
            vendor = str(raw.get("Recipient Name") or raw.get("vendor_name") or "UNKNOWN_VENDOR").strip()

            results.append(
                CanonicalProcurementRecord(
                    transaction_id=tx_id,
                    buyer=buyer or "UNKNOWN_BUYER",
                    vendor=vendor or "UNKNOWN_VENDOR",
                    amount=amount,
                    currency="USD",
                    timestamp=timestamp,
                    source=self.source_name,
                    description=raw.get("Description"),
                    category=raw.get("NAICS Description"),
                    country=raw.get("Place of Performance Country Code"),
                    raw=raw,
                )
            )
        logger.info("USASpendingAdapter normalized {} records", len(results))
        return results


class CPPPAdapter(BaseIngestionAdapter):
    """India CPPP adapter for semi-structured tender payloads."""

    source_name = "CPPP"

    def __init__(self, endpoint: str = "") -> None:
        settings = get_settings()
        self.endpoint = endpoint or settings.CPPP_BASE_URL

    async def fetch(self, limit: int = 100, payload: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        if payload is not None:
            return payload[:limit]

        records = await _fetch_source_records(endpoint=self.endpoint, source_name=self.source_name)
        return records[:limit]

    def normalize(self, raw_records: Iterable[Dict[str, Any]]) -> List[CanonicalProcurementRecord]:
        results: List[CanonicalProcurementRecord] = []
        for raw in raw_records:
            tx_id = str(raw.get("tender_id") or raw.get("id") or raw.get("reference_no") or "").strip()
            if not tx_id:
                continue

            amount_raw = raw.get("value") or raw.get("contract_value") or raw.get("estimated_value") or 0
            currency = str(raw.get("currency") or "INR").upper()
            try:
                amount = Decimal(str(amount_raw))
            except Exception:
                amount = Decimal("0")

            timestamp = _safe_datetime(raw.get("published_date") or raw.get("date") or raw.get("updated_at"))
            buyer = str(raw.get("buyer") or raw.get("department") or raw.get("authority") or "UNKNOWN_BUYER").strip()
            vendor = str(raw.get("vendor") or raw.get("awarded_to") or raw.get("supplier") or "UNKNOWN_VENDOR").strip()

            results.append(
                CanonicalProcurementRecord(
                    transaction_id=tx_id,
                    buyer=buyer or "UNKNOWN_BUYER",
                    vendor=vendor or "UNKNOWN_VENDOR",
                    amount=amount,
                    currency=currency,
                    timestamp=timestamp,
                    source=self.source_name,
                    description=raw.get("description") or raw.get("work_description"),
                    category=raw.get("category") or raw.get("sector"),
                    country="IN",
                    raw=raw,
                )
            )
        logger.info("CPPPAdapter normalized {} records", len(results))
        return results


class OCDSAdapter(BaseIngestionAdapter):
    """Open Contracting Data Standard adapter."""

    source_name = "OCDS"

    def __init__(self, endpoint: str = "") -> None:
        settings = get_settings()
        self.endpoint = endpoint or settings.OCDS_BASE_URL

    async def fetch(self, limit: int = 100, payload: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        if payload is not None:
            return payload[:limit]

        primary_payload = await _fetch_any_payload(endpoint=self.endpoint, source_name=self.source_name)
        records = _extract_records(primary_payload)
        if records:
            return records[:limit]

        # Support CKAN-style catalog endpoints (e.g., data.open-contracting.org)
        # by discovering JSON resources and extracting OCDS release records.
        discovered_urls = _extract_json_resource_urls(primary_payload)
        if not discovered_urls:
            return []

        aggregated: List[Dict[str, Any]] = []
        for url in discovered_urls[:10]:
            try:
                child_payload = await _fetch_any_payload(endpoint=url, source_name=self.source_name)
                child_records = _extract_records(child_payload)
                if child_records:
                    aggregated.extend(child_records)
            except Exception as exc:
                logger.warning("OCDS child resource fetch failed for {}: {}", url, exc)

            if len(aggregated) >= limit:
                break

        return aggregated[:limit]

    def normalize(self, raw_records: Iterable[Dict[str, Any]]) -> List[CanonicalProcurementRecord]:
        results: List[CanonicalProcurementRecord] = []
        for raw in raw_records:
            record = _coerce_ocds_record(raw)
            tx_id = str(record.get("ocid") or record.get("id") or record.get("releaseID") or "").strip()
            if not tx_id:
                continue

            tender = record.get("tender") or {}
            awards = record.get("awards") or []
            contracts = record.get("contracts") or []
            amount_raw, currency = _ocds_value(tender=tender, awards=awards, contracts=contracts)
            try:
                amount = Decimal(str(amount_raw))
            except Exception:
                amount = Decimal("0")

            buyer_block = record.get("buyer") or {}
            parties = record.get("parties") or []
            buyer = str(buyer_block.get("name") or "UNKNOWN_BUYER").strip()
            vendor = _ocds_vendor_name(awards, contracts, parties)

            timestamp = _safe_datetime(
                record.get("date")
                or record.get("publishedDate")
                or tender.get("tenderPeriod", {}).get("startDate")
            )

            results.append(
                CanonicalProcurementRecord(
                    transaction_id=tx_id,
                    buyer=buyer or "UNKNOWN_BUYER",
                    vendor=vendor or "UNKNOWN_VENDOR",
                    amount=amount,
                    currency=str(currency or "USD").upper(),
                    timestamp=timestamp,
                    source=self.source_name,
                    description=tender.get("description") or record.get("description"),
                    category=tender.get("mainProcurementCategory"),
                    country=_ocds_country(record),
                    raw=record,
                )
            )
        logger.info("OCDSAdapter normalized {} records", len(results))
        return results


def _safe_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        candidate = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _ocds_value(tender: Dict[str, Any], awards: List[Dict[str, Any]], contracts: List[Dict[str, Any]]) -> tuple[Any, Any]:
    for block in contracts:
        value = (block or {}).get("value") or {}
        if value.get("amount") is not None:
            return value.get("amount"), value.get("currency")
    for block in awards:
        value = (block or {}).get("value") or {}
        if value.get("amount") is not None:
            return value.get("amount"), value.get("currency")
    value = tender.get("value") or {}
    return value.get("amount", 0), value.get("currency", "USD")


def _ocds_vendor_name(awards: List[Dict[str, Any]], contracts: List[Dict[str, Any]], parties: List[Dict[str, Any]]) -> str:
    award_supplier_ids: List[str] = []
    for award in awards:
        suppliers = (award or {}).get("suppliers") or []
        for supplier in suppliers:
            sid = supplier.get("id")
            if sid:
                award_supplier_ids.append(str(sid))
            if supplier.get("name"):
                return str(supplier.get("name"))

    for contract in contracts:
        if contract.get("title"):
            continue

    if award_supplier_ids:
        party_map = {str((party or {}).get("id")): party for party in parties if (party or {}).get("id")}
        for sid in award_supplier_ids:
            party = party_map.get(sid)
            if party and party.get("name"):
                return str(party.get("name"))

    for party in parties:
        roles = [str(role).lower() for role in ((party or {}).get("roles") or [])]
        if "supplier" in roles and (party or {}).get("name"):
            return str((party or {}).get("name"))

    return "UNKNOWN_VENDOR"


def _ocds_country(raw: Dict[str, Any]) -> Optional[str]:
    buyer = raw.get("buyer") or {}
    address = buyer.get("address") or {}
    country = address.get("countryName") or address.get("countryCode")
    if country:
        return str(country)
    return None


async def _fetch_source_records(endpoint: str, source_name: str) -> List[Dict[str, Any]]:
    payload = await _fetch_any_payload(endpoint=endpoint, source_name=source_name)
    records = _extract_records(payload)
    if records:
        logger.info("{} fetched {} raw records", source_name, len(records))
        return records

    if isinstance(payload, str):
        html_records = _extract_cppp_records_from_html(payload)
        if html_records:
            logger.info("{} parsed {} records from HTML source", source_name, len(html_records))
            return html_records

    logger.info("{} fetched 0 raw records", source_name)
    return []


async def _fetch_any_payload(endpoint: str, source_name: str) -> Any:
    if not endpoint:
        logger.warning("{} endpoint not configured; returning empty pull", source_name)
        return []

    payload: Any
    endpoint_path = Path(endpoint)
    if endpoint_path.exists() and endpoint_path.is_file():
        payload = json.loads(endpoint_path.read_text(encoding="utf-8"))
    elif endpoint.startswith("file://"):
        payload = json.loads(Path(endpoint.replace("file://", "", 1)).read_text(encoding="utf-8"))
    else:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            lower_endpoint = endpoint.lower()

            if lower_endpoint.endswith(".jsonl.gz") or "gzip" in content_type:
                payload = _parse_jsonl_gzip(response.content)
                return payload

            if lower_endpoint.endswith(".jsonl"):
                payload = _parse_jsonl_text(response.text)
                return payload

            if "json" in content_type:
                payload = response.json()
            else:
                text = response.text
                try:
                    payload = response.json()
                except Exception:
                    payload = text

    return payload


def _extract_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("records", "results", "data", "items", "tenders", "releases"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    return []


def _parse_jsonl_text(text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                records.append(row)
        except Exception:
            continue
    return records


def _parse_jsonl_gzip(binary_payload: bytes) -> List[Dict[str, Any]]:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(binary_payload)) as gz:
            text = gz.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    return _parse_jsonl_text(text)


def _coerce_ocds_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    if raw.get("ocid"):
        return raw

    releases = raw.get("releases")
    if isinstance(releases, list):
        for release in releases:
            if isinstance(release, dict) and release.get("ocid"):
                return release

    records = raw.get("records")
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            compiled = item.get("compiledRelease")
            if isinstance(compiled, dict) and compiled.get("ocid"):
                return compiled

    return raw


def _extract_json_resource_urls(payload: Any) -> List[str]:
    urls: List[str] = []
    if not isinstance(payload, dict):
        return urls

    result = payload.get("result")
    if not isinstance(result, dict):
        return urls

    packages = result.get("results")
    if not isinstance(packages, list):
        return urls

    for package in packages:
        if not isinstance(package, dict):
            continue
        resources = package.get("resources")
        if not isinstance(resources, list):
            continue
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            url = str(resource.get("url") or "").strip()
            fmt = str(resource.get("format") or "").lower()
            if url and (fmt in {"json", "jsonl", "ndjson"} or url.lower().endswith(".json")):
                urls.append(url)

    deduped: List[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _extract_cppp_records_from_html(html: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not html:
        return rows

    html_flat = re.sub(r"\s+", " ", html)
    pattern = re.compile(
        r"<tr[^>]*>\s*<td[^>]*>(?P<title>.*?)</td>\s*<td[^>]*>(?P<ref>.*?)</td>\s*<td[^>]*>(?P<close>.*?)</td>\s*<td[^>]*>(?P<open>.*?)</td>\s*</tr>",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(html_flat):
        title = _strip_html(match.group("title"))
        reference = _strip_html(match.group("ref"))
        closing_date = _strip_html(match.group("close"))
        opening_date = _strip_html(match.group("open"))

        if not reference or reference.lower() in {"reference no", "reference"}:
            continue
        if not title or title.lower() in {"tender title", "title"}:
            continue
        if "|" in reference or "|" in title:
            continue
        if len(reference) > 80 or len(title) > 500:
            continue
        if not re.search(r"\d", reference):
            continue
        if not _looks_like_date(closing_date) and not _looks_like_date(opening_date):
            continue

        rows.append(
            {
                "id": reference,
                "tender_id": reference,
                "description": title,
                "reference_no": reference,
                "published_date": closing_date or opening_date,
                "updated_at": opening_date,
                "buyer": "CPPP",
                "vendor": "UNKNOWN_VENDOR",
                "estimated_value": 0,
                "currency": "INR",
                "source_page": "CPPP_ACTIVE_TENDERS",
            }
        )

    return rows


def _looks_like_date(value: str) -> bool:
    if not value:
        return False
    checks = [
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
    ]
    return any(re.search(pattern, value) for pattern in checks)


def _strip_html(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value or "")
    decoded = (
        no_tags.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"\s+", " ", decoded).strip()
