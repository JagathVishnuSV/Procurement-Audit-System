"""Data quality and schema-normalization utilities for multi-source ingestion."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Dict, Iterable, List, Sequence, Tuple

from backend.ingestion.multi_source import CanonicalProcurementRecord


_CURRENCY_TO_USD: Dict[str, Decimal] = {
    "USD": Decimal("1.00"),
    "INR": Decimal("0.012"),
    "EUR": Decimal("1.09"),
    "GBP": Decimal("1.28"),
}


def normalize_vendor_name(name: str) -> str:
    value = unicodedata.normalize("NFKD", name or "")
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\b(private|pvt|limited|ltd|corp|corporation|inc|llp|co|company)\b", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "unknown_vendor"


def normalize_buyer_name(name: str) -> str:
    value = unicodedata.normalize("NFKD", name or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value or "UNKNOWN_BUYER"


def convert_to_usd(amount: Decimal, currency: str) -> Tuple[Decimal, str]:
    code = (currency or "USD").upper()
    rate = _CURRENCY_TO_USD.get(code)
    if rate is None:
        return amount, code
    return (amount * rate).quantize(Decimal("0.01")), "USD"


def handle_missing_fields(record: CanonicalProcurementRecord) -> CanonicalProcurementRecord:
    amount = record.amount
    try:
        amount = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError):
        amount = Decimal("0")

    return replace(
        record,
        buyer=normalize_buyer_name(record.buyer),
        vendor=record.vendor or "UNKNOWN_VENDOR",
        amount=amount,
        currency=(record.currency or "USD").upper(),
    )


def normalize_currency(record: CanonicalProcurementRecord, force_usd: bool = True) -> CanonicalProcurementRecord:
    if not force_usd:
        return record
    usd_amount, currency = convert_to_usd(record.amount, record.currency)
    return replace(record, amount=usd_amount, currency=currency)


def record_fingerprint(record: CanonicalProcurementRecord) -> str:
    canonical = "|".join(
        [
            record.transaction_id.strip().lower(),
            normalize_vendor_name(record.vendor),
            normalize_buyer_name(record.buyer).lower(),
            str(record.amount),
            record.timestamp.date().isoformat(),
            record.source.upper(),
        ]
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def deduplicate_records(records: Sequence[CanonicalProcurementRecord]) -> List[CanonicalProcurementRecord]:
    seen_tx_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    deduped: List[CanonicalProcurementRecord] = []

    for record in records:
        tx_id = record.transaction_id.strip().lower()
        if tx_id and tx_id in seen_tx_ids:
            continue

        fingerprint = record_fingerprint(record)
        if fingerprint in seen_fingerprints:
            continue

        if tx_id:
            seen_tx_ids.add(tx_id)
        seen_fingerprints.add(fingerprint)
        deduped.append(record)

    return deduped


def run_data_quality_pipeline(records: Iterable[CanonicalProcurementRecord]) -> List[CanonicalProcurementRecord]:
    staged: List[CanonicalProcurementRecord] = []

    for record in records:
        with_missing = handle_missing_fields(record)
        normalized = replace(
            with_missing,
            vendor=normalize_vendor_name(with_missing.vendor),
            buyer=normalize_buyer_name(with_missing.buyer),
        )
        normalized = normalize_currency(normalized, force_usd=True)
        staged.append(normalized)

    return deduplicate_records(staged)
