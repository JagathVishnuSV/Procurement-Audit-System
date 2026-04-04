"""Normalization helpers for source-level procurement analytics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NormalizedSource:
    source_system: str
    agency_family: str
    category_family: str


def _normalize_source_system(raw_source: Optional[str]) -> str:
    source = (raw_source or "UNKNOWN").upper()
    source_map = {
        "USASPENDING": "US_FEDERAL",
        "CPPP": "INDIA_CPPP",
        "OCDS": "OPEN_CONTRACTING",
        "WORLDBANK": "INTERNATIONAL",
        "SEC_EDGAR": "CORPORATE_DISCLOSURES",
        "NYC_OPENDATA": "US_MUNICIPAL",
        "MANUAL": "MANUAL",
        "UNKNOWN": "UNKNOWN",
    }
    return source_map.get(source, source)


def _normalize_agency_family(awarding_agency: Optional[str]) -> str:
    value = (awarding_agency or "").strip().lower()
    if not value:
        return "UNKNOWN"

    rules = [
        ("defense|army|navy|air force|dod", "DEFENSE"),
        ("health|hhs|medicare|nih", "HEALTH"),
        ("transport|dot|aviation|faa", "TRANSPORT"),
        ("energy|doe", "ENERGY"),
        ("homeland|dhs", "HOMELAND_SECURITY"),
        ("veterans|va", "VETERANS"),
        ("nasa", "SPACE"),
        ("treasury|irs", "TREASURY"),
        ("gsa", "GSA"),
    ]

    for pattern, family in rules:
        if re.search(pattern, value):
            return family

    return "OTHER_FEDERAL"


def _normalize_category_family(raw_category: Optional[str]) -> str:
    value = (raw_category or "").strip().lower()
    if not value:
        return "UNKNOWN"

    if re.match(r"^54\\d{2}", value):
        return "PROFESSIONAL_SERVICES"
    if re.match(r"^23\\d{2}", value):
        return "CONSTRUCTION"
    if re.match(r"^48|^49", value):
        return "LOGISTICS"
    if re.match(r"^33|^31|^32", value):
        return "MANUFACTURING"

    keyword_rules = [
        ("computer|software|it|cloud|cyber", "INFORMATION_TECHNOLOGY"),
        ("construction|building|facility", "CONSTRUCTION"),
        ("consult|advisory|professional", "PROFESSIONAL_SERVICES"),
        ("medical|health|pharma|clinical", "HEALTHCARE"),
        ("transport|freight|shipping|logistics", "LOGISTICS"),
        ("research|engineering|scientific", "ENGINEERING_RND"),
    ]

    for pattern, family in keyword_rules:
        if re.search(pattern, value):
            return family

    return "OTHER"


def normalize_source_dimensions(
    raw_source: Optional[str],
    awarding_agency: Optional[str],
    category: Optional[str],
) -> NormalizedSource:
    return NormalizedSource(
        source_system=_normalize_source_system(raw_source),
        agency_family=_normalize_agency_family(awarding_agency),
        category_family=_normalize_category_family(category),
    )
