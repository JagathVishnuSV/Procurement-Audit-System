"""Entity resolution for vendor canonicalization and clustering."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple
from uuid import NAMESPACE_DNS, uuid5


@dataclass(slots=True)
class ResolvedEntity:
    canonical_name: str
    canonical_id: str
    variants: List[str]


class EntityResolver:
    """
    Resolves vendor string variants into canonical entities.

    Example:
      ABC Ltd / A.B.C Limited / ABC Pvt Ltd -> abc
    """

    def __init__(self, fuzzy_threshold: float = 0.88) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    def normalize_name(self, name: str) -> str:
        text = unicodedata.normalize("NFKD", name or "")
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\b(private|pvt|limited|ltd|corp|corporation|inc|co|company|llp)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or "unknown_vendor"

    def similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(a=left, b=right).ratio()

    def canonical_vendor_id(self, name: str) -> str:
        canonical = self.normalize_name(name)
        return str(uuid5(NAMESPACE_DNS, f"vendor:{canonical}"))

    def cluster_vendors(self, names: Iterable[str]) -> List[ResolvedEntity]:
        clusters: List[ResolvedEntity] = []

        for raw_name in names:
            clean = self.normalize_name(raw_name)
            assigned = False

            for cluster in clusters:
                score = self.similarity(clean, cluster.canonical_name)
                if score >= self.fuzzy_threshold:
                    if raw_name not in cluster.variants:
                        cluster.variants.append(raw_name)
                    assigned = True
                    break

            if not assigned:
                clusters.append(
                    ResolvedEntity(
                        canonical_name=clean,
                        canonical_id=self.canonical_vendor_id(clean),
                        variants=[raw_name],
                    )
                )

        return clusters

    def mapping(self, names: Iterable[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for cluster in self.cluster_vendors(names):
            for variant in cluster.variants:
                result[variant] = cluster.canonical_name
        return result

    def mapping_with_ids(self, names: Iterable[str]) -> Dict[str, Tuple[str, str]]:
        result: Dict[str, Tuple[str, str]] = {}
        for cluster in self.cluster_vendors(names):
            for variant in cluster.variants:
                result[variant] = (cluster.canonical_name, cluster.canonical_id)
        return result

    def pairwise_matches(self, names: Iterable[str]) -> List[Tuple[str, str, float]]:
        normalized = [(name, self.normalize_name(name)) for name in names]
        matches: List[Tuple[str, str, float]] = []

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                left_raw, left = normalized[i]
                right_raw, right = normalized[j]
                score = self.similarity(left, right)
                if score >= self.fuzzy_threshold:
                    matches.append((left_raw, right_raw, round(score, 3)))

        return sorted(matches, key=lambda item: item[2], reverse=True)
