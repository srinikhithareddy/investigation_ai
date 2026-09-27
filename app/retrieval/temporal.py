from __future__ import annotations

import re
from typing import Optional


_YEAR_PATTERN = re.compile(r"(?<![\w.-])(?:18|19|20|21)\d{2}(?![\w.])")
_VERSION_PATTERN = re.compile(r"\bv?\d+(?:\.\d+)*\b", re.IGNORECASE)


def historical_year_from_text(text: str) -> Optional[int]:
    """Return a single explicit target year; leave multi-year comparisons unfiltered."""
    years = {int(match.group(0)) for match in _YEAR_PATTERN.finditer(text)}
    return next(iter(years)) if len(years) == 1 else None


def document_topic_key(document: dict) -> tuple[str, str, str]:
    """Group versioned records by service, type, and version-neutral title."""
    title = str(document.get("title") or "").lower()
    title = _YEAR_PATTERN.sub(" ", title)
    title = _VERSION_PATTERN.sub(" ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    return (
        str(document.get("service") or "").strip().lower(),
        str(document.get("type") or "").strip().lower(),
        title,
    )