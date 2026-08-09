"""Sentence splitting and language detection.

Deliberately dependency-free (no punkt model downloads): a regex splitter
with an abbreviation guard, good enough for news/prose in Ukrainian and
English and fully reproducible in any environment.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_BOUNDARY = re.compile(r"(?<=[.!?\u2026])\s+(?=[«\"'(\[]?[A-ZА-ЯЄІЇҐ0-9])")
_TRAILING_WORD = re.compile(r"([\w'\u2019]+)\.$", re.UNICODE)

# Common abbreviations that end with a period mid-sentence
_ABBREVIATIONS = {
    # Ukrainian
    "т",
    "тис",
    "грн",
    "напр",
    "див",
    "ст",
    "п",
    "пп",
    "р",
    "рр",
    "с",
    "обл",
    "вул",
    "буд",
    "ім",
    "акад",
    "проф",
    "доц",
    "канд",
    # English
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "st",
    "no",
    "vs",
    "etc",
    "approx",
    "e.g",
    "i.e",
    "fig",
    "vol",
}


def _ends_with_abbreviation(fragment: str) -> bool:
    match = _TRAILING_WORD.search(fragment)
    return bool(match) and match.group(1).lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split raw text into sentences, merging false splits after abbreviations."""
    flat = _WS.sub(" ", text).strip()
    if not flat:
        return []
    merged: list[str] = []
    for part in _BOUNDARY.split(flat):
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def detect_lang(text: str) -> str:
    """Return ``"uk"`` if Cyrillic letters dominate, otherwise ``"en"``."""
    cyr = len(re.findall(r"[а-яєіїґ]", text, re.IGNORECASE))
    lat = len(re.findall(r"[a-z]", text, re.IGNORECASE))
    return "uk" if cyr >= lat else "en"
