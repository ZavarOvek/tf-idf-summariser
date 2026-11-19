"""Apostrophe normalization.

Ukrainian words like м'ясо are typed with many look-alike characters. Only the
ASCII apostrophe U+0027 is recognized by pymorphy3 dictionaries, so it is the
canonical form here too: all variants collapse to it, and м'ясо / м’ясо / мʼясо
count as one word.
"""

from __future__ import annotations

import re

# Variants: ’ ʼ ‘ ` ´ ′ ʹ ＇. Replaced only between two letters, so quotation
# marks around a word (‘цитата’) stay untouched.
_INTRAWORD_APOSTROPHE = re.compile(r"(?<=[^\W\d_])[’ʼ‘`´′ʹ＇](?=[^\W\d_])")


def normalize_apostrophes(text: str) -> str:
    """Replace intra-word apostrophe variants with the ASCII apostrophe."""
    return _INTRAWORD_APOSTROPHE.sub("'", text)
