"""Extractive TF-IDF summarization.

Each sentence is treated as a document; its weight is the sum (or the
per-token mean with ``normalize=True``) of the TF-IDF weights of its terms.
The top-weighted sentences are returned in their original order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .sentences import detect_lang, split_sentences
from .stopwords import stopwords_for

# Words with letters (Cyrillic or Latin), apostrophes and in-word hyphens
_TOKEN_PATTERN = r"[а-щьюяєіїґa-z]+(?:['\u2019\-][а-щьюяєіїґa-z]+)*"


@dataclass
class Summary:
    sentences: list[str]
    selected_indices: list[int]
    compression: float
    top_terms: list[tuple[str, float]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def summarize(
    text: str,
    ratio: float = 0.3,
    n_sentences: int | None = None,
    lang: str = "auto",
    normalize: bool = False,
    n_top_terms: int = 15,
) -> Summary:
    """Summarize ``text`` down to ``n_sentences`` or a ``ratio`` of sentences.

    ``lang`` is ``"uk"``, ``"en"`` or ``"auto"``; ``normalize=True`` scores
    sentences by mean token weight instead of sum, removing the bias toward
    long sentences.
    """
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return Summary(sentences, list(range(len(sentences))), 1.0)

    if lang == "auto":
        lang = detect_lang(text)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=_TOKEN_PATTERN,
        stop_words=stopwords_for(lang),
    )
    matrix = vectorizer.fit_transform(sentences)

    scores = np.asarray(matrix.sum(axis=1)).ravel()
    if normalize:
        lengths = np.maximum(matrix.getnnz(axis=1), 1)
        scores = scores / lengths

    k = n_sentences if n_sentences else max(1, round(len(sentences) * ratio))
    k = min(k, len(sentences))
    selected = sorted(np.argsort(scores)[::-1][:k].tolist())

    term_weights = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = np.argsort(term_weights)[::-1][:n_top_terms]
    top_terms = [(terms[i], float(term_weights[i])) for i in top_idx]

    picked = [sentences[i] for i in selected]
    compression = len(" ".join(picked)) / max(len(" ".join(sentences)), 1)
    return Summary(picked, selected, compression, top_terms)
