**English** | [Українська](README.uk.md)

[![CI](https://github.com/ZavarOvek/tf-idf-summariser/actions/workflows/ci.yml/badge.svg)](https://github.com/ZavarOvek/tf-idf-summariser/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# tfidf-sum — Extractive TF-IDF Summarizer (Ukrainian & English)

A command-line tool and Python library for extractive text summarization:
each sentence is scored by the TF-IDF weights of its terms, and the
top-weighted sentences are returned **in their original order** to preserve
coherence.

Built as a production rewrite of my applied-linguistics coursework at Taras
Shevchenko National University of Kyiv.

## Features

- Works for Ukrainian and English out of the box, with automatic language
  detection and per-language stopword lists
- No model downloads: a self-contained regex sentence splitter with an
  abbreviation guard (`тис.`, `напр.`, `Dr.`, `e.g.`) replaces punkt, so the
  tool runs identically in any environment
- Summary size by ratio (`--ratio 0.3`) or exact sentence count
  (`--sentences 5`)
- Optional length-bias correction (`--normalize`): score by mean token weight
  instead of sum, so long sentences don't win automatically
- Reports compression ratio and the highest-weighted terms of the document
- Transparent and deterministic — no training, no API calls; fully tested
  (pytest)

## Install

```bash
pip install .
```

## Usage

```bash
tfidf-sum article.txt --sentences 3 --top-terms 5
```

```
Він фіксує реальну вживаність мовних одиниць у текстах певного стилю. На
основі частотних даних будуються статистичні моделі лексики. Лематизація
дозволяє звести словоформи до початкової форми і точніше рахувати частоти.

[3 sentences, compression: 32%]

top terms:
  мови        0.677
  частотний   0.663
  стилю       0.654
```

Off-topic sentences in the source (in the bundled example — two sentences
about the Carpathians in a text about corpus linguistics) are correctly left
out of the summary.

## As a library

```python
from tfidfsum import summarize

result = summarize(open("article.txt", encoding="utf-8").read(), ratio=0.3)
print(result.text)
print(result.compression, result.top_terms[:5])
```

## Limitations & roadmap

Extractive summarization reuses source sentences verbatim — no paraphrasing.
Planned: optional lemmatization for Ukrainian (pymorphy3) to merge inflected
forms before weighting, and a positional prior for news-style texts.

## Testing

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## License

MIT
