# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- Adopted `ruff format` as the project formatter; the whole tree was
  reformatted in a single dedicated commit (see `.git-blame-ignore-revs`).
- `E501` disabled in `ruff check`: line length is the formatter's job, and
  the formatter cannot split string literals anyway.
- CI lint job now also runs `ruff format --check .`.

## [0.1.1] - 2026-08-08

### Added
- GitHub Actions CI: `ruff` lint job and a `pytest` matrix on Python 3.10-3.12.
- Ruff configuration in `pyproject.toml` (`RUF001-003` disabled: they flag
  Cyrillic letters as ambiguous look-alikes of Latin ones, which is a false
  positive in a project about Ukrainian text).
- Package metadata: `keywords`, `classifiers` and `[project.urls]`.
- Status badges in both READMEs.

### Changed
- `license` migrated to the PEP 639 SPDX form (`license = "MIT"` +
  `license-files`); build now requires `setuptools>=77`.
- Import order normalized by ruff; no behaviour changes.

## [0.1.0] - 2026-07-20

### Added
- Initial release: extractive TF-IDF summarizer for Ukrainian and English.
- Dependency-free sentence splitter with an abbreviation guard (no punkt
  download required).
- Automatic language detection and per-language stopword lists.
- Summary size by ratio (`--ratio`) or exact sentence count (`--sentences`).
- Optional length-bias correction (`--normalize`) scoring by mean token weight.
- Compression ratio and top weighted terms reported on stderr.
- `tfidf-sum` CLI entry point and `summarize()` library API.
- Test suite covering splitting, detection, summarization and the CLI.
