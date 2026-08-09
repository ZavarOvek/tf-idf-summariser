"""Tests for tfidfsum."""

from pathlib import Path

import pytest

from tfidfsum.cli import main
from tfidfsum.sentences import detect_lang, split_sentences
from tfidfsum.summarizer import summarize

UK_TEXT = (
    "Корпусна лінгвістика вивчає мову на основі великих зібрань текстів. "
    "Корпуси дозволяють досліджувати частоту слів у реальному вжитку. "
    "Кіт спав на підвіконні цілий день. "
    "Статистичні методи є основою аналізу корпусів текстів. "
    "Аналіз корпусів допомагає укладати сучасні словники. "
    "Погода сьогодні була сонячна і тепла."
)


class TestSplitSentences:
    def test_basic_split(self):
        assert len(split_sentences("Перше речення. Друге речення!")) == 2

    def test_abbreviation_not_split(self):
        sents = split_sentences("Це коштує 5 тис. Грн зібрали швидко.")
        # "тис." must not end a sentence
        assert sents[0].startswith("Це коштує 5 тис.")

    def test_ellipsis(self):
        sents = split_sentences("Він пішов… Вона лишилась.")
        assert len(sents) == 2

    def test_english_abbrev(self):
        sents = split_sentences("Dr. Smith arrived. He was late.")
        assert len(sents) == 2
        assert sents[0] == "Dr. Smith arrived."

    def test_empty(self):
        assert split_sentences("   ") == []


class TestDetectLang:
    def test_ukrainian(self):
        assert detect_lang("Мова — це зброя нації.") == "uk"

    def test_english(self):
        assert detect_lang("Language is a weapon.") == "en"


class TestSummarize:
    def test_sentence_count(self):
        result = summarize(UK_TEXT, n_sentences=2)
        assert len(result.sentences) == 2

    def test_original_order_preserved(self):
        result = summarize(UK_TEXT, n_sentences=3)
        assert result.selected_indices == sorted(result.selected_indices)

    def test_topic_sentences_beat_off_topic(self):
        result = summarize(UK_TEXT, n_sentences=3)
        # the cat and the weather are off-topic and should not be picked
        assert not any("Кіт" in s or "Погода" in s for s in result.sentences)

    def test_compression_below_one(self):
        result = summarize(UK_TEXT, ratio=0.3)
        assert 0 < result.compression < 1

    def test_top_terms_present(self):
        result = summarize(UK_TEXT, n_sentences=2)
        terms = [t for t, _ in result.top_terms]
        assert "корпусів" in terms or "корпуси" in terms

    def test_single_sentence_passthrough(self):
        result = summarize("Одне речення.")
        assert result.text == "Одне речення."
        assert result.compression == 1.0

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            summarize(UK_TEXT, ratio=1.5)

    def test_normalize_flag_runs(self):
        result = summarize(UK_TEXT, n_sentences=2, normalize=True)
        assert len(result.sentences) == 2


class TestCli:
    def test_end_to_end(self, tmp_path: Path, capsys):
        src = tmp_path / "in.txt"
        src.write_text(UK_TEXT, encoding="utf-8")
        out = tmp_path / "summary.txt"
        code = main([str(src), "--sentences", "2", "-o", str(out), "--top-terms", "5"])
        assert code == 0
        assert out.exists()
        assert len(out.read_text(encoding="utf-8").strip()) > 0

    def test_missing_file(self, tmp_path: Path):
        assert main([str(tmp_path / "nope.txt")]) == 1

    def test_empty_file(self, tmp_path: Path):
        src = tmp_path / "empty.txt"
        src.write_text("", encoding="utf-8")
        assert main([str(src)]) == 1
