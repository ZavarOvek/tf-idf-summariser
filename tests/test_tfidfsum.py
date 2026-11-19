"""Tests for tfidfsum."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from tfidfsum import SummarizationError
from tfidfsum.apostrophes import normalize_apostrophes
from tfidfsum.cli import main
from tfidfsum.sentences import detect_lang, split_sentences
from tfidfsum.stopwords import UKRAINIAN_STOPWORDS, stopwords_for
from tfidfsum.summarizer import _TOKEN_PATTERN, summarize

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

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_default_scoring_matches_sklearn_l2(self, k):
        # The default path must stay what it was: sum of sklearn's default
        # (L2-normalized) TF-IDF rows. Guards the split into raw + L2 matrices.
        sentences = split_sentences(UK_TEXT)
        matrix = TfidfVectorizer(
            token_pattern=_TOKEN_PATTERN, stop_words=stopwords_for("uk")
        ).fit_transform(sentences)
        expected = sorted(np.argsort(np.asarray(matrix.sum(axis=1)).ravel())[::-1][:k].tolist())
        assert summarize(UK_TEXT, n_sentences=k).selected_indices == expected

    def test_normalize_does_not_reward_shortness_itself(self):
        # The mean of an L2-normalized row is 1/sqrt(n) for n equal weights,
        # so under the old implementation any one-word sentence scored 1.0,
        # the maximum possible, whatever the word. Here the one-word sentence
        # is the most common word in the text and must lose on content.
        text = "Корпус росте. Корпус великий. Корпус. Лематизація вимагає морфологічного словника."
        result = summarize(text, n_sentences=1, normalize=True)
        assert result.sentences == ["Лематизація вимагає морфологічного словника."]

    @pytest.mark.parametrize("n", [0, -1])
    def test_invalid_sentence_count(self, n):
        # -1 used to slice argsort as [:-1] and return all but one sentence.
        with pytest.raises(ValueError, match="n_sentences"):
            summarize(UK_TEXT, n_sentences=n)


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

    @pytest.mark.parametrize("n", ["0", "-1"])
    def test_rejects_non_positive_sentences(self, tmp_path: Path, n):
        src = tmp_path / "in.txt"
        src.write_text(UK_TEXT, encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            main([str(src), "--sentences", n])
        assert exc.value.code == 2

    def test_empty_file(self, tmp_path: Path):
        src = tmp_path / "empty.txt"
        src.write_text("", encoding="utf-8")
        assert main([str(src)]) == 1


APOSTROPHE_VARIANTS = ["\u2019", "\u02bc", "\u2018", "`", "\u00b4", "\u2032", "\u02b9", "\uff07"]


class TestApostrophes:
    @pytest.mark.parametrize("variant", APOSTROPHE_VARIANTS)
    def test_variant_normalized_between_letters(self, variant):
        assert normalize_apostrophes(f"м{variant}ясо, don{variant}t") == "м'ясо, don't"

    def test_ascii_unchanged(self):
        assert normalize_apostrophes("м'ясо") == "м'ясо"

    @pytest.mark.parametrize("variant", APOSTROPHE_VARIANTS)
    def test_quotes_around_words_untouched(self, variant):
        text = f"він сказав {variant}цитата{variant} і пішов, {variant}12{variant}"
        assert normalize_apostrophes(text) == text

    def test_stopwords_are_normalized(self):
        assert all(normalize_apostrophes(w) == w for w in UKRAINIAN_STOPWORDS)

    @pytest.mark.parametrize("variant", ["\u2019", "\u02bc"])
    def test_variant_counts_with_ascii_twin(self, variant):
        # м'ясо and мʼясо must be one term, not two with split weight.
        text = (
            f"Ми купили м'ясо на ринку. Свіже м{variant}ясо коштує дорого. Кіт спав на підвіконні."
        )
        terms = dict(summarize(text, n_sentences=2).top_terms)
        ascii_terms = dict(summarize(text.replace(variant, "'"), n_sentences=2).top_terms)
        assert terms == ascii_terms
        assert "м'ясо" in terms
        assert not any(variant in t for t in terms)


STOPWORD_ONLY = {
    "uk": "Це так. Це і ми. Ви та він. 2024 1999.",
    "en": "It is the one. And so it was. 2024 1999.",
}


class TestEmptyVocabulary:
    @pytest.mark.parametrize("lang", ["uk", "en"])
    def test_stopword_only_text_raises(self, lang):
        with pytest.raises(SummarizationError, match="no content words"):
            summarize(STOPWORD_ONLY[lang], lang=lang)

    def test_is_a_value_error(self):
        assert issubclass(SummarizationError, ValueError)
        with pytest.raises(ValueError):
            summarize(STOPWORD_ONLY["uk"], lang="uk")

    def test_no_fallback_to_first_sentences(self):
        # Must not quietly return something that looks like a summary.
        with pytest.raises(SummarizationError):
            summarize(STOPWORD_ONLY["en"], n_sentences=1)

    def test_cli_prints_message_and_exits_1(self, tmp_path: Path, capsys):
        src = tmp_path / "stop.txt"
        src.write_text(STOPWORD_ONLY["uk"], encoding="utf-8")
        assert main([str(src), "--lang", "uk"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: no content words left")
        assert "Traceback" not in captured.err
