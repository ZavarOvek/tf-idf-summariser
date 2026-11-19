"""tfidfsum — extractive TF-IDF summarization for Ukrainian and English."""

from .sentences import detect_lang, split_sentences
from .summarizer import SummarizationError, Summary, summarize

__all__ = ["SummarizationError", "Summary", "detect_lang", "split_sentences", "summarize"]
__version__ = "0.1.1"
