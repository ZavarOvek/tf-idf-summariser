"""tfidfsum — extractive TF-IDF summarization for Ukrainian and English."""
from .sentences import detect_lang, split_sentences
from .summarizer import Summary, summarize

__all__ = ["Summary", "detect_lang", "split_sentences", "summarize"]
__version__ = "0.1.0"
