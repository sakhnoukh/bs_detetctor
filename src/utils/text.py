"""
Text normalization utilities.
"""
import re
import unicodedata

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_USER_MENTION = re.compile(r"@\w+")
_WHITESPACE = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Apply minimal, reversible normalization to a text string."""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Strip HTML tags
    text = _HTML_TAG.sub(" ", text)
    # Replace URLs
    text = _URL.sub("<URL>", text)
    # Replace @mentions
    text = _USER_MENTION.sub("<USER>", text)
    # Collapse repeated spaces/tabs (but preserve single newlines)
    text = _WHITESPACE.sub(" ", text)
    # Collapse 3+ newlines to 2
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()
