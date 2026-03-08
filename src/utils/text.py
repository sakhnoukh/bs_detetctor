import re
import unicodedata

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_USER_MENTION = re.compile(r"@\w+")
_WHITESPACE = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub("<URL>", text)
    text = _USER_MENTION.sub("<USER>", text)
    text = _WHITESPACE.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()
