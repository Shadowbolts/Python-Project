import re
import unicodedata
from collections.abc import Iterator

WORD_RE = re.compile(r"\w+(?:['’]\w+)*")


def tokenize(text: str) -> Iterator[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in WORD_RE.finditer(normalized):
        yield match.group(0)
