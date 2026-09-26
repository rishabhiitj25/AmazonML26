import re
import unicodedata

from anyascii import anyascii

_PLACEHOLDER = re.compile(r"<[A-Za-z_]+>")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_MISSING_COMPONENTS = {"null", "none", "n/a", "na", "nil", "-"}


def _symbol_to_space(c: str) -> str:
    """Non-ASCII punctuation/symbols are separators, not text: anyascii would spell some of them out
    ("N°12" -> "ndeg12", "¢" -> "c"). Letters, marks (accents) and digits are kept for romanisation."""
    return " " if ord(c) > 127 and unicodedata.category(c)[0] in "PS" else c


def to_latin(s: str) -> str:
    """NFKC, non-ASCII punctuation/symbols -> space, then romanise non-ASCII text (folds accents), lowercase."""
    s = unicodedata.normalize("NFKC", s)
    if not s.isascii():
        s = anyascii("".join(map(_symbol_to_space, s)))
    return s.lower()


def clean_tokens(s: str) -> str:
    s = s.replace("&", " and ")
    return _NON_ALNUM.sub(" ", s).strip()


def norm_name(raw: str) -> str:
    return clean_tokens(to_latin(_PLACEHOLDER.sub(" ", raw)))


def norm_addr(raw: str) -> str:
    s = _PLACEHOLDER.sub(" ", raw)
    parts = [p for p in s.split(",") if p.strip().lower() not in _MISSING_COMPONENTS]
    return clean_tokens(to_latin(",".join(parts)))


def digits(norm_text: str) -> str:
    return " ".join(_DIGITS.findall(norm_text))


def has_non_latin(raw: str) -> bool:
    return any(ord(c) > 0x24F and unicodedata.category(c).startswith("L") for c in raw)
