"""from https://github.com/keithito/tacotron"""

import re
from unidecode import unidecode
from .numbers import normalize_numbers


_whitespace_re = re.compile(r"\s+")

_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text: str) -> str:
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text: str) -> str:
    return normalize_numbers(text)


def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text: str) -> str:
    return unidecode(text)


def basic_cleaners(text: str) -> str:
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text: str) -> str:
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text: str) -> str:
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text
