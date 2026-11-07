"""Query-time PII masking: stored text is untouched (and stays searchable); the text
handed back to callers/LLMs is redacted. Regex-based and conservative; not a guarantee."""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_DIGITS = re.compile(r"(?<![\w.])\+?\d(?:[\s().-]{0,2}\d){8,18}(?![\w])")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _digit_run(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group())
    if 13 <= len(digits) <= 19 and _luhn(digits):
        return "[CARD]"
    if 9 <= len(digits) <= 15:
        return "[PHONE]"
    return m.group()


KINDS = ("email", "ssn", "ip", "card", "phone")


def mask_pii(text: str, kinds=KINDS) -> str:
    if "email" in kinds:
        text = _EMAIL.sub("[EMAIL]", text)
    if "ssn" in kinds:
        text = _SSN.sub("[SSN]", text)
    if "ip" in kinds:
        text = _IPV4.sub("[IP]", text)
    if "card" in kinds or "phone" in kinds:
        def sub(m):
            out = _digit_run(m)
            return out if (out == "[CARD]" and "card" in kinds) or (out == "[PHONE]" and "phone" in kinds) else m.group()
        text = _DIGITS.sub(sub, text)
    return text
