"""GSM8K answer extraction and scoring.

Standard convention: the gold field ends with '#### <number>'. For the
prediction we take the LAST number mentioned, which is the usual heuristic
for chain-of-thought output that reasons before concluding.
"""
import re

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_GOLD_MARKER = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def _normalize(raw: str) -> str:
    return raw.replace(",", "").replace("$", "").rstrip(".")


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return _normalize(matches[-1])


def is_correct(prediction: str, gold: str) -> bool:
    marker = _GOLD_MARKER.search(gold or "")
    gold_value = _normalize(marker.group(1)) if marker else extract_answer(gold)
    pred_value = extract_answer(prediction)
    if gold_value is None or pred_value is None:
        return False
    try:
        return float(pred_value) == float(gold_value)
    except ValueError:
        return pred_value == gold_value
