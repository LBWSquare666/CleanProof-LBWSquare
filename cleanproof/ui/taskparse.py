"""Zone range in the task box: a sentence the operator wrote → which zones to wipe.

Supports named zones ("B2 and C3 only"), ranges (A1-A3, A1 to A3, A1–A3, A1-3), exclusions (not / don't / do not / except / avoid / without / skip /
other than / leave out ...) and "the rest" (the rest / everything else). Only zone names that exist in this site's grid are recognised.
A phrasing that is not recognised ("left half") falls back to the whole table and says so; a negation without a parsed zone requires confirmation before the job starts, and never silently wipes everything.

static/operator.html in the frontend has zonesInTask, the JS version of the same rule: change one and the other must change too (tests/test_taskparse.py has the sentence table, and the frontend is checked against the same sentences).
"""

from __future__ import annotations

import re
from typing import Any

ZONE = r"(?:[A-Za-z][1-9][0-9]?|[A-Z] [1-9][0-9]?)"
# range: A1-A3 / A1 to A3 / A1–A3 / A1 through C3 / A1-3 (when the second end omits the row letter = the same row)
_RANGE = re.compile(r"\b([A-Za-z]) ?([1-9][0-9]?)\s*(?:-|–|—|to|through|thru|until)\s*([A-Za-z])? ?([1-9][0-9]?)\b", re.I)
_ZONE = re.compile(r"\b(?:([A-Za-z])([1-9][0-9]?)|([A-Z]) ([1-9][0-9]?))\b")
_NEG = re.compile(r"\b(?:not|don'?t|dont|doesn'?t|never|except(?: for)?|excluding|exclude|avoid(?:ing)?|without|skip(?:ping)?|"
                  r"other than|apart from|aside from|besides|leave out|leaving out|ignore|ignoring|stay away from|keep away from|steer clear of)\b", re.I)
_POS_RESET = re.compile(r"\b(?:but|then|only|also|and clean|and wipe)\b", re.I)
_REST = re.compile(r"\b(?:the )?rest\b|\beverything(?: else)?\b|\ball (?:the )?(?:other|others|remaining|zones|zone)\b|\bthe others\b|\bremaining\b|"
                   r"\bwhole\b|\bentire\b|\bfull\b|\bany(?:thing)? else\b", re.I)
_SPATIAL = re.compile(r"\b(?:left|right|top|bottom|upper|lower|front|back|rear|middle|center|centre|corner|corners|edge|edges|half|side|"
                      r"row|rows|column|columns|col)\b", re.I)
_CLAUSE = re.compile(r"[.,;!?\n]|\band then\b|\bthen\b")


def _ids(rows: int, cols: int) -> set[str]:
    return {f"{chr(65 + r)}{c + 1}" for r in range(rows) for c in range(cols)}


def _rect(a: tuple[int, int], b: tuple[int, int]) -> list[str]:
    (r0, c0), (r1, c1) = a, b
    return [f"{chr(65 + r)}{c + 1}" for r in range(min(r0, r1), max(r0, r1) + 1) for c in range(min(c0, c1), max(c0, c1) + 1)]


def parse_task(text: str, rows: int, cols: int) -> dict[str, Any]:
    """Returns {"zones": zones to wipe (empty = the whole surface), "mode": all / only / except, "included", "excluded", "notes", "confirm", "label"}."""
    valid = _ids(rows, cols)
    text = (text or "").replace("’", "'")
    inc: list[str] = []
    exc: list[str] = []
    scope_all = False
    neg_seen = False
    for clause in _CLAUSE.split(text):
        # first replace ranges with a placeholder zone list, then handle negations and zones in order of appearance
        items: list[tuple[int, str, Any]] = []
        used: list[tuple[int, int]] = []
        for m in _RANGE.finditer(clause):
            r0, c0 = ord(m.group(1).upper()) - 65, int(m.group(2)) - 1
            r1, c1 = (ord(m.group(3).upper()) - 65 if m.group(3) else r0), int(m.group(4)) - 1
            zs = [z for z in _rect((r0, c0), (r1, c1)) if z in valid]
            if f"{chr(65 + r0)}{c0 + 1}" in valid and f"{chr(65 + r1)}{c1 + 1}" in valid:
                items.append((m.start(), "z", zs))
                used.append((m.start(), m.end()))
        for m in _ZONE.finditer(clause):
            if any(a <= m.start() < b for a, b in used):
                continue
            z = (m.group(1) or m.group(3)).upper() + (m.group(2) or m.group(4))
            if z in valid:
                items.append((m.start(), "z", [z]))
        for m in _NEG.finditer(clause):
            items.append((m.start(), "neg", None))
        for m in _POS_RESET.finditer(clause):
            items.append((m.start(), "pos", None))
        for m in _REST.finditer(clause):
            items.append((m.start(), "rest", None))
        neg = False
        for _, kind, val in sorted(items, key=lambda x: x[0]):
            if kind == "neg":
                neg = neg_seen = True
            elif kind == "pos":
                neg = False
            elif kind == "rest":
                if not neg:
                    scope_all = True
            else:
                tgt = exc if neg else inc
                for z in val:
                    if z not in tgt:
                        tgt.append(z)
    notes: list[str] = []
    confirm = False
    spatial = bool(_SPATIAL.search(text)) and not (inc or exc)
    if exc:
        base = [z for z in sorted(valid, key=lambda z: (z[0], int(z[1:]))) if z not in exc] if (scope_all or not inc) else [z for z in inc]
        zones = [z for z in base if z not in exc]
        mode = "except"
        label = ("All zones except " + ", ".join(exc)) if (scope_all or not inc) else (", ".join(zones) + " (not " + ", ".join(exc) + ")")
    elif inc:
        zones, mode = list(inc), "only"
        label = ", ".join(zones)
    else:
        zones, mode, label = [], "all", ""
        if neg_seen:
            confirm = True
            notes.append("The task says to leave something out, but no zone was recognized. Name the zones, for example "
                         "\"everything except B3\". Otherwise the whole surface is cleaned.")
        elif spatial:
            notes.append("Positions such as \"left half\" are not understood, so the whole surface is cleaned. Name zones instead, for example A1-A3.")
    if mode != "all" and not zones:
        notes.append("Every zone is left out, so there is nothing to clean.")
    return {"zones": zones, "mode": mode, "included": inc, "excluded": exc, "notes": notes, "confirm": confirm, "label": label}
