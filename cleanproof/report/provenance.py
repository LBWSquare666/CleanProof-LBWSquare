"""Labels for "who did the check / who did the planning" in the report and the activity log: always written from the actual source recorded in the events, never guessed from the configuration.

provenance values (perception/verifier.py): vlm / fused_conflict (Cosmos 3 took part) · cv (classical vision) · cv_fallback (Cosmos 3 did not respond,
fell back to classical vision) · oracle (simulation ground truth, evaluation only).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

COSMOS_SRC = {"vlm", "fused_conflict"}
CLASSICAL_SRC = {"cv", "cv_fallback"}
FALLBACK_NOTE = "Cosmos 3 not reachable, classical vision used"


def check_kind(provenance: Optional[dict[str, Any]] = None, verifier: str = "", ncalls: int = 0) -> str:
    """The actual source of one check: cosmos / classical / fallback (Cosmos did not respond, fell back to classical vision) / mixed."""
    vals = {str(v) for v in (provenance or {}).values()}
    if not vals:                                   # old records have no provenance: go by the verifier name and whether there were model calls
        return "cosmos" if (ncalls and not str(verifier).startswith("cv")) else "classical"
    has_c, has_f = bool(vals & COSMOS_SRC), "cv_fallback" in vals
    has_cv = "cv" in vals
    if has_c and (has_f or has_cv):
        return "mixed"
    if has_c:
        return "cosmos"
    return "fallback" if has_f else "classical"


def check_name(kind: str) -> str:
    return {"cosmos": "Cosmos 3", "mixed": "Cosmos 3 and classical vision"}.get(kind, "Classical vision")


def planner_name(policy: str) -> str:
    p = str(policy or "")
    if p == "nemotron":
        return "Nemotron 3 Super"
    if p.startswith("fixed"):
        return "Fixed program"
    return "Rule planner"


def check_kinds(inspections: Iterable[dict[str, Any]], verifier: str = "") -> set[str]:
    return {check_kind(i.get("provenance"), i.get("verifier") or verifier, len((i.get("llm") or {}).get("calls") or [])) for i in inspections}
