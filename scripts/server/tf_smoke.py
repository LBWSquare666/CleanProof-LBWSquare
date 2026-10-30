"""Token Factory smoke test (PLAN section 12 step 3).

Call decision (Nemotron 3 Super) and fast (Nemotron 3.5 Lightning) through LLMClient, with the real decision prompt and the
Decision schema, testing thinking on and off once each. Every call writes runs/llm_calls.jsonl as usual and counts toward the spend ledger;
this script also writes a summary to runs/w0/tf_smoke.json.

  .venv/bin/python scripts/server/tf_smoke.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cleanproof.agent.policy import PROMPT_DIR  # noqa: E402
from cleanproof.agent.force_table import load_force_table  # noqa: E402
from cleanproof.agent.state import TaskState  # noqa: E402
from cleanproof.common.config import REPO_ROOT, load_yaml  # noqa: E402
from cleanproof.common.llm_client import LLMClient  # noqa: E402
from cleanproof.common.schemas import Decision, TaskSpec, VerifierZoneResult  # noqa: E402
from cleanproof.common.zones import ZoneGrid  # noqa: E402

# with thinking on, reasoning tokens also count toward max_tokens; leave plenty of headroom and see whether it still produces valid JSON
MAX_TOKENS = {False: None, True: 4000}


def sample_state() -> TaskState:
    """The same small scenario as tests/test_agent.py: B2 has residue, B3 is clean, C2 is uncertain."""
    st = TaskState(TaskSpec(zones=["B2", "B3", "C2"], max_passes_per_zone=2), ZoneGrid())
    st.update_inspection({
        "B2": VerifierZoneResult(zone_id="B2", status="residue", confidence=0.9),
        "B3": VerifierZoneResult(zone_id="B3", status="clean", confidence=0.9),
        "C2": VerifierZoneResult(zone_id="C2", status="uncertain", confidence=0.5),
    }, {})
    return st


def main() -> int:
    client = LLMClient()
    st = sample_state()
    system = (PROMPT_DIR / "decision_system.md").read_text(encoding="utf-8")
    user = json.dumps(st.summary(load_force_table().rules_for(st.task.surface)), ensure_ascii=False)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    spent0 = client.ledger.total()
    rows = []
    for role in ("decision", "fast"):
        for thinking in (False, True):
            ov = {"thinking": thinking}
            if MAX_TOKENS[thinking]:
                ov["max_tokens"] = MAX_TOKENS[thinking]
            row = {"role": role, "thinking": thinking, "model": client.role_cfg(role)["model"]}
            try:
                r = client.chat(role, messages, schema=Decision, tag=f"tf_smoke/{role}/think={int(thinking)}", **ov)
                row.update(request_id=r.request_id, latency_s=round(r.latency_s, 2),
                           prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                           cost_usd=round(r.cost_usd, 6), finish_reason=r.raw.get("finish_reason"),
                           schema_ok=r.schema_ok, schema_error=r.schema_error,
                           decision=r.parsed.model_dump(mode="json") if r.parsed else None)
            except Exception as e:  # record the failure and keep testing the other combinations
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            print(f"{role:8s} thinking={thinking!s:5s} ok={row.get('schema_ok')!s:5s} "
                  f"lat={row.get('latency_s', '-')}s tok={row.get('prompt_tokens', '-')}/{row.get('completion_tokens', '-')} "
                  f"${row.get('cost_usd', 0):.5f} finish={row.get('finish_reason')} {row.get('error', '')}")

    out = REPO_ROOT / "runs" / "w0" / "tf_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "spent_this_run_usd": round(client.ledger.total() - spent0, 6),
               "ledger_total_usd": round(client.ledger.total(), 6), "calls": rows}
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"spend this run ${summary['spent_this_run_usd']:.5f}, ledger total ${summary['ledger_total_usd']:.5f} → {out}")
    return 0 if all(r.get("schema_ok") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
