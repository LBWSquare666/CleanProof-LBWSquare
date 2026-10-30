"""llm_client unit tests: calls go through the real openai SDK, with only the network layer replaced by httpx.MockTransport."""

import json

import httpx
import pytest

from cleanproof.common.config import load_yaml
from cleanproof.common.jsonl import read_jsonl
from cleanproof.common.llm_client import (
    BudgetExceeded,
    ComplianceError,
    LLMClient,
    SpendLedger,
    extract_json,
)
from cleanproof.common.schemas import Decision

DECISION = {
    "action": "wipe_zone", "zone_ids": ["B2"], "force_n": 10, "pattern": "zigzag",
    "passes": 1, "reason": "residue remains", "expected_outcome": "B2 clean",
}


def make_client(tmp_path, content, usage=(1000, 200), captured=None, limit=40.0):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(request.content))
        body = {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                      "total_tokens": sum(usage)},
        }
        return httpx.Response(200, json=body, headers={"x-request-id": "req-123"})

    cfg = load_yaml("models.yaml")
    ledger = SpendLedger(tmp_path / "spend.json", limit)
    return LLMClient(
        config=cfg, log_path=tmp_path / "calls.jsonl", ledger=ledger,
        http_client_factory=lambda backend: httpx.Client(transport=httpx.MockTransport(handler)),
    ), ledger


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key-not-real")


def test_structured_decision_logs_cost_and_request(tmp_path):
    captured = []
    client, ledger = make_client(tmp_path, json.dumps(DECISION), captured=captured)
    res = client.chat("decision", [{"role": "user", "content": "plan"}], schema=Decision, tag="t")
    assert res.schema_ok and res.parsed.zone_ids == ["B2"]
    assert res.request_id == "req-123"
    # super: 1000*0.30/1e6 + 200*0.90/1e6
    assert res.cost_usd == pytest.approx(0.00048)
    assert ledger.total() == pytest.approx(0.00048)
    req = captured[0]
    assert req["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert req["response_format"]["type"] == "json_schema"
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    log = read_jsonl(tmp_path / "calls.jsonl")
    assert log[-1]["schema_ok"] is True and log[-1]["backend"] == "tokenfactory"
    assert "test-key-not-real" not in json.dumps(log)


def test_invalid_json_is_reported_not_raised(tmp_path):
    client, _ = make_client(tmp_path, '{"action": "wipe_zone", "zone_ids": ["B2"]')
    res = client.chat("decision", [{"role": "user", "content": "plan"}], schema=Decision)
    assert res.schema_ok is False and "truncated" in res.schema_error


def test_semantic_validation_rejects_missing_params(tmp_path):
    bad = dict(DECISION, force_n=None)
    client, _ = make_client(tmp_path, json.dumps(bad))
    res = client.chat("decision", [{"role": "user", "content": "plan"}], schema=Decision)
    assert res.schema_ok is False


def test_budget_blocks_billable_calls(tmp_path):
    client, ledger = make_client(tmp_path, json.dumps(DECISION), limit=0.0005)
    client.chat("decision", [{"role": "user", "content": "plan"}], schema=Decision)
    ledger.add("x", 0.001)
    with pytest.raises(BudgetExceeded):
        client.chat("decision", [{"role": "user", "content": "plan"}], schema=Decision)


def test_local_backend_not_billed_and_sampling_params(tmp_path):
    captured = []
    client, ledger = make_client(tmp_path, '{"zones": []}', captured=captured)
    client.chat("verifier", [{"role": "user", "content": "inspect"}])
    assert ledger.total() == 0.0
    assert captured[0]["top_k"] == 20 and captured[0]["presence_penalty"] == 1.5


def test_extract_json_variants():
    assert extract_json('<think>hmm {"a": 0}</think>{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('<answer>{"a": 3}</answer>') == {"a": 3}
    assert extract_json('text before {"a": {"b": "}"}} after') == {"a": {"b": "}"}}


def test_demo_compliance(tmp_path):
    client, _ = make_client(tmp_path, "{}")
    client.assert_demo_compliance()
    client.cfg["roles"]["decision"]["backend"] = "local_vllm"
    with pytest.raises(ComplianceError):
        client.assert_demo_compliance()


def test_budget_rejected_call_is_logged_and_classified(tmp_path):
    """: A call refused for exceeding the budget is also logged as a row (with error_kind), it does not vanish."""
    from cleanproof.common.llm_client import classify_model_error
    client, ledger = make_client(tmp_path, json.dumps(DECISION), limit=0.01)
    ledger.add("m", 0.02)
    with pytest.raises(BudgetExceeded) as ei:
        client.chat("decision", [{"role": "user", "content": "plan"}], tag="over")
    rows = read_jsonl(tmp_path / "calls.jsonl")
    assert len(rows) == 1 and rows[0]["error_kind"] == "budget" and rows[0]["tag"] == "over" and "BudgetExceeded" in rows[0]["error"]
    ce = classify_model_error(ei.value)
    assert ce["kind"] == "budget" and "Settings" in ce["fix"] and "Traceback" not in ce["message"]


def test_failed_calls_are_classified_for_operators(tmp_path, monkeypatch):
    """: Missing key / rejected key / cannot connect: each gives one sentence the operator understands, with what to do."""
    from cleanproof.common.llm_client import classify_model_error

    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    client, _ = make_client(tmp_path, "{}")
    with pytest.raises(Exception) as ei:
        client.chat("decision", [{"role": "user", "content": "x"}])
    assert classify_model_error(ei.value)["kind"] == "no_key"
    monkeypatch.setenv("NEBIUS_API_KEY", "bad")

    def handler(request):
        return httpx.Response(401, json={"error": {"message": "invalid"}})
    cfg = load_yaml("models.yaml")
    c2 = LLMClient(config=cfg, log_path=tmp_path / "c2.jsonl", ledger=SpendLedger(tmp_path / "s2.json", 40),
                   http_client_factory=lambda b: httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(Exception) as ei:
        c2.chat("decision", [{"role": "user", "content": "x"}])
    ce = classify_model_error(ei.value)
    assert ce["kind"] == "key_rejected" and "rejected the API key" in ce["message"] and "Settings → Cloud models" in ce["fix"]
    row = read_jsonl(tmp_path / "c2.jsonl")[0]
    assert row["http_status"] == 401 and row["error_kind"] == "key_rejected"
