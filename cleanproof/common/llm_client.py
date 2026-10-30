"""Unified model-call entry point (project rule: all model calls go through this module only).

- Backends: tokenfactory (Nebius Token Factory, billed) / local_vllm / local_vllm_exp (local vLLM).
  All three are OpenAI-compatible interfaces.
- Every call writes one JSONL line: time, role, backend, model, request ID, latency, tokens, cost, validation result, error.
  Images are not written to the log, only their count and sha1.
- Billed backends accumulate cost at list price into a cross-process ledger; calls are refused above the limit (BudgetExceeded).
- Structured output: with a pydantic model, use response_format=json_schema and validate again locally;
  JSON wrapped by the model in <think> or a code block can also be extracted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import expand, runs_root, load_secrets, load_yaml
from .jsonl import append_jsonl, locked
from .schemas import response_format


class BudgetExceeded(RuntimeError):
    pass


class ComplianceError(RuntimeError):
    pass


class MissingApiKey(RuntimeError):
    """The backend needs a key but the environment does not have it."""


def classify_model_error(exc: BaseException) -> dict[str, Any]:
    """Model call failed → the reason and what to do, for the operator (no raw exception, address or key). kind: no_key / key_rejected / unreachable / budget / rate_limit / server / other."""
    import openai
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if isinstance(exc, MissingApiKey) or (isinstance(exc, RuntimeError) and "missing environment variable" in str(exc)):
        kind, msg, fix = "no_key", "No Nebius Token Factory API key is set.", "Add it in Settings → Cloud models."
    elif isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)) or status in (401, 403):
        kind, msg, fix = "key_rejected", "Token Factory rejected the API key.", "Check it in Settings → Cloud models."
    elif isinstance(exc, BudgetExceeded):
        kind, msg, fix = "budget", "The spending limit for cloud models has been reached.", "Raise the limit in Settings → Cloud models, or use the rule planner."
    elif isinstance(exc, openai.RateLimitError) or status == 429:
        kind, msg, fix = "rate_limit", "Token Factory is limiting requests right now.", "Wait a minute, then start the job again."
    elif isinstance(exc, openai.APIConnectionError):
        kind, msg, fix = "unreachable", "Cannot reach Nebius Token Factory.", "Check the internet connection, then start the job again."
    elif isinstance(status, int) and status >= 500:
        kind, msg, fix = "server", "Token Factory reported a server error.", "Try again in a few minutes."
    else:
        kind, msg, fix = "other", "The planning model returned an error.", "See Logs for the call that failed."
    return {"kind": kind, "message": msg, "fix": fix, "error_type": name, "http_status": status if isinstance(status, int) else None}


@dataclass
class LLMResult:
    role: str
    backend: str
    model: str
    text: str
    parsed: Optional[BaseModel] = None
    schema_ok: Optional[bool] = None
    schema_error: Optional[str] = None
    request_id: Optional[str] = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class SpendLedger:
    """Cross-process cumulative cost ledger (JSON file + file lock)."""

    def __init__(self, path: str | Path, limit_usd: float):
        self.path = expand(path)
        self.limit_usd = float(limit_usd)

    def _read(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"total_usd": 0.0, "by_model": {}, "calls": 0}

    def total(self) -> float:
        with locked(self.path):
            return float(self._read()["total_usd"])

    def check(self) -> None:
        spent = self.total()
        if spent >= self.limit_usd:
            raise BudgetExceeded(
                f"Token Factory cumulative cost ${spent:.4f} has reached the limit ${self.limit_usd:.2f}; "
                "ask the project owner before changing budget.usd_limit in configs/models.yaml"
            )

    def add(self, model: str, cost_usd: float) -> float:
        with locked(self.path):
            data = self._read()
            data["total_usd"] = float(data["total_usd"]) + cost_usd
            data["by_model"][model] = float(data["by_model"].get(model, 0.0)) + cost_usd
            data["calls"] = int(data.get("calls", 0)) + 1
            data["updated"] = datetime.now(timezone.utc).isoformat()
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
            return data["total_usd"]


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Extract JSON from model output: strip the <think> section and code fences, then take the first complete object."""
    t = _THINK_RE.sub("", text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S)
    if m:
        t = m.group(1).strip()
    fence = _FENCE_RE.search(t)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in the output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("JSON object is incomplete (possibly truncated by max_tokens)")


def _digest_messages(messages: list[dict[str, Any]], max_chars: int = 4000) -> dict[str, Any]:
    """Message digest for the log: keep text (truncated), for images / video only the count and sha1."""
    texts, media = [], []
    for msg in messages:
        content = msg.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for p in parts:
            if p.get("type") == "text":
                texts.append(f"[{msg.get('role')}] {p.get('text', '')}")
            else:
                url = (p.get(p.get("type"), {}) or {}).get("url", "")
                media.append({"type": p.get("type"), "sha1": hashlib.sha1(url.encode()).hexdigest()[:12]})
    joined = "\n".join(texts)
    return {"text": joined[:max_chars], "text_truncated": len(joined) > max_chars, "media": media}


class LLMClient:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        log_path: str | Path | None = None,
        http_client_factory: Optional[Callable[[str], httpx.Client]] = None,
        ledger: Optional[SpendLedger] = None,
    ):
        load_secrets()
        self.cfg = config or load_yaml("models.yaml")
        self.log_path = Path(log_path) if log_path else runs_root() / "llm_calls.jsonl"
        self._http_client_factory = http_client_factory
        budget = self.cfg.get("budget", {})
        self.ledger = ledger or SpendLedger(budget.get("ledger_path", "~/.local/state/cleanproof/spend.json"),
                                            budget.get("usd_limit", 40.0))
        self._clients: dict[str, OpenAI] = {}

    # ---- Config ----
    def role_cfg(self, role: str) -> dict[str, Any]:
        try:
            return dict(self.cfg["roles"][role])
        except KeyError as e:
            raise KeyError(f"no role {role!r} in configs/models.yaml") from e

    def _client(self, backend: str) -> OpenAI:
        if backend not in self._clients:
            b = self.cfg["backends"][backend]
            key_env = b.get("api_key_env")
            api_key = os.environ.get(key_env, "") if key_env else "EMPTY"
            if key_env and not api_key:
                raise MissingApiKey(f"missing environment variable {key_env} (write it to ~/.config/cleanproof/secrets.env)")
            t = self.cfg.get("timeouts", {})
            # On a cluster another user's vLLM may already hold the default port on the same node: CP_BASE_URL_<BACKEND NAME IN UPPERCASE> overrides the address (e.g. CP_BASE_URL_LOCAL_VLLM)
            base_url = os.environ.get(f"CP_BASE_URL_{backend.upper()}", b["base_url"])
            timeout, retries = float(t.get("request_s", 90)), int(t.get("max_retries", 3))
            if b.get("billable", False):                       # cloud decision: tell the operator about a failure quickly (back off and retry at most 2 times); one fault must not stall a job for 6 minutes
                timeout, retries = min(timeout, 60.0), min(retries, 2)
            kwargs: dict[str, Any] = dict(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=retries)
            if self._http_client_factory:
                kwargs["http_client"] = self._http_client_factory(backend)
            self._clients[backend] = OpenAI(**kwargs)
        return self._clients[backend]

    def price(self, model: str) -> dict[str, float]:
        return self.cfg.get("pricing_usd_per_million", {}).get(model, {"input": 0.0, "output": 0.0})

    # ---- Calls ----
    def chat(
        self,
        role: str,
        messages: list[dict[str, Any]],
        schema: Optional[type[BaseModel]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tag: str = "",
        **overrides: Any,
    ) -> LLMResult:
        rc = self.role_cfg(role)
        rc.update(overrides)
        backend, model = rc["backend"], rc["model"]
        billable = bool(self.cfg["backends"][backend].get("billable", False))
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(), "role": role, "tag": tag,
            "backend": backend, "model": model, "billable": billable,
            "request": _digest_messages(messages), "schema": schema.__name__ if schema else None,
        }
        if billable:
            try:
                self.ledger.check()
            except BudgetExceeded as e:                      # a call refused for exceeding the budget is also logged as a row, so it shows in Logs
                record.update(latency_s=0.0, error=f"{type(e).__name__}: {e}", error_kind="budget", http_status=None)
                append_jsonl(self.log_path, record)
                raise

        req: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": int(rc.get("max_tokens", 800))}
        for k in ("temperature", "top_p", "seed"):
            if rc.get(k) is not None:
                req[k] = rc[k]
        extra: dict[str, Any] = {}
        for k in ("top_k", "presence_penalty", "repetition_penalty"):
            if rc.get(k) is not None:
                extra[k] = rc[k]
        if "thinking" in rc:
            # Nemotron family: turn reasoning off, otherwise reasoning tokens fill max_tokens and the JSON gets truncated
            extra["chat_template_kwargs"] = {"enable_thinking": bool(rc["thinking"])}
        if rc.get("media_io_kwargs"):
            extra["media_io_kwargs"] = rc["media_io_kwargs"]
        if extra:
            req["extra_body"] = extra
        if schema is not None:
            req["response_format"] = response_format(schema)
        if tools:
            req["tools"] = tools
            req["tool_choice"] = rc.get("tool_choice", "auto")

        t0 = time.perf_counter()
        try:
            raw = self._client(backend).chat.completions.with_raw_response.create(**req)
            resp = raw.parse()
        except Exception as e:  # log it and re-raise as is; the caller decides how to degrade (usually ask_human)
            ce = classify_model_error(e)
            record.update(latency_s=round(time.perf_counter() - t0, 3), error=f"{type(e).__name__}: {e}", error_kind=ce["kind"],
                          http_status=ce["http_status"])
            append_jsonl(self.log_path, record)
            raise
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = 0.0
        if billable:
            p = self.price(model)
            cost = (pt * p["input"] + ct * p["output"]) / 1e6
            self.ledger.add(model, cost)

        result = LLMResult(
            role=role, backend=backend, model=model, text=text,
            request_id=raw.headers.get("x-request-id") or getattr(resp, "id", None),
            latency_s=latency, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost,
            raw={"finish_reason": choice.finish_reason,
                 "tool_calls": [tc.model_dump() for tc in (choice.message.tool_calls or [])]},
        )
        if schema is not None:
            try:
                result.parsed = schema.model_validate(extract_json(text))
                result.schema_ok = True
            except (ValueError, ValidationError) as e:
                result.schema_ok = False
                result.schema_error = str(e)[:500]

        record.update(
            request_id=result.request_id, latency_s=round(latency, 3), prompt_tokens=pt,
            completion_tokens=ct, cost_usd=round(cost, 6), finish_reason=choice.finish_reason,
            schema_ok=result.schema_ok, schema_error=result.schema_error, response_text=text[:4000],
        )
        append_jsonl(self.log_path, record)
        return result

    # ---- Compliance ----
    def assert_demo_compliance(self) -> None:
        """DEMO_MODE: decision roles must go through Token Factory, and the verifier must be a Cosmos model on local vLLM."""
        problems = []
        for role in ("decision", "fast", "audit"):
            if self.role_cfg(role)["backend"] != "tokenfactory":
                problems.append(f"{role} does not use tokenfactory")
        v = self.role_cfg("verifier")
        if not v["backend"].startswith("local_vllm") or "cosmos" not in v["model"].lower():
            problems.append("verifier is not a Cosmos model on local vLLM")
        if problems:
            raise ComplianceError("; ".join(problems))


def image_part(image_bytes: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    """Encode an image as an OpenAI-compatible image_url content part (base64 data URL)."""
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
