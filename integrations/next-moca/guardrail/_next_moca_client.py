"""HTTP transport to the Next Moca select endpoint, and the reading of its answer.

This module is a transport. It sends one tool result and the task to
`POST /v1/context/select` and reports whether the server delivered a packet to
use in its place. It never splits, scores, ranks or rewrites text itself: the
only text it ever hands back is the server's `rendered_context`, unchanged.

The API base URL comes from the service environment only. The key is passed in
per call by the caller (forwarded from the gateway's bearer, or the service's
own key in self-hosted mode) and is never stored here. The per-request `config`
relayed by the gateway is not a trust boundary, so nothing in it can redirect
traffic or supply a credential.
"""

from __future__ import annotations

import math
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

DEFAULT_API_BASE = "https://api.nextmoca.com"
ENDPOINT_PATH = "/v1/context/select"

ENV_API_KEY = "NEXT_MOCA_API_KEY"
ENV_API_BASE = "NEXT_MOCA_API_BASE"

# Response header carrying the server's id for this selection (the receipt id).
SELECTION_ID_HEADER = "x-needlepath-selection-id"

USER_AGENT = "next-moca-truefoundry-guardrail/1.0.0"

# Defaults for the fields read from the gateway's `config`.
DEFAULT_OPERATING_POINT = "np-2026-10-r7"
DEFAULT_MAX_CONTEXT_TOKENS = 4000
DEFAULT_MIN_TOKENS = 1500
DEFAULT_MIN_TOKENS_SECURITY = 0  # on a security operating point every tool result is sent
SECURITY_OPERATING_POINTS = frozenset({"np-2026-10-r7"})
SECURITY_OPERATING_POINT_SUFFIX = "r7s-candidate"
DEFAULT_TIMEOUT_MS = 3000

# Bounds. `timeout_ms` is the deadline for all calls of one gateway request, and its
# ceiling leaves margin under the gateway's default 10 s guardrail timeout.
MIN_TIMEOUT_MS = 100
MAX_TIMEOUT_MS = 8000
MIN_MAX_CONTEXT_TOKENS = 256
MAX_MAX_CONTEXT_TOKENS = 128_000
MAX_CONCURRENT_CALLS = 8  # per gateway request
MAX_SERVICE_CONNECTIONS = 64  # per service process

# Outcomes on which the server delivered a packet that replaces the tool result.
DELIVERY_OUTCOMES = ("engaged", "sanitized_fallback", "blocked")

# Fields a response must carry to be read at all.
REQUIRED_RESPONSE_FIELDS = (
    "request_id",
    "rendered_context",
    "tokens_before",
    "tokens_after",
    "tokens_saved",
    "records_available",
    "records_selected",
    "fallback_used",
    "engine_latency_ms",
)

DECISION_APPLIED = "applied"
DECISION_PASSTHROUGH = "passthrough"


def estimate_tokens(text: str) -> int:
    """Size estimate for min_tokens: one token per four characters, rounded up."""
    return math.ceil(len(text) / 4)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Per-request settings resolved from the gateway `config` plus the environment."""

    operating_point: str = DEFAULT_OPERATING_POINT
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    min_tokens: int = DEFAULT_MIN_TOKENS
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    validate_only: bool = False


def _int_setting(config: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    value = config.get(key)
    if value is None or isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _bool_setting(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return default


def is_security_operating_point(label: str) -> bool:
    return label in SECURITY_OPERATING_POINTS or label.endswith(SECURITY_OPERATING_POINT_SUFFIX)


def default_min_tokens(operating_point: str) -> int:
    return DEFAULT_MIN_TOKENS_SECURITY if is_security_operating_point(operating_point) else DEFAULT_MIN_TOKENS


def resolve_settings(config: Optional[dict[str, Any]]) -> Settings:
    """Read the documented fields from the gateway `config`. Unknown keys are ignored.

    An invalid value falls back to its default rather than failing the request.
    """
    cfg = config if isinstance(config, dict) else {}
    operating_point = cfg.get("operating_point")
    if not isinstance(operating_point, str) or not operating_point.strip():
        operating_point = DEFAULT_OPERATING_POINT
    operating_point = operating_point.strip()
    return Settings(
        operating_point=operating_point,
        max_context_tokens=_int_setting(
            cfg, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS, MIN_MAX_CONTEXT_TOKENS, MAX_MAX_CONTEXT_TOKENS
        ),
        min_tokens=_int_setting(cfg, "min_tokens", default_min_tokens(operating_point), 0, 10_000_000),
        timeout_ms=_int_setting(cfg, "timeout_ms", DEFAULT_TIMEOUT_MS, MIN_TIMEOUT_MS, MAX_TIMEOUT_MS),
        validate_only=_bool_setting(cfg, "validate_only", False),
    )


def api_key() -> str:
    """The service's own key (self-hosted mode only)."""
    return os.environ.get(ENV_API_KEY, "").strip()


def api_base() -> str:
    """Base URL from the environment. https only: keys are never sent in plaintext."""
    base = (os.environ.get(ENV_API_BASE, "").strip() or DEFAULT_API_BASE).rstrip("/")
    if urlparse(base).scheme != "https":
        raise ValueError(f"{ENV_API_BASE} must be an https:// URL")
    return base


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def new_request_id() -> str:
    """Random correlation id. Carries no gateway, user or content identifiers."""
    return f"tfy-{uuid.uuid4().hex}"


def build_text_request(*, request_id: str, text: str, task_prompt: str, settings: Settings) -> dict[str, Any]:
    """The request body for the public simple-text ingress.

    The operating point goes inside `budget`; a top-level `operating_point` is ignored by the API.
    """
    return {
        "request_id": request_id,
        "text": text,
        "task": {"prompt": task_prompt},
        "budget": {
            "max_context_tokens": settings.max_context_tokens,
            "operating_point": settings.operating_point,
            "mode": "fixed",
            "autochunk": True,
        },
        "render": True,
        "render_format": "plain",
        "return_per_record": False,
    }


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """What to do with one tool result, derived only from fields the server set."""

    decision: str
    reason: str
    rendered: Optional[str] = None
    outcome: Optional[str] = None
    selection_id: Optional[str] = None
    receipt: dict[str, Any] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return self.decision == DECISION_APPLIED


def decide_response(request_id: str, response: Any) -> Decision:
    """Decide applied or passthrough from the server's answer.

    Applied on `engaged` with non-empty rendered text and no fallback or safety
    signal to the contrary, and on the two security deliveries
    (`sanitized_fallback`, `blocked`) whenever their rendered text is non-empty:
    the original text is never restored once the server has delivered a
    screened packet. Everything else, including a malformed body, passes the
    original tool result through.
    """
    if not isinstance(response, dict):
        return Decision(DECISION_PASSTHROUGH, "malformed_response:not_an_object")
    if any(f not in response for f in REQUIRED_RESPONSE_FIELDS):
        return Decision(DECISION_PASSTHROUGH, "malformed_response:missing_fields")
    if response.get("request_id") != request_id:
        return Decision(DECISION_PASSTHROUGH, "malformed_response:request_id_mismatch")
    rendered = response.get("rendered_context")
    if not isinstance(rendered, str):
        return Decision(DECISION_PASSTHROUGH, "malformed_response:rendered_context_not_text")

    outcome = response.get("outcome")
    outcome_str = outcome if isinstance(outcome, str) else None
    safety_raw = response.get("safety")
    safety: dict[str, Any] = safety_raw if isinstance(safety_raw, dict) else {}

    if outcome in ("sanitized_fallback", "blocked"):
        if not rendered.strip():
            return Decision(DECISION_PASSTHROUGH, "empty_selection", outcome=outcome_str)
        return Decision(DECISION_APPLIED, f"applied:{outcome}", rendered=rendered, outcome=outcome_str)
    if outcome == "stood_down":
        return Decision(DECISION_PASSTHROUGH, "stood_down", outcome=outcome_str)
    if outcome == "escalated":
        return Decision(DECISION_PASSTHROUGH, "escalated", outcome=outcome_str)
    if response.get("fallback_used") is True:
        return Decision(DECISION_PASSTHROUGH, "fallback_used", outcome=outcome_str)
    if safety.get("fallback_required") is True or safety.get("selection_safe") is False:
        return Decision(DECISION_PASSTHROUGH, "unsafe_selection", outcome=outcome_str)
    if response.get("selection_error"):
        return Decision(DECISION_PASSTHROUGH, "selection_error", outcome=outcome_str)
    if outcome is None:
        return Decision(DECISION_PASSTHROUGH, "no_outcome")
    if outcome != "engaged":
        return Decision(DECISION_PASSTHROUGH, "unknown_outcome", outcome=outcome_str)
    gate = response.get("gate")
    if isinstance(gate, dict) and gate.get("engaged") is False:
        return Decision(DECISION_PASSTHROUGH, "not_engaged", outcome=outcome_str)
    if not rendered.strip():
        return Decision(DECISION_PASSTHROUGH, "empty_selection", outcome=outcome_str)
    return Decision(DECISION_APPLIED, "applied", rendered=rendered, outcome=outcome_str)


# Shapes a server-supplied value must have before it is logged or put in a message.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_SAFE_OUTCOME = re.compile(r"^[a-z_]{1,40}$")


def safe_id(value: Any, key: str) -> Optional[str]:
    """A short identifier from the server, or None if it has an unexpected shape or carries the key."""
    if not isinstance(value, str) or not _SAFE_ID.match(value) or (key and key in value):
        return None
    return value


def safe_outcome(value: Optional[str], key: str) -> Optional[str]:
    if value is None:
        return None
    if not _SAFE_OUTCOME.match(value) or (key and key in value):
        return "unrecognized"
    return value


def receipt_of(response: Any, key: str = "") -> dict[str, Any]:
    """Identity and counts worth logging. Never any text: numbers and booleans as sent,
    strings only when they are short identifiers that do not carry the key."""
    if not isinstance(response, dict):
        return {}
    keys = (
        "policy_version",
        "outcome",
        "fallback_used",
        "records_available",
        "records_selected",
        "tokens_before",
        "tokens_after",
        "tokens_saved",
        "engine_latency_ms",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k not in response:
            continue
        v = response[k]
        if isinstance(v, (bool, int, float)) or v is None:
            out[k] = v
        else:
            cleaned = safe_id(v, key)
            if cleaned is not None:
                out[k] = cleaned
    return out


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


async def select_one(
    client: httpx.AsyncClient,
    *,
    url: str,
    key: str,
    text: str,
    task_prompt: str,
    settings: Settings,
) -> Decision:
    """One call for one tool result. Never raises: every failure is a passthrough decision."""
    request_id = new_request_id()
    body = build_text_request(request_id=request_id, text=text, task_prompt=task_prompt, settings=settings)
    try:
        resp = await client.post(
            url,
            json=body,
            headers={
                "authorization": f"Bearer {key}",
                "content-type": "application/json",
                "user-agent": USER_AGENT,
            },
            timeout=settings.timeout_ms / 1000.0,
        )
    except httpx.TimeoutException:
        return Decision(DECISION_PASSTHROUGH, "timeout")
    except Exception as exc:  # noqa: BLE001 - fail open on any transport error
        return Decision(DECISION_PASSTHROUGH, f"transport_error:{type(exc).__name__}")

    selection_id = safe_id(resp.headers.get(SELECTION_ID_HEADER), key)
    if resp.status_code < 200 or resp.status_code >= 300:
        return Decision(DECISION_PASSTHROUGH, f"http_{resp.status_code}", selection_id=selection_id)
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return Decision(DECISION_PASSTHROUGH, "malformed_response:not_json", selection_id=selection_id)

    decision = decide_response(request_id, payload)
    if decision.applied and key and key in (decision.rendered or ""):
        # Never put a credential into the model's context, whatever the server sent.
        decision = Decision(DECISION_PASSTHROUGH, "malformed_response:credential_in_render")
    decision.outcome = safe_outcome(decision.outcome, key)
    decision.selection_id = selection_id
    decision.receipt = receipt_of(payload, key)
    return decision
