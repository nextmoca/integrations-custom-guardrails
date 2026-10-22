"""FastAPI app for the Next Moca custom-guardrail wrapper.

On the llm_input hook, sends the tool results in a chat-completions request to
the Next Moca API (POST https://api.nextmoca.com/v1/context/select) with the
latest user message as the task, and returns the request with each tool result
replaced by the context Next Moca rendered for it.

Endpoints:
    GET  /                       health check (open)
    GET  /health                 health check (open)
    POST /select-input           llm_input hook, Operation: Mutate
                                 (or Operation: Validate with config validate_only: true)
    GET  /debug/loaded-config    diagnostics (no secrets)

Two modes, chosen by WRAPPER_API_KEY:

    Forwarded key (WRAPPER_API_KEY unset; one hosted endpoint for many customers).
        Each customer puts their own Next Moca API key in the TrueFoundry guardrail
        config under Auth Data > Custom Bearer Auth. The gateway sends it as
        `Authorization: Bearer <key>` and the service forwards it to Next Moca for
        that request only. It is never logged, echoed, stored or cached. A request
        without a bearer key passes through unchanged and makes no call.

    Self-hosted (WRAPPER_API_KEY set).
        The gateway's bearer must equal WRAPPER_API_KEY (401 otherwise), and the
        Next Moca key comes from NEXT_MOCA_API_KEY on the service. Without it,
        requests pass through unchanged.

Response contract:
    Mutate     -> HTTP 200 + {"verdict": true, "transformed": <bool>, "result": <requestBody>}
    Validate   -> HTTP 200 + {"verdict": true, "message": "<selection ids>"}
    The rail never denies. Next Moca errors and timeouts pass the request through unchanged.
"""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import re
from typing import Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

load_dotenv()

from guardrail import _next_moca_client as nm  # noqa: E402
from entities import InputGuardrailRequest, MutateGuardrailResponse, ValidateGuardrailResponse  # noqa: E402
from guardrail.select_input import close_client, open_client, select_input  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("next-moca-guardrails-tfy")

# Placeholder shipped in .env.example; never accepted as a real key.
_PLACEHOLDER_API_KEYS = frozenset({"change-me-to-a-random-string"})


def _wrapper_api_key() -> str:
    return os.environ.get("WRAPPER_API_KEY", "").strip()


def _self_hosted() -> bool:
    """Self-hosted mode: WRAPPER_API_KEY is set to a real value."""
    key = _wrapper_api_key()
    return bool(key) and key not in _PLACEHOLDER_API_KEYS


# A bearer token that can be forwarded: printable ASCII, no spaces, bounded length.
_FORWARDABLE_TOKEN = re.compile(r"^[\x21-\x7e]{1,512}$")


def _bearer_token(request: Request) -> Optional[str]:
    """The token of exactly one `Authorization: Bearer <token>` header, else None.

    Several Authorization headers are ambiguous and count as none.
    """
    values = request.headers.getlist("authorization")
    if len(values) != 1:
        return None
    scheme, _, token = values[0].strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _check_wrapper_bearer(request: Request) -> None:
    """Self-hosted mode only: the gateway's bearer must equal WRAPPER_API_KEY."""
    provided = _bearer_token(request)
    if provided is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    if not hmac.compare_digest(provided.encode(), _wrapper_api_key().encode()):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def resolve_next_moca_key(request: Request) -> Optional[str]:
    """The Next Moca key for this request, or None (the request then passes through).

    Never logged, echoed or stored: it lives only in this request's call stack.
    """
    if _wrapper_api_key() in _PLACEHOLDER_API_KEYS:
        raise HTTPException(status_code=503, detail="WRAPPER_API_KEY is the example placeholder")
    if _self_hosted():
        _check_wrapper_bearer(request)
        return nm.api_key() or None
    # Forwarded-key mode never falls back to a key held by the service.
    token = _bearer_token(request)
    if token is not None and _FORWARDABLE_TOKEN.match(token):
        return token
    return None


def require_debug_access(request: Request) -> None:
    """/debug carries no secrets; in self-hosted mode it still requires the wrapper bearer."""
    if _wrapper_api_key() in _PLACEHOLDER_API_KEYS:
        raise HTTPException(status_code=503, detail="WRAPPER_API_KEY is the example placeholder")
    if _self_hosted():
        _check_wrapper_bearer(request)


log.info("mode: %s", "self-hosted" if _self_hosted() else "forwarded key")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await open_client()
    try:
        yield
    finally:
        await close_client()


app = FastAPI(
    title="next-moca-guardrails-tfy",
    description="Next Moca context selection for tool results, behind the TrueFoundry custom-guardrail contract.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """A malformed gateway envelope is an infrastructure error (non-2xx). The body names
    only the top-level envelope fields that failed (requestBody, context, config) and
    never echoes request content, including dictionary keys below those fields."""
    envelope = {"requestBody", "context", "config"}
    fields = sorted(
        {
            f"body.{loc[1]}" if len(loc) > 1 and loc[1] in envelope else "body"
            for loc in (tuple(e.get("loc", ())) for e in exc.errors())
        }
    )
    return JSONResponse(status_code=422, content={"error": "invalid guardrail request", "fields": fields})


@app.get("/")
@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


async def select_input_route(
    request: InputGuardrailRequest,
    next_moca_key: Optional[str] = Depends(resolve_next_moca_key),
) -> Union[MutateGuardrailResponse, ValidateGuardrailResponse]:
    return await select_input(request, next_moca_key)


RAIL_ROUTES: dict[str, object] = {
    "/select-input": select_input_route,
}

for path, fn in RAIL_ROUTES.items():
    app.add_api_route(path, endpoint=fn, methods=["POST"], status_code=200)


@app.get("/debug/loaded-config", dependencies=[Depends(require_debug_access)])
async def debug_loaded_config() -> dict:
    """What the running service has loaded. Never returns secret values."""
    try:
        base = nm.api_base()
    except ValueError as exc:
        base = f"invalid: {exc}"
    return {
        "routes": {
            "input": [p for p in RAIL_ROUTES if p.endswith("-input")],
            "output": [p for p in RAIL_ROUTES if p.endswith("-output")],
        },
        "next_moca_api_base": base,
        "mode": "self-hosted" if _self_hosted() else "forwarded key",
        "next_moca_api_key_set": bool(nm.api_key()),
        "defaults": {
            "operating_point": nm.DEFAULT_OPERATING_POINT,
            "max_context_tokens": nm.DEFAULT_MAX_CONTEXT_TOKENS,
            "min_tokens": {"security operating points": nm.DEFAULT_MIN_TOKENS_SECURITY, "other": nm.DEFAULT_MIN_TOKENS},
            "timeout_ms": nm.DEFAULT_TIMEOUT_MS,
            "validate_only": False,
        },
        "wrapper_auth_enabled": _self_hosted(),
        "wrapper_version": os.environ.get("BUILD_REF", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
