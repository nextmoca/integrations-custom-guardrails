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
    GET  /debug/loaded-config    bearer-auth gated diagnostics

A shared bearer token (WRAPPER_API_KEY) gates the rail and /debug. Configure the
same token in the TrueFoundry dashboard under Custom Bearer Auth. The Next Moca
API key is read from NEXT_MOCA_API_KEY on the service, never from the request.

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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

load_dotenv()

from guardrail import _next_moca_client as nm  # noqa: E402
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


def _allow_no_auth() -> bool:
    return os.environ.get("ALLOW_NO_AUTH", "").strip().lower() in ("1", "true", "yes")


def _auth_is_configured() -> bool:
    key = _wrapper_api_key()
    return bool(key) and key not in _PLACEHOLDER_API_KEYS


def require_bearer(request: Request) -> None:
    """Bearer-auth dependency. Fails closed (503) when no real key is configured,
    unless ALLOW_NO_AUTH=true is set for local development."""
    if not _auth_is_configured():
        if _allow_no_auth():
            return
        raise HTTPException(status_code=503, detail="wrapper authentication is not configured")
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    provided = header.split(" ", 1)[1].strip()
    if not hmac.compare_digest(provided.encode(), _wrapper_api_key().encode()):
        raise HTTPException(status_code=401, detail="invalid bearer token")


if not _auth_is_configured():
    if _allow_no_auth():
        log.warning("WRAPPER_API_KEY unset and ALLOW_NO_AUTH=true: bearer auth DISABLED (local development only)")
    else:
        log.error("WRAPPER_API_KEY unset or placeholder: rail and debug routes return 503")


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
    the failing fields only and never echoes request content."""
    fields = sorted({".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()})
    return JSONResponse(status_code=422, content={"error": "invalid guardrail request", "fields": fields})


@app.get("/")
@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


RAIL_ROUTES: dict[str, object] = {
    "/select-input": select_input,
}

for path, fn in RAIL_ROUTES.items():
    app.add_api_route(
        path,
        endpoint=fn,
        methods=["POST"],
        status_code=200,
        dependencies=[Depends(require_bearer)],
    )


@app.get("/debug/loaded-config", dependencies=[Depends(require_bearer)])
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
        "next_moca_api_key_set": bool(nm.api_key()),
        "defaults": {
            "operating_point": nm.DEFAULT_OPERATING_POINT,
            "max_context_tokens": nm.DEFAULT_MAX_CONTEXT_TOKENS,
            "min_tokens": nm.DEFAULT_MIN_TOKENS,
            "timeout_ms": nm.DEFAULT_TIMEOUT_MS,
            "validate_only": False,
        },
        "wrapper_auth_enabled": _auth_is_configured(),
        "wrapper_version": os.environ.get("BUILD_REF", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
