"""Pydantic models matching the TrueFoundry custom-guardrail HTTP contract.

This file is intentionally identical across every integration in this monorepo
(no shared/ import). If the gateway contract changes, update EVERY integration's
copy in a coordinated PR. See docs/gateway-contract.md.
"""

from typing import Any, Optional

from pydantic import BaseModel


class ValidateGuardrailResponse(BaseModel):
    """Response body for validate-operation guardrails (AI Gateway JSON contract)."""

    verdict: bool
    message: Optional[str] = None


class MutateGuardrailResponse(BaseModel):
    """Response body for mutate-operation guardrails (AI Gateway JSON contract)."""

    verdict: bool
    transformed: bool
    result: dict[str, Any]


class RequestContext(BaseModel):
    """RequestContext encapsulates contextual information added by the AI Gateway."""

    user: dict  # Expected: {"subjectId", "subjectType", optional subjectSlug, ...}
    metadata: Optional[dict[str, str]] = None


class InputGuardrailRequest(BaseModel):
    """Request schema for input guardrail endpoints."""

    requestBody: dict
    context: RequestContext
    config: Optional[dict] = None


class OutputGuardrailRequest(BaseModel):
    """Request schema for output guardrail endpoints."""

    requestBody: dict
    responseBody: dict
    config: Optional[dict] = None
    context: RequestContext
