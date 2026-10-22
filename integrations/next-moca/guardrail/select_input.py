"""Input rail (llm_input hook): send tool results to Next Moca, apply what it rendered.

For every `role: "tool"` message in `requestBody.messages` whose text is at least
`min_tokens` in size, the rail sends that text with the task (the latest user
message) to Next Moca, and replaces the message content with the server's
`rendered_context` when the server delivered one. Every other message, and
every other field of the request body, is returned unchanged.

The rail never blocks: `verdict` is always true. On any error, timeout or
non-delivery the original tool result is kept (fail open). With
`validate_only: true` in the config, nothing is transformed and the selection
ids are returned in `message`, for shadow measurement under Operation: Validate.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, Optional, Union

import httpx

from entities import InputGuardrailRequest, MutateGuardrailResponse, ValidateGuardrailResponse
from guardrail import _next_moca_client as nm

log = logging.getLogger("next-moca-guardrails-tfy")

ToolSlot = tuple[int, str]  # (index in messages, text sent)


def _text_of(content: Any) -> Optional[str]:
    """Text of a message content: a string, or a list made only of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                return None
            parts.append(part["text"])
        return "\n".join(parts)
    return None


def latest_user_prompt(messages: list[Any]) -> Optional[str]:
    """Content of the most recent user message, or None when there is none with text."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            text = _text_of(m.get("content"))
            if text and text.strip():
                return text
            return None
    return None


def tool_slots(messages: list[Any], min_tokens: int) -> list[ToolSlot]:
    """Tool-role messages with text at or above the size gate."""
    slots: list[ToolSlot] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        text = _text_of(m.get("content"))
        if text is None or not text.strip():
            continue
        if nm.estimate_tokens(text) < min_tokens:
            continue
        slots.append((i, text))
    return slots


def _passthrough(body: dict, settings: Optional[nm.Settings], message: Optional[str] = None) -> Union[
    MutateGuardrailResponse, ValidateGuardrailResponse
]:
    if settings is not None and settings.validate_only:
        return ValidateGuardrailResponse(verdict=True, message=message)
    return MutateGuardrailResponse(verdict=True, transformed=False, result=copy.deepcopy(body))


async def _select_all(slots: list[ToolSlot], task_prompt: str, settings: nm.Settings) -> list[nm.Decision]:
    key = nm.api_key()
    url = nm.api_base() + nm.ENDPOINT_PATH
    semaphore = asyncio.Semaphore(nm.MAX_CONCURRENT_CALLS)

    async with httpx.AsyncClient() as client:

        async def run(text: str) -> nm.Decision:
            async with semaphore:
                return await nm.select_one(
                    client, url=url, key=key, text=text, task_prompt=task_prompt, settings=settings
                )

        return list(await asyncio.gather(*(run(text) for _, text in slots)))


def _shadow_message(decisions: list[nm.Decision]) -> str:
    """One `<selection id>:<outcome or reason>:<would_apply|passthrough>` item per tool message sent."""
    items = []
    for d in decisions:
        action = "would_apply" if d.applied else nm.DECISION_PASSTHROUGH
        items.append(f"{d.selection_id or 'none'}:{d.outcome or d.reason}:{action}")
    return "Next Moca shadow: " + ", ".join(items)


async def select_input(request: InputGuardrailRequest) -> Union[MutateGuardrailResponse, ValidateGuardrailResponse]:
    body = request.requestBody
    settings: Optional[nm.Settings] = None
    try:
        settings = nm.resolve_settings(request.config)
        messages = body.get("messages")
        if not isinstance(messages, list):
            return _passthrough(body, settings)

        slots = tool_slots(messages, settings.min_tokens)
        if not slots:
            return _passthrough(body, settings)

        task_prompt = latest_user_prompt(messages)
        if task_prompt is None:
            return _passthrough(body, settings)

        if not nm.api_key():
            log.warning("%s is not set; passing the request through unchanged", nm.ENV_API_KEY)
            return _passthrough(body, settings)

        decisions = await _select_all(slots, task_prompt, settings)
        for (index, _), d in zip(slots, decisions):
            log.info(
                "tool_message=%d decision=%s reason=%s outcome=%s selection_id=%s receipt=%s",
                index,
                d.decision,
                d.reason,
                d.outcome,
                d.selection_id,
                d.receipt,
            )

        if settings.validate_only:
            return ValidateGuardrailResponse(verdict=True, message=_shadow_message(decisions))

        if not any(d.applied for d in decisions):
            return _passthrough(body, settings)

        result = copy.deepcopy(body)
        for (index, _), d in zip(slots, decisions):
            if d.applied:
                result["messages"][index]["content"] = d.rendered
        return MutateGuardrailResponse(verdict=True, transformed=True, result=result)
    except Exception as exc:  # noqa: BLE001 - the rail fails open on any error
        log.error("select-input failed open: %s", type(exc).__name__)
        return _passthrough(body, settings)
