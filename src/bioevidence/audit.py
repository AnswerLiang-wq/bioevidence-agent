"""Response auditing primitives shared by the smoke test and the comparison run.

The point of this module is that a model response is recorded **losslessly**: fields
the SDK does not model still survive, and every field carries whether the server
sent it, sent null, sent an empty value, or never sent it at all.  Dropping that
distinction is what lets an unreported token count masquerade as a confirmed zero.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ABSENT",
    "USAGE_FIELDS",
    "MESSAGE_FIELDS",
    "AuditingClient",
    "declared_states",
    "extension_states",
    "field_state",
    "json_safe",
    "response_to_dict",
    "summarize_rounds",
    "usage_to_dict",
]


class _Absent:
    """Sentinel for a field the server did not send at all."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<absent>"


ABSENT = _Absent()

# Declared fields of the openai SDK's CompletionUsage / ChatCompletionMessage.
# Anything outside these sets is an extension field the payload carried.
USAGE_FIELDS: tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
)
MESSAGE_FIELDS: tuple[str, ...] = (
    "role",
    "content",
    "refusal",
    "annotations",
    "audio",
    "function_call",
    "tool_calls",
)


def field_state(value: Any) -> str:
    """Classify one field value.

    absent  — the server did not send the field at all
    null    — the server sent an explicit JSON null
    empty   — the server sent "", [] or {}
    present — the server sent a non-empty value

    absent and null are deliberately not collapsed.  "We were not told" and "we
    were told nothing" are different audit facts, and empty is a third; merging
    them is what lets an unreported field masquerade as a reported zero.
    """
    if value is ABSENT:
        return "absent"
    if value is None:
        return "null"
    if isinstance(value, (str, bytes, list, tuple, set, dict)) and len(value) == 0:
        return "empty"
    return "present"


def json_safe(value: Any) -> Any:
    """Coerce a captured value into something json.dumps can write.

    SDK objects expose model_dump(), so real responses serialise structurally.
    Anything else (a test double, a future type the SDK adds) degrades to repr
    rather than crashing the run whose audit record is being written.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return json_safe(dump())
        except Exception:  # noqa: BLE001 - fall back to repr below
            pass
    return repr(value)


def declared_states(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Four-state map, keyed off what the server actually sent.

    pydantic's model_fields_set records which fields were present in the parsed
    payload, so a declared-but-unsent field is distinguishable from one the
    server explicitly set to null.  Duck-typed test doubles lack that attribute
    and fall back to attribute presence.
    """
    sent = getattr(obj, "model_fields_set", None)
    states: dict[str, Any] = {}
    for name in names:
        if sent is not None and name not in sent:
            value: Any = ABSENT
        else:
            value = getattr(obj, name, ABSENT)
        states[name] = {
            "state": field_state(value),
            "value": None if value is ABSENT else json_safe(value),
        }
    return states


def extension_states(obj: Any, declared: tuple[str, ...]) -> dict[str, Any]:
    """Fields the payload carried beyond the SDK's declared schema.

    The SDK models declare extra="allow", so server-side additions survive
    parsing and land in model_extra.  Reading only named attributes is exactly
    what discarded DeepSeek's prompt_cache_hit_tokens / prompt_cache_miss_tokens
    (the cache split needed for an honest cost estimate) and the message-level
    reasoning_content in earlier runs.
    """
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        items = list(extra.items())
    else:
        items = [
            (name, value)
            for name, value in vars(obj).items()
            if name not in declared and not name.startswith("_") and not callable(value)
        ]
    return {name: {"state": field_state(value), "value": json_safe(value)} for name, value in items}


def usage_to_dict(usage: Any | None) -> dict[str, Any]:
    """Serialise usage without discarding cache counters or collapsing states.

    Usage handling:
      - usage present → flat counts plus a four-state map of every declared field
        and a map of extension fields the server added.  Do NOT read the flat
        counts alone: whether a count was sent, sent as null, or omitted is
        recorded in `fields`.
      - usage absent  → status "unknown" and every declared field marked absent.
        Do NOT write zero — zero is indistinguishable from a confirmed-zero and
        masks missing data.  "unknown" makes cost estimates impossible to
        compute honestly.
    """
    if usage is None:
        return {
            "status": "unknown",
            "note": "response.usage was absent; token counts are unknown, not zero.",
            "fields": {name: {"state": "absent", "value": None} for name in USAGE_FIELDS},
            "extension_fields": {},
        }
    return {
        "status": "confirmed",
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "fields": declared_states(usage, USAGE_FIELDS),
        "extension_fields": extension_states(usage, USAGE_FIELDS),
    }


def response_to_dict(response: Any) -> dict[str, Any]:
    """Extract the auditable fields from a real ChatCompletion object, losslessly.

    Beyond the flat summary, every choice carries `message_fields` (four-state)
    and `message_extension_fields`, so a field the SDK does not model — such as
    DeepSeek's reasoning_content — is preserved rather than dropped.
    """
    usage_dict = usage_to_dict(response.usage)

    choices = []
    for choice in response.choices:
        msg = choice.message
        tool_calls_out = None
        if msg.tool_calls:
            tool_calls_out = [
                {
                    "id": tc.id,
                    "function_name": tc.function.name,
                    "function_arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]
        choices.append({
            "finish_reason": choice.finish_reason,
            "message_role": msg.role,
            "message_content": msg.content,
            "tool_calls": tool_calls_out,
            "message_fields": declared_states(msg, MESSAGE_FIELDS),
            "message_extension_fields": extension_states(msg, MESSAGE_FIELDS),
        })
    return {
        "id": response.id,
        "model": response.model,
        "usage": usage_dict,
        "choices": choices,
    }


class AuditingClient:
    """Wraps a real OpenAI-compatible client and records every create() round-trip.

    Every round record contains a deep-copied snapshot of the full request
    messages list, whatever extra_body was passed, and the full response or the
    exception raised.  Records survive an exception, which is the point: a run
    that dies on round 3 must still account for rounds 1 and 2.
    """

    def __init__(self, real_client: Any) -> None:
        self._real = real_client
        self.rounds: list[dict[str, Any]] = []
        self.final_messages_snapshot: list[dict[str, Any]] | None = None
        self.chat = self
        self.completions = self

    def reset(self) -> None:
        self.rounds = []
        self.final_messages_snapshot = None

    @property
    def real_client(self) -> Any:
        return self._real

    def create(self, **kwargs: Any) -> Any:
        import copy

        call_index = len(self.rounds)
        # Deep-copy the messages list so subsequent appends don't mutate this record.
        messages_snapshot = copy.deepcopy(kwargs.get("messages", []))
        entry: dict[str, Any] = {
            "call_index": call_index,
            "request_model": kwargs.get("model"),
            "request_messages": messages_snapshot,
            "request_messages_count": len(messages_snapshot),
            "request_max_tokens": kwargs.get("max_tokens"),
            "request_temperature": kwargs.get("temperature"),
            # Capture the thinking-disable config (or any other extra_body) so
            # tests can assert it was actually sent.
            "request_extra_body": kwargs.get("extra_body"),
            "response": None,
            "error": None,
        }
        try:
            response = self._real.chat.completions.create(**kwargs)
            entry["response"] = response_to_dict(response)
            self.rounds.append(entry)
            return response
        except Exception as exc:
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.rounds.append(entry)
            raise

    def attach_agent(self, agent: Any) -> None:
        """Install this wrapper as ``agent``'s client, keeping the real one reachable."""
        agent._client = self

    def finalize(self, messages: list[Any]) -> None:
        """Snapshot the messages list after the loop ends.

        Tool results appended after the last create() call are never captured in
        any round's request_messages, so without this they would be lost.
        """
        import copy

        self.final_messages_snapshot = copy.deepcopy(messages)


def summarize_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact, checkable summary of the round-trips actually performed.

    Deliberately smaller than the raw records: this is what belongs in a
    per-case comparison report, and it is enough to check request count, the
    model identifier the server actually reported, token usage, the tool calls
    the model emitted, and why the loop ended.
    """
    summary: dict[str, Any] = {
        "request_count": len(rounds),
        "response_models": [],
        "tool_calls": [],
        "errors": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "rounds_with_usage": 0,
            "rounds_without_usage": 0,
            "coverage": "unknown",
            "per_round": [],
        },
        "finish_reasons": [],
        "request_extra_body": None,
        "request_max_tokens": None,
        "request_temperature": None,
        "cache_counters": {},
    }
    for entry in rounds:
        response = entry.get("response")
        if entry.get("error"):
            summary["errors"].append({
                "call_index": entry.get("call_index"),
                "type": entry["error"].get("type"),
                "message": entry["error"].get("message"),
            })
        if entry.get("request_extra_body") is not None:
            summary["request_extra_body"] = entry["request_extra_body"]
        if entry.get("request_max_tokens") is not None:
            summary["request_max_tokens"] = entry["request_max_tokens"]
        if entry.get("request_temperature") is not None:
            summary["request_temperature"] = entry["request_temperature"]
        if not isinstance(response, dict):
            summary["usage"]["per_round"].append({"call_index": entry.get("call_index"),
                                                  "status": "unknown"})
            summary["usage"]["rounds_without_usage"] += 1
            continue

        model = response.get("model")
        if model not in summary["response_models"]:
            summary["response_models"].append(model)

        usage = response.get("usage") or {}
        if usage.get("status") == "confirmed":
            summary["usage"]["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            summary["usage"]["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            summary["usage"]["rounds_with_usage"] += 1
            summary["usage"]["per_round"].append({
                "call_index": entry.get("call_index"),
                "status": "confirmed",
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "extension_fields": sorted((usage.get("extension_fields") or {}).keys()),
            })
            # Cache counters the server volunteered are the only way to judge how
            # far the peak cache-miss cost estimate overstates the real charge.
            for name, entry_value in (usage.get("extension_fields") or {}).items():
                if "cache" in name.lower():
                    summary["cache_counters"][name] = entry_value.get("value")
        else:
            summary["usage"]["rounds_without_usage"] += 1
            summary["usage"]["per_round"].append({"call_index": entry.get("call_index"),
                                                  "status": "unknown"})

        for choice in response.get("choices") or []:
            if choice.get("finish_reason"):
                summary["finish_reasons"].append(choice["finish_reason"])
            for call in choice.get("tool_calls") or []:
                summary["tool_calls"].append({
                    "call_index": entry.get("call_index"),
                    "name": call.get("function_name"),
                    "arguments": call.get("function_arguments"),
                })

    total_rounds = len(rounds)
    if total_rounds and summary["usage"]["rounds_with_usage"] == total_rounds:
        summary["usage"]["coverage"] = "confirmed"
    elif summary["usage"]["rounds_with_usage"]:
        summary["usage"]["coverage"] = "partial"
    else:
        summary["usage"]["coverage"] = "unknown"
    return summary
