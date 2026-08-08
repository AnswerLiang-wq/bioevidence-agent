"""Privacy-minimized local event log for moderated usability sessions."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
ALLOWED_EVENTS = {
    "session_started",
    "search_started",
    "search_completed",
    "search_failed",
    "card_action",
    "card_useful",
    "source_opened",
    "note_changed",
    "export_started",
    "export_completed",
    "error_shown",
    "timeout",
}
EVENT_METADATA_FIELDS = {
    "session_started": frozenset({"mode"}),
    "search_started": frozenset({"mode"}),
    "search_completed": frozenset({"mode", "card_count", "result_status"}),
    "search_failed": frozenset({"error_code"}),
    "card_action": frozenset({"field", "value"}),
    "card_useful": frozenset({"useful"}),
    "source_opened": frozenset(),
    "note_changed": frozenset({"has_content"}),
    "export_started": frozenset({"format"}),
    "export_completed": frozenset({"format", "accepted_count"}),
    "error_shown": frozenset({"error_code"}),
    "timeout": frozenset({"threshold_minutes"}),
}
ERROR_CODE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


class EventValidationError(ValueError):
    """Raised when a client event would violate the logging contract."""


class EventStore:
    """Append allowlisted metadata without storing questions or notes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._sequence_by_session: dict[str, int] = {}
        self._write_lock = threading.Lock()

    def append(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise EventValidationError("event payload must be an object")
        session_id = _session_id(payload.get("session_id"))
        event_type = payload.get("event_type")
        if not isinstance(event_type, str) or event_type not in ALLOWED_EVENTS:
            raise EventValidationError("event type is not allowlisted")
        task_id = _optional_token(payload.get("task_id"), "task_id")
        card_id = _optional_token(payload.get("card_id"), "card_id")
        elapsed_ms = payload.get("elapsed_ms")
        if type(elapsed_ms) is not int or not 0 <= elapsed_ms <= 86_400_000:
            raise EventValidationError("elapsed_ms must be a non-negative integer")
        cleaned_metadata = _metadata(event_type, payload.get("metadata", {}))

        with self._write_lock:
            path = self.root / f"{session_id}.jsonl"
            previous_sequence = self._sequence_by_session.get(session_id)
            if previous_sequence is None:
                previous_sequence = _last_sequence(path, session_id)
            sequence = previous_sequence + 1
            self._sequence_by_session[session_id] = sequence
            event = {
                "event_version": "1.0.0",
                "session_id": session_id,
                "sequence": sequence,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "task_id": task_id,
                "card_id": card_id,
                "elapsed_ms": elapsed_ms,
                "metadata": cleaned_metadata,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )
        return event


def _session_id(value: object) -> str:
    if not isinstance(value, str) or SESSION_ID_RE.fullmatch(value) is None:
        raise EventValidationError("invalid session_id")
    return value


def _optional_token(value: object, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 80
        or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None
    ):
        raise EventValidationError(f"invalid {label}")
    return value


def _metadata(event_type: str, value: object) -> dict[str, str | int | bool]:
    if not isinstance(value, dict):
        raise EventValidationError("metadata must be an object")
    expected_fields = EVENT_METADATA_FIELDS[event_type]
    if set(value) != expected_fields:
        raise EventValidationError("metadata fields do not match the event contract")

    cleaned: dict[str, str | int | bool] = {}
    for key, item in value.items():
        if key == "mode" and isinstance(item, str) and item in {"standard", "live"}:
            cleaned[key] = item
        elif (
            key == "result_status"
            and isinstance(item, str)
            and item in {"complete", "partial"}
        ):
            cleaned[key] = item
        elif key == "format" and isinstance(item, str) and item in {"json", "markdown"}:
            cleaned[key] = item
        elif (
            key == "field"
            and isinstance(item, str)
            and item in {"action", "user_direction"}
        ):
            cleaned[key] = item
        elif key == "value" and isinstance(item, str) and item in {
                "accepted",
                "excluded",
                "undecided",
                "supports",
                "opposes",
                "unclear",
            }:
            cleaned[key] = item
        elif key == "error_code" and isinstance(item, str) and ERROR_CODE_RE.fullmatch(item):
            cleaned[key] = item
        elif key in {"card_count", "accepted_count"} and type(item) is int and 0 <= item <= 1000:
            cleaned[key] = item
        elif key == "threshold_minutes" and type(item) is int and 1 <= item <= 1440:
            cleaned[key] = item
        elif key in {"useful", "has_content"} and type(item) is bool:
            cleaned[key] = item
        else:
            raise EventValidationError("metadata value does not match the event contract")

    if event_type == "card_action":
        action_values = {"accepted", "excluded", "undecided"}
        direction_values = {"supports", "opposes", "unclear"}
        allowed_values = action_values if cleaned["field"] == "action" else direction_values
        if cleaned["value"] not in allowed_values:
            raise EventValidationError("card action field and value do not match")
    return cleaned


def _last_sequence(path: Path, session_id: str) -> int:
    if not path.exists():
        return 0
    last_sequence = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise EventValidationError("existing event log is invalid") from exc
            sequence = event.get("sequence") if isinstance(event, dict) else None
            if (
                not isinstance(event, dict)
                or event.get("session_id") != session_id
                or type(sequence) is not int
                or sequence != last_sequence + 1
            ):
                raise EventValidationError("existing event log is invalid")
            last_sequence = sequence
    return last_sequence
