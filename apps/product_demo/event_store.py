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
FORBIDDEN_METADATA_KEYS = {
    "question",
    "query",
    "note",
    "text",
    "snippet",
    "title",
    "email",
    "phone",
    "name",
}


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
        if event_type not in ALLOWED_EVENTS:
            raise EventValidationError("event type is not allowlisted")
        task_id = _optional_token(payload.get("task_id"), "task_id")
        card_id = _optional_token(payload.get("card_id"), "card_id")
        elapsed_ms = payload.get("elapsed_ms")
        if not isinstance(elapsed_ms, int) or not 0 <= elapsed_ms <= 86_400_000:
            raise EventValidationError("elapsed_ms must be a non-negative integer")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict) or len(metadata) > 10:
            raise EventValidationError("metadata must be a small object")
        if FORBIDDEN_METADATA_KEYS.intersection(str(key).lower() for key in metadata):
            raise EventValidationError("metadata contains prohibited content fields")
        cleaned_metadata: dict[str, str | int | float | bool | None] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not key or len(key) > 48:
                raise EventValidationError("metadata keys must be short strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise EventValidationError("metadata values must be scalar")
            if isinstance(value, str) and len(value) > 120:
                raise EventValidationError("metadata strings must be short")
            cleaned_metadata[key] = value

        with self._write_lock:
            sequence = self._sequence_by_session.get(session_id, 0) + 1
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
            path = self.root / f"{session_id}.jsonl"
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
