"""Zero-dependency local HTTP server for the BioEvidence product demo."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .event_store import EventStore, EventValidationError
from .service import ProductDemoError, ProductDemoService


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
MAX_REQUEST_BYTES = 1_000_000
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def create_handler(
    service: ProductDemoService,
    event_store: EventStore,
) -> type[BaseHTTPRequestHandler]:
    class ProductDemoHandler(BaseHTTPRequestHandler):
        server_version = "BioEvidenceProductDemo/0.4-dev"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "scope": "local_research_demo",
                        "product_version": "0.4.0-dev",
                    },
                )
                return
            if self.path == "/api/standard-tasks":
                self._json(HTTPStatus.OK, {"tasks": service.standard_tasks()})
                return
            static = STATIC_FILES.get(self.path)
            if static is None:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "页面不存在。")
                return
            filename, content_type = static
            payload = (STATIC_ROOT / filename).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", _content_security_policy())
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._request_json()
                if self.path == "/api/search":
                    result = service.search(
                        session_id=_string(payload, "session_id"),
                        mode=_string(payload, "mode"),
                        question=payload.get("question"),
                        task_id=payload.get("task_id")
                        if isinstance(payload.get("task_id"), str)
                        else None,
                    )
                    self._json(HTTPStatus.OK, result)
                    return
                if self.path == "/api/export":
                    result = service.export_pack(
                        session_id=_string(payload, "session_id"),
                        decisions=payload.get("decisions"),
                        pack_status=_string(payload, "pack_status"),
                        synthesis=_string(payload, "synthesis", allow_empty=True),
                    )
                    self._json(HTTPStatus.OK, result)
                    return
                if self.path == "/api/events":
                    event = event_store.append(payload)
                    self._json(
                        HTTPStatus.CREATED,
                        {"status": "recorded", "sequence": event["sequence"]},
                    )
                    return
                self._error(HTTPStatus.NOT_FOUND, "not_found", "API不存在。")
            except ProductDemoError as exc:
                self._error(exc.http_status, exc.code, str(exc))
            except EventValidationError as exc:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_event", str(exc))
            except RequestError as exc:
                self._error(exc.http_status, exc.code, str(exc))
            except Exception as exc:  # keep internals out of browser responses
                print(f"product demo internal error: {type(exc).__name__}: {exc}")
                self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "本地服务发生未预期错误，请保留事件记录并停止本次任务。",
                )

        def _request_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None or not raw_length.isdigit():
                raise RequestError("missing_length", "请求缺少有效长度。")
            length = int(raw_length)
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise RequestError(
                    "request_too_large",
                    "请求内容为空或超过本地Demo限制。",
                    http_status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            try:
                value = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestError("invalid_json", "请求不是有效JSON。") from exc
            if not isinstance(value, dict):
                raise RequestError("invalid_json", "JSON顶层必须是对象。")
            return value

        def _json(self, status: int, value: object) -> None:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

        def log_message(self, format: str, *args: object) -> None:
            # The default log contains method/path/status only, never request bodies.
            print(f"product-demo {self.address_string()} {format % args}")

    return ProductDemoHandler


class RequestError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int = HTTPStatus.BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def _string(
    payload: dict[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RequestError("invalid_request", f"{key}必须是文本。")
    return value


def _content_security_policy() -> str:
    return (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local BioEvidence demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--event-dir",
        type=Path,
        default=APP_ROOT / "session_data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Product Demo is local-only; host must be 127.0.0.1")
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    service = ProductDemoService()
    event_store = EventStore(args.event_dir)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(service, event_store),
    )
    print(f"BioEvidence Product Demo: http://{args.host}:{args.port}")
    print(f"Privacy-minimized event directory: {args.event_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping BioEvidence Product Demo")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
