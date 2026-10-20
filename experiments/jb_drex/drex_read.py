"""Print the text, tables and layout of a PDF, Office file or image (OCR) as Markdown.

Usage: python scripts/drex_read.py <path>

Reads DREX_API_KEY and, optionally, DREX_BASE_URL (default https://drex.nace.ai).
Files of any other type are printed as text.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DREX_EXTENSIONS = frozenset({
    "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
    "png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp",
})
DREX_DEFAULT_BASE = "https://drex.nace.ai"
DREX_DEADLINE_S = 170
DREX_POLL_S = 2
DREX_DONE = ("succeeded", "failed", "cancelled")


class DrexError(Exception):
    """A failure to read a document through Drex, worded for stderr."""


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc


class _DropAuthCrossHost(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the host: signed storage URLs refuse it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _host(newurl) != _host(req.full_url):
            new.headers.pop("Authorization", None)
        return new


_drex_opener = urllib.request.build_opener(_DropAuthCrossHost)


def _drex_request(method: str, url: str, *, key: str | None = None, body: bytes | None = None,
                  headers: dict | None = None) -> bytes:
    """One HTTP call; `key` adds the bearer header. A non-2xx answer raises DrexError."""
    route = urllib.parse.urlsplit(url).path
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with _drex_opener.open(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(detail)["error"]
            detail = f"{err.get('code', 'error')}: {err.get('message', '')}"
        except (ValueError, KeyError, TypeError, AttributeError):
            detail = detail.strip()[:200]
        raise DrexError(f"Drex {method} {route} -> HTTP {e.code} {detail}")
    except (urllib.error.URLError, OSError) as e:
        raise DrexError(f"Drex {method} {route} failed: {e}")


def _drex_json(method: str, url: str, key: str, payload: dict,
               headers: dict | None = None) -> dict:
    return json.loads(_drex_request(
        method, url, key=key, body=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})}))


def _drex_upload(path: Path, data: bytes, remote: str, base: str, key: str) -> tuple[str, str]:
    """Upload through a one-shot grant and return (workspace_id, file_id)."""
    grant = _drex_json("POST", f"{base}/v1/documents/upload-grants", key,
                       {"path": remote, "ttl_seconds": 600})
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode(), data,
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n'.encode(),
        json.dumps({"path": remote}).encode(), f"\r\n--{boundary}--\r\n".encode(),
    ])
    up = json.loads(_drex_request(
        "POST", grant["upload_url"], body=body,
        headers={"X-Upload-Token": grant["token"],
                 "Content-Type": f"multipart/form-data; boundary={boundary}"}))
    file = (up.get("result") or {}).get("file") or up["file"]
    return up.get("workspace_id") or file["workspace_id"], file.get("file_id") or file["id"]


def _drex_parse(workspace_id: str, file_id: str, base: str, key: str) -> dict:
    """Create the parse job and wait for it to finish."""
    job = _drex_json(
        "POST", f"{base}/v1/documents/parse?wait_seconds=60", key,
        {"source": {"type": "workspace_file", "workspace_id": workspace_id, "file_id": file_id},
         "output": {"formats": ["markdown"]}},
        headers={"Idempotency-Key": f"jb-drex-read-{file_id}"})
    deadline = time.monotonic() + DREX_DEADLINE_S
    while job["status"] not in DREX_DONE:
        if time.monotonic() > deadline:
            raise DrexError(
                f"Drex parse job {job['job_id']} still {job['status']} after {DREX_DEADLINE_S}s")
        time.sleep(DREX_POLL_S)
        job = json.loads(_drex_request("GET", f"{base}/v1/documents/jobs/{job['job_id']}", key=key))
    if job["status"] != "succeeded":
        err = job.get("error") or {}
        raise DrexError(
            f"Drex parse {job['status']}: {err.get('code', 'unknown')}: {err.get('message', '')}")
    return job


def _drex_fetch(url: str, key: str) -> str:
    return _drex_request("GET", url, key=key).decode("utf-8", errors="replace")


def _drex_text(job: dict, key: str) -> str:
    """The full markdown: `document.markdown` is only a preview when a content_url is set."""
    doc = job["result"]["document"]
    sheets = [s for s in (doc.get("spreadsheet") or {}).get("sheets") or [] if s.get("content_url")]
    if sheets:
        return "\n\n".join(
            f"## {s['sheet_name']}\n\n{_drex_fetch(s['content_url'], key)}" for s in sheets)
    if doc.get("content_url"):
        return _drex_fetch(doc["content_url"], key)
    return doc.get("markdown") or ""


def parse_with_drex(path: Path) -> int:
    """Parse through Drex's Documents API, caching the markdown by content hash."""
    key = os.environ.get("DREX_API_KEY")
    if not key:
        print("DREX_API_KEY is not set", file=sys.stderr)
        return 1
    base = os.environ.get("DREX_BASE_URL", DREX_DEFAULT_BASE).rstrip("/")
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache = cache_home / "harness2" / "drex" / f"{sha}.md"
    if cache.is_file():
        sys.stdout.write(cache.read_text(encoding="utf-8"))
        return 0
    try:
        remote = f"harness2/{sha[:16]}/{path.name}"
        workspace_id, file_id = _drex_upload(path, data, remote, base, key)
        text = _drex_text(_drex_parse(workspace_id, file_id, base, key), key)
    except DrexError as e:
        print(e, file=sys.stderr)
        return 1
    except (KeyError, ValueError) as e:
        print(f"unexpected Drex response: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f".{cache.name}.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, cache)
    sys.stdout.write(text)
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/drex_read.py <path>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1]).resolve()
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    if path.suffix.lower().lstrip(".") not in DREX_EXTENSIONS:
        sys.stdout.write(path.read_text(encoding="utf-8", errors="replace"))
        return 0
    return parse_with_drex(path)


if __name__ == "__main__":
    sys.exit(main())
