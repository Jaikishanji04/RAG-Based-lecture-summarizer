"""Lecture source ingestion, document extraction, and cache management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
LECTURES_FILE = DATA_DIR / "lectures.json"
SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".pdf", ".pptx"}


class IngestionError(RuntimeError):
    """Raised when a lecture source cannot be ingested."""


def ensure_data_directories() -> None:
    """Create the data directories and empty lecture registry if needed."""

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LECTURES_FILE.exists():
        LECTURES_FILE.write_text("{}\n", encoding="utf-8")


def load_lectures() -> dict[str, dict[str, Any]]:
    """Load the persisted lecture registry."""

    ensure_data_directories()
    try:
        value = json.loads(LECTURES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestionError(f"Could not read lecture registry: {exc}") from exc
    if not isinstance(value, dict):
        raise IngestionError("Lecture registry must contain a JSON object.")
    return value


def save_lectures(lectures: dict[str, dict[str, Any]]) -> None:
    """Persist the lecture registry atomically."""

    ensure_data_directories()
    temporary_file = LECTURES_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(lectures, indent=2), encoding="utf-8")
    temporary_file.replace(LECTURES_FILE)


def source_fingerprint(source: Path | str) -> str:
    """Return a stable SHA-256 identifier for a local path or URL."""

    if isinstance(source, Path):
        digest = hashlib.sha256()
        with source.open("rb") as file_handle:
            for block in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    return hashlib.sha256(source.strip().encode("utf-8")).hexdigest()


def _safe_title(source: Path | str) -> str:
    if isinstance(source, Path):
        return source.stem.replace("_", " ").replace("-", " ").strip() or "Lecture"
    hostname = urlparse(source).hostname or "YouTube lecture"
    return f"Lecture from {hostname}"


def _copy_uploaded_source(source: Path, lecture_id: str) -> Path:
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type: {source.suffix or 'unknown'}")
    destination = UPLOADS_DIR / f"{lecture_id}{suffix}"
    shutil.copy2(source, destination)
    return destination


def _download_youtube(url: str, lecture_id: str) -> Path:
    """Download a YouTube source as an audio file using yt-dlp."""

    destination = UPLOADS_DIR / f"{lecture_id}.mp3"
    try:
        from yt_dlp import YoutubeDL

        options = {
            "format": "bestaudio/best",
            "outtmpl": str(UPLOADS_DIR / f"{lecture_id}.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
            ],
        }
        browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip().lower()
        if browser:
            options["cookiesfrombrowser"] = (browser,)
        with YoutubeDL(options) as downloader:
            downloader.download([url])
    except Exception as exc:
        if "Sign in to confirm" in str(exc) or "cookies" in str(exc).lower():
            raise IngestionError(
                "YouTube requires authentication for this video. Set "
                "YTDLP_COOKIES_FROM_BROWSER=chrome or edge in .env, close that browser, "
                "and retry; otherwise upload a local MP3 or MP4."
            ) from exc
        raise IngestionError(f"Could not download YouTube audio: {exc}") from exc
    if not destination.exists():
        raise IngestionError("YouTube download completed without producing an audio file.")
    return destination


def _extract_pdf(path: Path) -> list[dict[str, Any]]:
    """Extract PDF pages in parallel to speed up large documents.
    Each page is turned into a dict {"slide_no": N, "text": "..."}.
    """
    try:
        import fitz
        from concurrent.futures import ThreadPoolExecutor, as_completed

        doc = fitz.open(path)
        # Prepare a callable for each page
        def _read_page(idx_page):
            idx, page = idx_page
            return {"slide_no": idx + 1, "text": page.get_text("text").strip()}

        # Submit all pages to a thread pool (IO‑bound)
        futures = []
        with ThreadPoolExecutor(max_workers=min(8, len(doc))) as executor:
            for idx_page in enumerate(doc):
                futures.append(executor.submit(_read_page, (idx_page, doc[idx_page])))
            pages = [future.result() for future in as_completed(futures)]
        doc.close()
        # Preserve original order by sorting on slide_no
        pages.sort(key=lambda p: p["slide_no"])
        return pages
    except Exception as exc:
        raise IngestionError(f"Could not extract PDF text: {exc}") from exc


def _extract_pptx(path: Path) -> list[dict[str, Any]]:
    try:
        from pptx import Presentation

        presentation = Presentation(path)
        pages: list[dict[str, Any]] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            text = "\n".join(
                shape.text.strip()
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text.strip()
            )
            pages.append({"slide_no": slide_number, "text": text})
        return pages
    except Exception as exc:
        raise IngestionError(f"Could not extract PowerPoint text: {exc}") from exc


def extract_document_text(path: Path) -> list[dict[str, Any]]:
    """Extract page or slide text while retaining its source number."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf(path)
    elif suffix == ".pptx":
        pages = _extract_pptx(path)
    else:
        raise IngestionError("Only PDF and PPTX files contain directly extractable document text.")
    output_path = path.with_suffix(".pages.json")
    output_path.write_text(json.dumps(pages, indent=2), encoding="utf-8")
    return pages


def ingest_source(
    source: Path | str,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Register and prepare a local file or YouTube URL, reusing cached work."""

    ensure_data_directories()
    local_source = source if isinstance(source, Path) else None
    if local_source is not None and not local_source.is_file():
        raise IngestionError(f"Source file does not exist: {local_source}")
    if local_source is not None and local_source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type: {local_source.suffix or 'unknown'}")

    fingerprint = source_fingerprint(source)
    lectures = load_lectures()
    for lecture in lectures.values():
        if lecture.get("source_fingerprint") == fingerprint and lecture.get("status") == "ready":
            if progress_callback:
                progress_callback(1.0, "Source already processed; using cached lecture")
            return lecture

    lecture_id = uuid4().hex[:12]
    record: dict[str, Any] = {
        "lecture_id": lecture_id,
        "title": _safe_title(source),
        "source": str(source),
        "source_fingerprint": fingerprint,
        "source_type": "youtube" if isinstance(source, str) else local_source.suffix.lower()[1:],
        "status": "processing",
        "created_at": datetime.now(UTC).isoformat(),
        "transcript_path": None,
        "pages_path": None,
    }
    lectures[lecture_id] = record
    save_lectures(lectures)
    try:
        if progress_callback:
            progress_callback(0.1, "Preparing source")
        prepared_path = _download_youtube(source, lecture_id) if isinstance(source, str) else _copy_uploaded_source(source, lecture_id)
        record["path"] = str(prepared_path)
        if prepared_path.suffix.lower() in {".pdf", ".pptx"}:
            if progress_callback:
                progress_callback(0.55, "Extracting document text")
            pages = extract_document_text(prepared_path)
            record["pages_path"] = str(prepared_path.with_suffix(".pages.json"))
            record["page_count"] = len(pages)
        else:
            from .transcribe import transcribe_media

            transcript_path = DATA_DIR / f"{lecture_id}.transcript.json"
            if progress_callback:
                progress_callback(0.25, "Loading speech transcription model")
            transcribe_media(prepared_path, transcript_path, progress_callback=progress_callback)
            record["transcript_path"] = str(transcript_path)
        record["status"] = "ready"
        save_lectures(lectures)
        if progress_callback:
            progress_callback(1.0, "Lecture ready")
        return record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        save_lectures(lectures)
        raise IngestionError(f"Could not process lecture: {exc}") from exc
