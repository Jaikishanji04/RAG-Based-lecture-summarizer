"""Audio preparation and timestamped faster-whisper transcription."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)

def _needs_conversion(p: Path) -> bool:
    """Return True if *p* is not already a mono 16 kHz PCM‑S16LE WAV.
    Uses ffprobe to inspect codec, channel count and sample rate.
    """
    try:
        import subprocess, json
        info = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_name,channels,sample_rate",
            "-of", "json", str(p)
        ])
        streams = json.loads(info)["streams"][0]
        return not (
            streams["codec_name"] == "pcm_s16le"
            and int(streams["channels"]) == 1
            and int(streams["sample_rate"]) == 16000
        )
    except Exception:
        # If probing fails, assume conversion is needed.
        return True


class TranscriptionError(RuntimeError):
    """Raised when audio preparation or speech recognition fails."""


def _decode_prepared_wav(input_file: str | os.PathLike[str], sampling_rate: int = 16000) -> Any:
    """Decode our FFmpeg-produced mono WAV without PyAV metadata arguments."""

    try:
        import numpy as np

        with wave.open(str(input_file), "rb") as wav_file:
            if wav_file.getframerate() != sampling_rate or wav_file.getnchannels() != 1:
                raise TranscriptionError("Prepared audio must be mono 16 kHz WAV.")
            samples = wav_file.readframes(wav_file.getnframes())
        return np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
    except (OSError, ValueError) as exc:
        raise TranscriptionError(f"Could not decode prepared WAV audio: {exc}") from exc


def prepare_audio(media_path: Path, output_path: Path) -> Path:
    """Convert media to mono 16 kHz PCM WAV using the installed FFmpeg."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Skip ffmpeg if input already matches target PCM format
    if not _needs_conversion(media_path):
        return media_path
    command = [
        "ffmpeg",
        "-y",
        "-threads", "0",
        "-loglevel", "error",
        "-i", str(media_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-preset", "veryfast",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise TranscriptionError("FFmpeg was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip().splitlines()[-1] if exc.stderr else "unknown FFmpeg error"
        raise TranscriptionError(f"FFmpeg could not prepare audio: {detail}") from exc
    return output_path


def prepare_audio_segments(media_path: Path, output_dir: Path, segment_seconds: int = 600) -> list[Path]:
    """Convert media to bounded mono 16 kHz WAV segments."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "segment-%05d.wav"
    command = [
        "ffmpeg",
        "-y",
        "-threads", "0",
        "-loglevel", "error",
        "-i", str(media_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-preset", "veryfast",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        str(pattern),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise TranscriptionError("FFmpeg was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip().splitlines()[-1] if exc.stderr else "unknown FFmpeg error"
        raise TranscriptionError(f"FFmpeg could not prepare audio: {detail}") from exc
    segments = sorted(output_dir.glob("segment-*.wav"))
    if not segments:
        raise TranscriptionError("FFmpeg did not produce any audio segments.")
    return segments


def _device_and_compute_type() -> tuple[str, str]:
    configured_device = os.getenv("WHISPER_DEVICE", "auto").lower().strip()
    if configured_device in {"cpu", "cuda"}:
        return configured_device, os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", os.getenv("WHISPER_GPU_COMPUTE_TYPE", "float16")
    except ImportError:
        pass
    return "cpu", os.getenv("WHISPER_COMPUTE_TYPE", "int8")


@lru_cache(maxsize=2)
def get_whisper_model(model_size: str | None = None) -> Any:
    """Load and cache a faster-whisper model for the current device."""

    try:
        from faster_whisper import WhisperModel
        import faster_whisper.transcribe as faster_whisper_transcribe
    except ImportError as exc:
        raise TranscriptionError("Install faster-whisper before transcribing audio.") from exc
    faster_whisper_transcribe.decode_audio = _decode_prepared_wav
    size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
    device, compute_type = _device_and_compute_type()
    try:
        return WhisperModel(size, device=device, compute_type=compute_type)
    except Exception as exc:
        raise TranscriptionError(f"Could not load Whisper model '{size}': {exc}") from exc


def transcribe_media(
    media_path: Path,
    transcript_path: Path,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    """Transcribe media and save ``[{start, end, text}]`` JSON."""

    if not media_path.is_file():
        raise TranscriptionError(f"Media file does not exist: {media_path}")
    model = get_whisper_model()
    try:
        transcript: list[dict[str, float | str]] = []
        segment_seconds = int(os.getenv("WHISPER_SEGMENT_SECONDS", "600"))
        with tempfile.TemporaryDirectory(prefix="lecture-audio-") as temporary_directory:
            audio_segments = prepare_audio_segments(media_path, Path(temporary_directory), segment_seconds)
            # Transcribe segments in parallel to speed up processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = os.cpu_count() or 2

            def _transcribe_segment(args):
                seg_num, audio_path = args
                segments, _ = model.transcribe(
                    str(audio_path),
                    beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
                    vad_filter=True,
                )
                offset = seg_num * segment_seconds
                results = []
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        results.append({
                            "start": offset + float(segment.start),
                            "end": offset + float(segment.end),
                            "text": text,
                        })
                return seg_num, results

            # Submit transcription tasks
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for segment_number, audio_path in enumerate(audio_segments):
                    futures.append(executor.submit(_transcribe_segment, (segment_number, audio_path)))
                for future in as_completed(futures):
                    seg_num, seg_results = future.result()
                    transcript.extend(seg_results)
                    if progress_callback:
                        progress_callback(
                            min((seg_num + 1) / len(audio_segments), 0.95),
                            f"Transcribing audio segment {seg_num + 1}/{len(audio_segments)}",
                        )
    except Exception as exc:
        raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return transcript_path
