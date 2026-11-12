"""Funciones sin interfaz para crear audio sincronizado desde subtítulos SRT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Subtitle:
    index: int
    start_seconds: float
    end_seconds: float
    text: str

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class VoiceRecord:
    name: str
    audio_path: Path | None
    transcript_path: Path | None
    enabled: bool
    reason: str = ""


_TIMESTAMP = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")


def read_transcript(path: Path) -> str:
    """Lee transcripciones habituales de Windows y devuelve texto Unicode limpio."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "cp1252"):
        try:
            candidate = raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
        # Un UTF-16 leído como UTF-8 contiene NUL entre caracteres: probar la siguiente opción.
        if "\x00" not in candidate:
            return candidate
    raise ValueError("No se pudo leer el TXT como UTF-8, UTF-16 o Windows-1252.")


def _timestamp_to_seconds(value: str) -> float:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*", value)
    if not match:
        raise ValueError(f"Marca temporal inválida: {value!r}")
    hours, minutes, seconds, milliseconds = match.groups()
    if int(minutes) > 59 or int(seconds) > 59:
        raise ValueError(f"Marca temporal inválida: {value!r}")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds.ljust(3, "0")) / 1000


def parse_srt(path: Path) -> list[Subtitle]:
    """Lee un SRT UTF-8, conservando los saltos de línea del texto."""
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    subtitles: list[Subtitle] = []
    for number, block in enumerate(re.split(r"\n[ \t]*\n", text.strip()), start=1):
        lines = [line.rstrip() for line in block.split("\n")]
        if not lines or not any(line.strip() for line in lines):
            continue
        time_line_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_line_index is None:
            raise ValueError(f"Bloque SRT {number}: falta la línea de tiempos.")
        time_parts = lines[time_line_index].split("-->", maxsplit=1)
        if len(time_parts) != 2:
            raise ValueError(f"Bloque SRT {number}: tiempos inválidos.")
        start = _timestamp_to_seconds(time_parts[0])
        end = _timestamp_to_seconds(time_parts[1].split()[0])
        cue_text = "\n".join(lines[time_line_index + 1 :]).strip()
        if end <= start:
            raise ValueError(f"Bloque SRT {number}: el final debe ser posterior al inicio.")
        if not cue_text:
            continue
        subtitle_id = int(lines[0]) if time_line_index > 0 and lines[0].strip().isdigit() else number
        subtitles.append(Subtitle(subtitle_id, start, end, cue_text))
    if not subtitles:
        raise ValueError("El SRT no contiene subtítulos con texto válidos.")
    return subtitles


def discover_voices(voices_dir: Path) -> list[VoiceRecord]:
    """Encuentra pares WAV/TXT por nombre y los valida sin modificar archivos."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    stems = {item.stem for item in voices_dir.iterdir() if item.is_file() and item.suffix.lower() in {".wav", ".txt"}}
    records: list[VoiceRecord] = []
    for stem in sorted(stems, key=str.casefold):
        wav = voices_dir / f"{stem}.wav"
        txt = voices_dir / f"{stem}.txt"
        if not wav.is_file() or not txt.is_file():
            missing = "WAV" if not wav.is_file() else "TXT"
            records.append(VoiceRecord(stem, wav if wav.is_file() else None, txt if txt.is_file() else None, False, f"Falta {missing}"))
            continue
        try:
            if not read_transcript(txt):
                raise ValueError("El TXT está vacío")
            sf.info(wav)
        except Exception as error:  # SoundFile exposes backend-specific exception classes.
            records.append(VoiceRecord(stem, wav, txt, False, str(error)))
        else:
            records.append(VoiceRecord(stem, wav, txt, True))
    return records


def to_mono_audio(audio: np.ndarray) -> np.ndarray:
    """Convierte el audio generado a mono sin cambiar duración ni velocidad."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("El modelo devolvió audio vacío o con una forma no compatible.")
    return samples.astype(np.float32, copy=False)


def mix_timeline(clips: Iterable[tuple[Subtitle, np.ndarray]], sample_rate: int) -> tuple[np.ndarray, list[str]]:
    """Mezcla clips posicionados por SRT y conserva silencios y solapamientos."""
    clips = list(clips)
    if not clips:
        return np.zeros(0, dtype=np.float32), []
    total_samples = max(
        round(max(cue.end_seconds, cue.start_seconds + len(audio) / sample_rate) * sample_rate)
        for cue, audio in clips
    )
    master = np.zeros(total_samples, dtype=np.float32)
    warnings: list[str] = []
    last_audio_end = 0.0
    for cue, audio in sorted(clips, key=lambda item: item[0].start_seconds):
        if cue.start_seconds < last_audio_end:
            warnings.append(f"Subtítulo {cue.index}: se mezcla con el anterior porque el audio previo aún no terminó.")
        start = round(cue.start_seconds * sample_rate)
        end = start + len(audio)
        master[start:end] += audio
        last_audio_end = max(last_audio_end, cue.start_seconds + len(audio) / sample_rate)
    peak = float(np.max(np.abs(master))) if len(master) else 0.0
    if peak > 1.0:
        master /= peak
        warnings.append("Se normalizó el WAV maestro para evitar saturación durante una mezcla.")
    return master, warnings
