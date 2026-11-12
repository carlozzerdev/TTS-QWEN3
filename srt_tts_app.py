"""Aplicación Windows para doblar un SRT con una voz clonada de Qwen3-TTS."""

from __future__ import annotations

import gc
import json
from datetime import datetime
import os
from pathlib import Path
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# PyInstaller places PyTorch DLLs under _internal\torch\lib.  Register that
# directory before importing torch so that c10.dll can resolve CUDA/runtime DLLs.
if getattr(sys, "frozen", False):
    bundled_torch_lib = Path(sys._MEIPASS) / "torch" / "lib"
    if bundled_torch_lib.is_dir():
        os.add_dll_directory(str(bundled_torch_lib))
        os.environ["PATH"] = f"{bundled_torch_lib}{os.pathsep}{os.environ.get('PATH', '')}"

import numpy as np
import soundfile as sf
import torch

# Qwen3-TTS uses Transformers for text and audio, not image models.  Keeping
# torchvision out of the frozen app avoids loading its optional C++ NMS operator.
from transformers.utils import import_utils as transformers_import_utils
transformers_import_utils._torchvision_available = False
from qwen_tts import Qwen3TTSModel
from srt_tts_core import Subtitle, VoiceRecord, discover_voices, mix_timeline, parse_srt, read_transcript, to_mono_audio


APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
VOICES_DIR = APP_ROOT / "voices"
DEFAULT_MODEL = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"


class SrtTtsEngine:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        self.model_path = model_path
        self.tts: Qwen3TTSModel | None = None
        self.device = "cuda:0"
        self.prompt = None
        self.voice: VoiceRecord | None = None

    @staticmethod
    def _is_oom(error: BaseException) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()

    def _release(self) -> None:
        self.prompt = None
        self.tts = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, device: str) -> None:
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"No se encontró el modelo Base: {self.model_path}")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.tts = Qwen3TTSModel.from_pretrained(
            str(self.model_path), device_map=device, dtype=dtype, attn_implementation="sdpa"
        )
        self.device = device

    def prepare_voice(self, voice: VoiceRecord, status) -> None:
        if not voice.enabled or voice.audio_path is None or voice.transcript_path is None:
            raise ValueError(f"La voz no está disponible: {voice.reason}")
        self.voice = voice
        reference_text = read_transcript(voice.transcript_path)
        if torch.cuda.is_available():
            try:
                status("Cargando modelo 1.7B en GPU con SDPA…")
                self._load("cuda:0")
            except Exception as error:
                if not self._is_oom(error):
                    raise
                self._release()
                status("VRAM insuficiente; cambiando el modelo a CPU…")
                self._load("cpu")
        else:
            status("CUDA no disponible; cargando el modelo en CPU…")
            self._load("cpu")
        try:
            self.prompt = self.tts.create_voice_clone_prompt(
                ref_audio=str(voice.audio_path), ref_text=reference_text, x_vector_only_mode=False
            )
        except Exception as error:
            if self.device.startswith("cuda") and self._is_oom(error):
                self._release()
                status("VRAM insuficiente al preparar la voz; cambiando a CPU…")
                self._load("cpu")
                self.prompt = self.tts.create_voice_clone_prompt(
                    ref_audio=str(voice.audio_path), ref_text=reference_text, x_vector_only_mode=False
                )
            else:
                raise

    def generate(self, text: str) -> tuple[np.ndarray, int]:
        if self.tts is None or self.prompt is None:
            raise RuntimeError("La voz no está preparada.")
        max_new_tokens = min(512, max(192, len(text.replace("\n", " ")) * 4))
        try:
            wavs, sample_rate = self.tts.generate_voice_clone(
                text=text, language="Spanish", voice_clone_prompt=self.prompt, max_new_tokens=max_new_tokens
            )
            return np.asarray(wavs[0]), int(sample_rate)
        except Exception as error:
            if not (self.device.startswith("cuda") and self._is_oom(error)):
                raise
            # Volver a preparar el prompt evita conservar tensores de GPU en el reintento.
            voice = self.voice
            self._release()
            self._load("cpu")
            assert voice is not None and voice.audio_path is not None and voice.transcript_path is not None
            self.prompt = self.tts.create_voice_clone_prompt(
                ref_audio=str(voice.audio_path),
                ref_text=read_transcript(voice.transcript_path),
                x_vector_only_mode=False,
            )
            wavs, sample_rate = self.tts.generate_voice_clone(
                text=text, language="Spanish", voice_clone_prompt=self.prompt, max_new_tokens=max_new_tokens
            )
            return np.asarray(wavs[0]), int(sample_rate)

    def close(self) -> None:
        self._release()


class Application(ttk.Frame):
    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root, padding=14)
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.records: dict[str, VoiceRecord] = {}
        self.voice_var = tk.StringVar()
        self.srt_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(APP_ROOT / "outputs"))
        self.status_var = tk.StringVar(value="Selecciona una voz y un archivo SRT.")
        self._build()
        self.refresh_voices()
        self.after(100, self._poll_events)

    def _build(self) -> None:
        self.root.title("Qwen3-TTS · SRT sincronizado")
        self.root.minsize(690, 420)
        self.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text="Voz").grid(row=0, column=0, sticky="w", pady=4)
        self.voice_combo = ttk.Combobox(self, state="readonly", textvariable=self.voice_var)
        self.voice_combo.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(self, text="Actualizar voces", command=self.refresh_voices).grid(row=0, column=2, sticky="ew")
        ttk.Label(self, text="Archivo SRT").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=self.srt_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(self, text="Examinar…", command=self.choose_srt).grid(row=1, column=2, sticky="ew")
        ttk.Label(self, text="Carpeta salida").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(self, text="Examinar…", command=self.choose_output).grid(row=2, column=2, sticky="ew")
        self.generate_button = ttk.Button(self, text="Generar WAV sincronizado", command=self.start_generation)
        self.generate_button.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 6))
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew")
        ttk.Label(self, textvariable=self.status_var, wraplength=650).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 4))
        ttk.Label(self, text="Avisos y voces no disponibles:").grid(row=6, column=0, columnspan=3, sticky="w")
        self.log = tk.Text(self, height=12, wrap="word", state="disabled")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew")
        self.rowconfigure(7, weight=1)

    def _write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh_voices(self) -> None:
        records = discover_voices(VOICES_DIR)
        self.records = {record.name: record for record in records}
        valid = [record.name for record in records if record.enabled]
        self.voice_combo["values"] = valid
        if self.voice_var.get() not in valid:
            self.voice_var.set(valid[0] if valid else "")
        self._write_log(f"Carpeta de voces: {VOICES_DIR}")
        for record in records:
            if not record.enabled:
                self._write_log(f"No disponible — {record.name}: {record.reason}")
        if not records:
            self._write_log("No hay voces. Añade pares como voices/Ana.wav y voices/Ana.txt.")

    def choose_srt(self) -> None:
        selected = filedialog.askopenfilename(title="Elegir subtítulos", filetypes=[("Subtítulos SRT", "*.srt"), ("Todos", "*.*")])
        if selected:
            self.srt_var.set(selected)

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Elegir carpeta de salida")
        if selected:
            self.output_var.set(selected)

    def start_generation(self) -> None:
        voice = self.records.get(self.voice_var.get())
        srt_path = Path(self.srt_var.get())
        output_root = Path(self.output_var.get())
        if voice is None or not voice.enabled:
            messagebox.showerror("Voz requerida", "Selecciona una voz válida de la carpeta voices.")
            return
        if not srt_path.is_file():
            messagebox.showerror("SRT requerido", "Selecciona un archivo SRT existente.")
            return
        self.generate_button.configure(state="disabled")
        self.progress.configure(value=0)
        threading.Thread(target=self._generate_worker, args=(voice, srt_path, output_root), daemon=True).start()

    def _generate_worker(self, voice: VoiceRecord, srt_path: Path, output_root: Path) -> None:
        engine = SrtTtsEngine()
        try:
            subtitles = parse_srt(srt_path)
            run_dir = output_root / f"{srt_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
            clips_dir = run_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=False)
            warnings: list[str] = []
            failures: list[dict[str, object]] = []
            self.events.put(("status", f"Leyendo {len(subtitles)} subtítulos…"))
            engine.prepare_voice(voice, lambda message: self.events.put(("status", message)))
            positioned: list[tuple[Subtitle, np.ndarray]] = []
            sample_rate: int | None = None
            for position, subtitle in enumerate(subtitles, start=1):
                self.events.put(("status", f"Generando {position}/{len(subtitles)}: subtítulo {subtitle.index}"))
                try:
                    raw, rate = engine.generate(subtitle.text)
                    if sample_rate is None:
                        sample_rate = rate
                    elif sample_rate != rate:
                        raw = np.asarray(raw, dtype=np.float32)
                        raw = __import__("librosa").resample(raw, orig_sr=rate, target_sr=sample_rate)
                    natural = to_mono_audio(raw)
                    natural_duration = len(natural) / sample_rate
                    if natural_duration > subtitle.duration_seconds:
                        warnings.append(
                            f"Subtítulo {subtitle.index}: el audio natural dura {natural_duration:.2f}s, "
                            f"pero su intervalo SRT dura {subtitle.duration_seconds:.2f}s; no se alteró la velocidad."
                        )
                    sf.write(clips_dir / f"{position:03d}_srt_{subtitle.index}.wav", natural, sample_rate)
                    positioned.append((subtitle, natural))
                except Exception as error:
                    failures.append({"subtitle": subtitle.index, "start": subtitle.start_seconds, "error": str(error)})
                self.events.put(("progress", position / len(subtitles) * 100))
            if sample_rate is None:
                raise RuntimeError("No se pudo generar ningún subtítulo; no se creó un WAV maestro.")
            master, mix_warnings = mix_timeline(positioned, sample_rate)
            warnings.extend(mix_warnings)
            master_path = run_dir / f"{srt_path.stem}_master.wav"
            sf.write(master_path, master, sample_rate)
            report = {
                "srt": str(srt_path.resolve()), "voice": voice.name, "model": str(DEFAULT_MODEL),
                "device_final": engine.device, "master_wav": str(master_path.resolve()),
                "subtitles_total": len(subtitles), "subtitles_generated": len(positioned),
                "failures": failures, "warnings": warnings,
            }
            (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.events.put(("done", (master_path, warnings, failures, engine.device)))
        except Exception as error:
            self.events.put(("error", f"{error}\n\n{traceback.format_exc()}"))
        finally:
            engine.close()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(str(value))
                elif kind == "progress":
                    self.progress.configure(value=float(value))
                elif kind == "done":
                    master, warnings, failures, device = value
                    self.generate_button.configure(state="normal")
                    self.status_var.set(f"Terminado con {device}: {master}")
                    for warning in warnings:
                        self._write_log("Aviso — " + warning)
                    for failure in failures:
                        self._write_log(f"Error — subtítulo {failure['subtitle']}: {failure['error']}")
                    messagebox.showinfo("Generación terminada", f"WAV maestro:\n{master}\n\nFallos: {len(failures)}")
                elif kind == "error":
                    self.generate_button.configure(state="normal")
                    self.status_var.set("La generación no pudo completarse.")
                    self._write_log(str(value))
                    messagebox.showerror("Error de generación", str(value).split("\n\n", 1)[0])
        except queue.Empty:
            pass
        self.after(100, self._poll_events)


def main() -> None:
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
