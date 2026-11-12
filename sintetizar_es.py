"""Genera un archivo WAV en español con un modelo Qwen3-TTS local.

Ejemplo:
    ./.venv/Scripts/python.exe ./sintetizar_es.py --text "Hola, esta es una prueba de voz en español." --output ./outputs/prueba.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sintetiza texto español con Qwen3-TTS.")
    parser.add_argument("--text", required=True, help="Texto que se convertirá a voz.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "voz_es.wav",
        help="Ruta del WAV de salida (predeterminada: outputs/voz_es.wav).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Carpeta local del modelo descargado.",
    )
    parser.add_argument(
        "--speaker",
        default="Ryan",
        help="Voz predefinida (predeterminada: Ryan).",
    )
    parser.add_argument(
        "--instruct",
        default="Habla con un tono natural, claro y agradable.",
        help="Instrucción de estilo para el modelo 1.7B.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Límite de tokens de audio; reduce este valor si falta VRAM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    output_path = args.output.resolve()

    if not model_path.is_dir():
        raise SystemExit(
            f"No se encontró el modelo local: {model_path}\n"
            "Descárgalo primero con el comando indicado en la guía."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible en PyTorch. Revisa la instalación cu126 y el controlador NVIDIA.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Cargando el modelo en FP16 con SDPA...")
    try:
        tts = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map="cuda:0",
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        with torch.inference_mode():
            wavs, sample_rate = tts.generate_custom_voice(
                text=args.text,
                language="Spanish",
                speaker=args.speaker,
                instruct=args.instruct,
                max_new_tokens=args.max_new_tokens,
            )
    except torch.OutOfMemoryError as error:
        raise SystemExit(
            "Memoria de GPU insuficiente para el modelo 1.7B. Usa el modelo "
            "Qwen3-TTS-12Hz-0.6B-CustomVoice y pásalo con --model."
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wavs[0], sample_rate)
    print(f"WAV guardado: {output_path}")
    print(f"Pico de VRAM reservada: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
