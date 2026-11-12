"""Clona una voz española a partir de un WAV y su transcripción.

Ejemplo:
    ./.venv/Scripts/python.exe ./clonar_es.py --text "Buenos días."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REFERENCE_AUDIO = ROOT / "miVoz.wav"
DEFAULT_REFERENCE_TEXT = ROOT / "miVoz.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clona una voz y sintetiza texto en español.")
    parser.add_argument("--text", required=True, help="Texto que se convertirá a voz.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "voz_clonada_es.wav")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", type=Path, default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Límite conservador para una GPU de 6 GB (predeterminado: 256).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    reference_audio = args.reference_audio.resolve()
    reference_text_path = args.reference_text.resolve()
    output_path = args.output.resolve()

    for path, description in (
        (model_path, "modelo"),
        (reference_audio, "audio de referencia"),
        (reference_text_path, "transcripción de referencia"),
    ):
        if not path.exists():
            raise SystemExit(f"No se encontró {description}: {path}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible en PyTorch.")

    reference_text = reference_text_path.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise SystemExit("La transcripción de referencia está vacía.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Cargando el modelo Base en FP16 con SDPA...")
    try:
        tts = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map="cuda:0",
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        with torch.inference_mode():
            wavs, sample_rate = tts.generate_voice_clone(
                text=args.text,
                language="Spanish",
                ref_audio=str(reference_audio),
                ref_text=reference_text,
                x_vector_only_mode=False,
                max_new_tokens=args.max_new_tokens,
            )
    except torch.OutOfMemoryError as error:
        raise SystemExit(
            "VRAM insuficiente. Reduce --max-new-tokens a 128 o utiliza "
            "Qwen3-TTS-12Hz-0.6B-Base."
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wavs[0], sample_rate)
    print(f"WAV guardado: {output_path}")
    print(f"Pico de VRAM reservada: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
