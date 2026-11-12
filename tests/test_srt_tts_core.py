import tempfile
import unittest
from pathlib import Path

import numpy as np

from srt_tts_core import Subtitle, mix_timeline, parse_srt, read_transcript, to_mono_audio


class SrtCoreTests(unittest.TestCase):
    def test_parse_srt_preserves_multiline_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.srt"
            source.write_text("1\n00:00:01,000 --> 00:00:02,500\nHola\nmundo\n\n2\n00:00:04,000 --> 00:00:05,000\nAdiós\n", encoding="utf-8")
            subtitles = parse_srt(source)
        self.assertEqual(2, len(subtitles))
        self.assertEqual("Hola\nmundo", subtitles[0].text)
        self.assertEqual(1.5, subtitles[0].duration_seconds)

    def test_audio_keeps_natural_duration(self):
        audio = to_mono_audio(np.ones(4096, dtype=np.float32))
        self.assertEqual(4096, len(audio))

    def test_timeline_preserves_silence(self):
        cue = Subtitle(1, 1.0, 2.0, "Hola")
        master, warnings = mix_timeline([(cue, np.ones(1000, dtype=np.float32))], 1000)
        self.assertEqual(2000, len(master))
        self.assertTrue(np.allclose(master[:1000], 0))
        self.assertEqual([], warnings)

    def test_timeline_does_not_cut_an_audio_that_exceeds_its_cue(self):
        cue = Subtitle(1, 1.0, 1.5, "Hola")
        master, _ = mix_timeline([(cue, np.ones(1000, dtype=np.float32))], 1000)
        self.assertEqual(2000, len(master))

    def test_read_transcript_accepts_utf16le_without_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "voice.txt"
            transcript.write_bytes("Hola, ¿qué tal?".encode("utf-16-le"))
            self.assertEqual("Hola, ¿qué tal?", read_transcript(transcript))


if __name__ == "__main__":
    unittest.main()
