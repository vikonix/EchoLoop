"""Autonomous checks for pronounce/speech.py.

Two layers:
  1. Fast unit tests over the pure logic (scoring, DTW alignment, helpers).
     They never load the Wav2Vec2 model, so they run offline and quickly.
  2. An optional end-to-end runner (``python test_speech.py user.wav [ref.wav]``)
     that exercises ``analyze`` on real audio. This downloads the ~1.2 GB model
     and needs espeak-ng installed, so it is gated behind the CLI and is not run
     by the unit tests.

Run unit tests:   python -m unittest pronounce.test_speech
End-to-end check: python pronounce/test_speech.py path/to/user.wav [path/to/ref.wav]
"""

import sys
import unittest
from typing import Optional

import numpy as np

from pronounce import speech


class TestPureLogic(unittest.TestCase):
    """Tests that need no model weights."""

    def test_score_is_perfect_for_zero_distances(self):
        self.assertEqual(speech.compute_pronunciation_score(0, 0, 0), 100.0)

    def test_score_is_clamped_to_zero_for_huge_distances(self):
        score = speech.compute_pronunciation_score(10_000, 10_000, 10_000)
        self.assertEqual(score, 0.0)

    def test_score_stays_within_bounds(self):
        score = speech.compute_pronunciation_score(250, 150, 15)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_score_decreases_as_distance_grows(self):
        better = speech.compute_pronunciation_score(50, 30, 3)
        worse = speech.compute_pronunciation_score(400, 200, 25)
        self.assertGreater(better, worse)

    def test_phoneme_embeddings_shape_and_values(self):
        emb = speech.get_phoneme_embeddings("ab")
        self.assertEqual(emb.shape, (2, 1))
        self.assertEqual(emb[0][0], ord("a"))
        self.assertEqual(emb[1][0], ord("b"))

    def test_align_sequences_dtw_returns_equal_length(self):
        seq1 = [[1.0], [2.0], [3.0], [4.0]]
        seq2 = [[1.0], [4.0]]
        aligned1, aligned2 = speech.align_sequences_dtw(seq1, seq2)
        self.assertEqual(len(aligned1), len(aligned2))
        self.assertGreater(len(aligned1), 0)

    def test_clean_transcription_normalises_text(self):
        self.assertEqual(speech.clean_transcription("  Hello, WORLD!! "), "hello world")

    def test_interpolate_f0_fills_gaps(self):
        f0 = np.array([0.0, 100.0, 0.0, 200.0, 0.0])
        out = speech.interpolate_f0(f0)
        self.assertEqual(len(out), len(f0))
        self.assertTrue((out > 0).all())  # zeros between voiced frames get filled

    def test_interpolate_f0_handles_all_silent(self):
        f0 = np.zeros(5)
        out = speech.interpolate_f0(f0)
        self.assertEqual(len(out), 5)  # no crash on fully unvoiced input

    def test_prepare_waveform_downmixes_and_resamples(self):
        # 2 channels of 24 kHz audio -> mono 16 kHz.
        stereo_24k = np.ones((2, 24_000), dtype=np.float32)
        out = speech._prepare_waveform(stereo_24k, orig_sr=24_000)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.float32)
        # 1 s at 24 kHz -> ~1 s at 16 kHz.
        self.assertAlmostEqual(len(out), 16_000, delta=50)

    def test_result_has_spec_fields(self):
        result = speech.PronunciationResult(
            score=88.0, word_errors=[], prosody={"f0": [], "energy": []},
            transcription="hello world")
        self.assertTrue(hasattr(result, "score"))
        self.assertTrue(hasattr(result, "word_errors"))
        self.assertTrue(hasattr(result, "prosody"))
        self.assertTrue(hasattr(result, "transcription"))


def _run_end_to_end(user_path: str, reference_path: Optional[str]) -> None:
    """Run the full analyze() pipeline on real WAV files (loads the model)."""
    import soundfile as sf

    user_audio, user_sr = sf.read(user_path, dtype="float32")
    if reference_path:
        reference_audio, reference_sr = sf.read(reference_path, dtype="float32")
    else:
        # No reference supplied: reuse the user's own audio. A faithful repetition
        # of itself should score high — a quick sanity check of the pipeline.
        print("No reference WAV given; using the user audio as its own reference.")
        reference_audio, reference_sr = user_audio, user_sr

    expected_text = input("Expected text for this recording: ").strip()

    print("Loading Wav2Vec2 (first run downloads ~1.2 GB)...")
    speech.load_models()

    result = speech.analyze(
        user_audio=user_audio,
        expected_text=expected_text,
        reference_audio=reference_audio,
        user_sr=user_sr,
        reference_sr=reference_sr,
    )

    print(f"\nScore:         {result.score} (passed={result.passed})")
    print(f"Transcription: {result.transcription!r}")
    print(f"Problem words: {result.words_with_errors}")
    print(f"Acoustic DTW:  {result.acoustic_distance}")
    print(result.feedback)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        _run_end_to_end(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)
    else:
        unittest.main()
