import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from src.boundaries import final_frames
from src.diagnostics import TestRecorder
from src.ocr import readable_text
from src.video.region_activity import classify_frames, _cell_activity_map
from src.video.frame_extractor import ExtractedFrame
from src.config import AMBIENT_GRID_SIZE, AMBIENT_RATIO, MIN_CONTENT_AREA_FRAC


class TrialTests(unittest.TestCase):
    def test_boundary_recovers_last_frame_even_if_detector_lags(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = str(Path(tmp) / "cut.avi")
            writer = cv2.VideoWriter(
                video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100)
            )
            for i in range(30):
                writer.write(
                    np.full((100, 100, 3), 255 if i < 17 else 0, dtype=np.uint8)
                )
            writer.release()
            session = {"video": video, "width": 100, "height": 100, "fps": 10}
            cues = [
                {"frame_idx": 0, "kind": "opening"},
                {"frame_idx": 20, "kind": "slide_change"},
            ]
            selection = [
                {"frame_idx": 0, "kind": "manual", "included": False, "ai": False}
            ]
            result = final_frames(session, selection, cues)
            self.assertEqual(result[-1]["frame_idx"], 16)
            self.assertFalse(result[0]["included"])
            self.assertEqual(len(final_frames(session, result, cues)), 2)

    def test_streaming_classifier_matches_batch_logic(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(12)
            images = [
                rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(5)
            ]
            frames = []
            grays = []
            for i, img in enumerate(images):
                path = str(Path(tmp) / f"{i}.png")
                cv2.imwrite(path, img)
                frames.append(ExtractedFrame(path, "unused", i))
                grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            grid = AMBIENT_GRID_SIZE
            acts = [_cell_activity_map(a, b, grid) for a, b in zip(grays, grays[1:])]
            mask = np.sum(acts, axis=0) / len(acts) >= AMBIENT_RATIO
            result, actual_mask = classify_frames(frames)
            np.testing.assert_array_equal(mask, actual_mask)
            expected = [
                round(float((act & ~mask).sum() / (grid * grid)), 4) for act in acts
            ]
            self.assertEqual([r["content_change_frac"] for r in result[1:]], expected)

    def test_unclear_symbols_are_explicit_and_clean_prose_unchanged(self):
        text = "Properties of Norms\n1. 向 · ≥ 0."
        clean, count = readable_text(text, [], np.zeros((30, 30, 3), dtype=np.uint8))
        self.assertTrue(clean.startswith("Properties of Norms\n"))
        self.assertNotIn("向", clean)
        self.assertIn("[unclear symbol]", clean)
        self.assertEqual(count, 1)

    def test_recorder_flushes_durable_errors_and_stage_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TestRecorder(tmp, {"video": "trial.mp4"})
            recorder.event("job_error", error="decoder unavailable")
            recorder.event("checkpoint_decision", kind="pre_change", frame_idx=16)
            recorder.stage("ocr", 2.5, frames=3)
            recorder.close("interrupted")
            data = json.loads((recorder.folder / "summary.json").read_text())
            self.assertEqual(data["status"], "interrupted")
            self.assertEqual(data["stages"][0]["seconds"], 2.5)
            self.assertIn(
                "decoder unavailable", (recorder.folder / "events.jsonl").read_text()
            )
