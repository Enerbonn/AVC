import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.app import Application
from src.pipeline import detect, frame_range, time_label, validate_markers


class SetupTests(unittest.TestCase):
    def video(self, folder):
        path = Path(folder) / "lecture.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (400, 180)
        )
        for i in range(120):
            image = np.full((180, 400, 3), 240 if i < 60 else 40, np.uint8)
            image[:, 300:] = 0 if i % 2 else 255
            writer.write(image)
        writer.release()
        return path

    def test_prepare_does_not_detect_or_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            video = self.video(folder)
            app = Application(Path(folder) / "output", batch_path=Path(folder) / "batch.json")
            with patch.object(app, "start_recording"), patch("src.app.detect") as detector:
                session = app.prepare(str(video))
                self.assertTrue(session["preflight"])
                self.assertEqual(app.state["phase"], "setup")
                self.assertEqual(session["markers"], [])
                app.write_review()
                detector.assert_not_called()
                self.assertFalse((Path(folder) / "output").exists())
                self.assertGreater(len(app.preview_image(str(video), 60, 240)), 100)
            app.preview_capture.release()

    def test_trim_skips_intro_outro_and_preserves_source_frame_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            video = self.video(folder)
            events = []

            class Recorder:
                def __init__(self):
                    self.folder = Path(folder) / "logs"

                def event(self, event_type, **data):
                    events.append((event_type, data))

                def stage(self, *a, **k):
                    pass

            s = detect(
                video,
                Path(folder) / "output",
                lambda *a, **k: None,
                Recorder(),
                content_roi=[0, 0, 300, 180],
                start_s=1,
                end_s=3,
            )
            self.assertEqual((s["trim_start_frame"], s["trim_end_frame"]), (30, 90))
            self.assertEqual(s["content_roi"], [0, 0, 300, 180])
            self.assertEqual(s["markers"][0]["frame_idx"], 30)
            self.assertTrue(all(30 <= m["frame_idx"] < 90 for m in s["markers"]))
            complete = next(
                data for kind, data in events if kind == "detector_complete"
            )
            self.assertEqual(complete["processed_frames"], 60)
            self.assertEqual(complete["frame_idx"], 89)
            self.assertTrue(
                any(
                    kind == "detector_candidate" and data["frame_idx"] == 60
                    for kind, data in events
                )
            )
            self.assertFalse(any(m["frame_idx"] in (38, 46, 54) for m in s["markers"]))
            self.assertEqual(s["markers"][0]["timestamp_s"], 1)
            with self.assertRaisesRegex(ValueError, "selected video range"):
                validate_markers([{"frame_idx": 29}], s)

    def test_millisecond_frame_labels_round_trip_and_invalid_ranges_rejected(self):
        metadata = {"fps": 30, "duration": 232, "frame_count": 6960}
        for frame in [814, 1003, 1004, 2066, 6959]:
            seconds = round(frame / 30, 3)
            first, stop = frame_range(metadata, seconds, round((frame + 1) / 30, 3))
            self.assertEqual((first, stop), (frame, frame + 1), time_label(seconds))
        for start, end in [(3, 1), (-1, 2), (1, 233), (float("nan"), 3), (0, 0)]:
            with self.assertRaises(ValueError):
                frame_range(metadata, start, end)


if __name__ == "__main__":
    unittest.main()
