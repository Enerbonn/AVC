import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import json, cv2, numpy as np
from src.app import Application
from src.pipeline import detect
from src.content import suggest_roi


class CropProgressTests(unittest.TestCase):
    def video(self, folder):
        p = Path(folder) / "lesson.avi"
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 240))
        for i in range(20):
            image = np.zeros((240, 400, 3), np.uint8)
            if i >= 10:
                image[40:210, 35:330] = 245
            w.write(image)
        w.release()
        return p

    def test_suggestion_uses_selected_frame_and_never_applies_without_confirmation(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            p = self.video(folder)
            app = Application(Path(folder) / "output", batch_path=Path(folder) / "batch.json")
            with patch.object(app, "start_recording"):
                s = app.prepare(str(p))
            opening = app.suggest_crop(0, s["folder"])
            later = app.suggest_crop(15, s["folder"])
            self.assertNotEqual(opening["content_roi"], later["content_roi"])
            self.assertEqual(later["timestamp_s"], 1.5)
            self.assertEqual(
                app.state["session"]["content_roi"], opening["content_roi"]
            )
            with self.assertRaises(ValueError):
                app.suggest_crop(15, "old-video")

    def test_stage_progress_and_small_candidate_proxies(self):
        with tempfile.TemporaryDirectory() as folder:
            reports = []
            p = self.video(folder)
            s = detect(
                p,
                Path(folder) / "output",
                lambda m, **d: reports.append(d),
                content_roi=[0, 0, 400, 240],
                adaptive=True,
                hide_camera=False,
            )
            self.assertFalse(s["analysis_pending"])
            self.assertTrue(
                any(d.get("stage") == "selection_analysis" for d in reports)
            )
            for path in s["candidate_paths"].values():
                self.assertEqual(cv2.imread(path).shape[:2], (360, 640))

    def test_interruption_keeps_resumable_analysis_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self.video(folder)
            with patch(
                "src.selection.analyze_selection",
                side_effect=RuntimeError("simulated interruption"),
            ):
                with self.assertRaises(RuntimeError):
                    detect(
                        p,
                        Path(folder) / "output",
                        lambda *a, **k: None,
                        adaptive=True,
                        hide_camera=False,
                    )
            path = next((Path(folder) / "output").glob("*/session.json"))
            s = json.loads(path.read_text())
            self.assertTrue(s["analysis_pending"])
            self.assertGreater(len(s["markers"]), 0)
            self.assertTrue(
                all(Path(p).exists() for p in s["candidate_paths"].values())
            )
