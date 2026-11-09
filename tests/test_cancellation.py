import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.app import Application
from src.batch import BatchQueue
from src.cancellation import JobCancelled, cancellable
from src.pdf_export import export_pdf
from src.pipeline import finalize


class CancellationTests(unittest.TestCase):
    def app(self, folder):
        return Application(
            Path(folder) / "output", batch_path=Path(folder) / "batch.json"
        )

    def test_stop_does_not_commit_late_job_result(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            started, finish = threading.Event(), threading.Event()

            def work():
                started.set()
                finish.wait(3)
                return {"folder": folder, "markers": []}

            app.launch("detecting", work)
            self.assertTrue(started.wait(2))
            self.assertTrue(app.stop()["stopping"])
            with self.assertRaisesRegex(ValueError, "running"):
                app.launch("detecting", work)
            finish.set()
            app.worker.join(3)
            self.assertFalse(app.worker.is_alive())
            self.assertIsNone(app.state["session"])
            self.assertEqual(app.state["phase"], "idle")
            self.assertIsNone(app.state["error"])
            self.assertFalse(app.state["stopping"])

    def test_closed_tabs_refresh_and_multiple_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            app.auto_close = True
            app.window_ping("one")
            app.window_ping("two")
            app.window_ping("one", True)
            self.assertFalse(app.windows_closed(app.last_client_at + 5))
            app.window_ping("two", True)
            self.assertFalse(app.windows_closed(app.last_client_at + 2))
            app.window_ping("two")  # reload within grace interval
            self.assertFalse(app.windows_closed(app.last_client_at + 5))
            app.window_ping("two", True)
            self.assertTrue(app.windows_closed(app.last_client_at + 5))
            app.auto_close = False
            self.assertFalse(app.windows_closed(app.last_client_at + 5))

    def test_background_heartbeat_does_not_cancel_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            app.auto_close = True
            app.window_ping("background")
            with patch.object(app.batch, "running", return_value=True):
                self.assertFalse(app.windows_closed(app.last_client_at + 100))
                app.window_ping("background", True)
                self.assertTrue(app.windows_closed(app.last_client_at + 5))

    def fixture(self, folder):
        path = Path(folder) / "demo.avi"
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 100))
        for i in range(3):
            w.write(np.full((100, 160, 3), 240 - i * 20, np.uint8))
        w.release()
        return {
            "video": str(path),
            "video_name": path.name,
            "folder": folder,
            "fps": 10,
            "frame_count": 3,
            "width": 160,
            "height": 100,
            "content_roi": [0, 0, 160, 100],
            "ocr_engine": "rapidocr",
            "ocr_profile": "prose",
        }

    def test_cancel_pdf_leaves_no_completed_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            s = self.fixture(folder)
            stopped = threading.Event()

            def report(*a, **kw):
                stopped.set()

            with cancellable(stopped.is_set), self.assertRaises(JobCancelled):
                export_pdf(
                    s, [{"frame_idx": i, "kind": "manual"} for i in range(3)], report
                )
            self.assertFalse(list(Path(folder).rglob("presentation_frames.pdf")))
            self.assertTrue(list(Path(folder).rglob("INCOMPLETE.txt")))

    def test_ai_folder_and_transcription_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            s = self.fixture(folder)
            with patch(
                "src.pipeline.read_image", return_value=("Plain text", 0.99, [])
            ):
                result = finalize(
                    s,
                    [
                        {"frame_idx": 0, "kind": "manual", "ai": True},
                        {"frame_idx": 2, "kind": "manual", "ai": False},
                    ],
                    lambda *a, **k: None,
                )
            out = Path(result["folder"])
            flagged = list((out / "AI-marked").glob("*.png"))
            self.assertEqual(len(flagged), 1)
            self.assertTrue(flagged[0].name.endswith("__AI.png"))
            self.assertEqual(
                flagged[0].read_bytes(), (out / flagged[0].name).read_bytes()
            )
            self.assertEqual(
                (out / "transcription.txt").read_bytes(),
                (out / "visual_transcript.txt").read_bytes(),
            )
            with zipfile.ZipFile(out / "AI_UPLOAD.zip") as z:
                self.assertIn("transcription.txt", z.namelist())
                self.assertIn(flagged[0].name, z.namelist())
            self.assertIn("transcription.txt", result["prompt"])

    def test_batch_stop_preserves_pending_and_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            q = BatchQueue(Path(folder) / "queue.json", Path(folder) / "output")
            s = self.fixture(folder)
            q.enqueue([s["video"]], {})
            job = q.snapshot()["jobs"][0]
            with cancellable(q.stop_path.exists):
                q.stop()
                q.process(job, q.snapshot()["settings"])
            state = q.snapshot()["jobs"][0]
            self.assertEqual(state["status"], "queued")
            self.assertIsNone(state["error"])
            self.assertEqual(state["outputs"], [])


if __name__ == "__main__":
    unittest.main()
