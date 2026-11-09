"""Destination persistence and real export routing, without private inputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.export_location import ExportLocation, writable_folder
from src.batch import BatchQueue
from src.pdf_export import export_pdf_parts
from src.pipeline import finalize
import test_cancellation


class ExportLocationTests(unittest.TestCase):
    def test_first_use_remember_restart_and_ask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            preferences = root / "private" / "settings.json"
            location = ExportLocation(preferences, root / "default")
            self.assertFalse(location.snapshot()["configured"])
            with self.assertRaisesRegex(ValueError, "Choose an export folder"):
                location.resolve()
            selected = root / "Exports with spaces ü"
            location.save(str(selected), "saved")
            restarted = ExportLocation(preferences, root / "other-default")
            self.assertEqual(restarted.resolve(), selected.resolve())
            restarted.save(str(selected), "ask")
            with self.assertRaisesRegex(ValueError, "Choose an export folder"):
                restarted.resolve()
            self.assertEqual(restarted.resolve(str(root / "one-off")), root / "one-off")
            self.assertEqual(restarted.snapshot()["folder"], str(selected))

    def test_invalid_folder_does_not_replace_preferences(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            location = ExportLocation(root / "settings.json", root / "out")
            previous = location.save(str(root / "good"), "saved")
            file = root / "not-a-folder"
            file.write_text("keep")
            for folder in ("", "relative/path", str(file)):
                with self.assertRaises(ValueError):
                    location.save(folder, "saved")
                self.assertEqual(location.snapshot(), previous)
            with patch("src.export_location.tempfile.TemporaryFile", side_effect=PermissionError):
                with self.assertRaisesRegex(ValueError, "Cannot write"):
                    writable_folder(str(root / "good"))

    def test_real_single_pdf_and_ocr_use_chosen_folder_and_preserve_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = test_cancellation.CancellationTests().fixture(temp)
            destination = root / "chosen"
            session["export_root"] = str(destination)
            marks = [{"frame_idx": 0, "kind": "manual", "ai": True}]
            first = export_pdf_parts(session, marks, lambda *a, **k: None)
            second = export_pdf_parts(session, marks, lambda *a, **k: None)
            self.assertNotEqual(first["folder"], second["folder"])
            self.assertEqual(Path(first["folder"]).parent, destination)
            self.assertTrue(Path(first["parts"][0]["file"]).is_file())
            with patch("src.pipeline.read_image", return_value=("Plain text", 0.99, [])):
                ocr = finalize(session, marks, lambda *a, **k: None)
            self.assertEqual(Path(ocr["folder"]).parent, destination)
            self.assertTrue((Path(ocr["folder"]) / "transcription.txt").is_file())
            self.assertTrue((Path(ocr["folder"]) / "AI-marked").is_dir())
            self.assertFalse((root / "exports").exists())

    def test_batch_new_destination_preserves_completed_jobs_and_reuses_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = test_cancellation.CancellationTests().fixture(temp)
            queue = BatchQueue(root / "queue.json", root / "old")
            other = root / "other.avi"
            other.write_bytes(Path(session["video"]).read_bytes())
            queue.enqueue([session["video"], str(other)], {"task": "pdf", "slides_only": False})
            jobs = queue.snapshot()["jobs"]
            old_pdf = root / "old.pdf"
            old_pdf.write_bytes(b"preserved")
            queue.update(jobs[0]["id"], status="completed", outputs=[str(old_pdf)])
            session["markers"] = [{"frame_idx": 0, "kind": "manual", "included": True, "ai": False}]
            saved = root / "session.json"
            saved.write_text(json.dumps(session))
            queue.update(jobs[1]["id"], session_path=str(saved))
            queue.set_output_root(root / "new")
            with patch("src.batch.crop_reference", return_value=([0, 0, 160, 100], 0)), \
                 patch("src.batch.detect") as detect, patch("src.batch.finalize") as ocr:
                queue.run()
            detect.assert_not_called()
            ocr.assert_not_called()
            state = queue.snapshot()
            self.assertEqual(state["jobs"][0]["outputs"], [str(old_pdf)])
            self.assertEqual(old_pdf.read_bytes(), b"preserved")
            self.assertEqual(state["jobs"][1]["status"], "completed")
            self.assertEqual(Path(state["jobs"][1]["outputs"][0]).parent, root / "new")
            with patch.object(queue, "running", return_value=True):
                with self.assertRaisesRegex(ValueError, "Pause or finish"):
                    queue.set_output_root(root / "forbidden-change")
