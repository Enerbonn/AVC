import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

from src.batch import BatchQueue, settings, presentation_content, crop_reference


class BatchTests(unittest.TestCase):
    def queue(self, folder, **options):
        root = Path(folder)
        videos = [root / "one.mp4", root / "two.mp4"]
        for p in videos:
            p.write_bytes(b"test video placeholder")
        queue = BatchQueue(root / "queue.json", root / "output")
        queue.enqueue([str(p) for p in videos], {"slides_only": False, **options})
        return queue

    def metadata(self):
        return {"frame_count": 10, "fps": 1, "width": 400, "height": 240}

    def session(self, folder):
        marks = [{"frame_idx": 0, "included": True, "kind": "opening",
                  "visual_analysis": {"page_group": 0, "camera": False,
                                      "noise": False, "changed_fraction": 1}}]
        return {"folder": str(folder), "markers": marks, "content_roi": [0, 0, 400, 240]}

    def test_pdf_only_never_runs_ocr_and_persists_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(folder, task="pdf")
            root = Path(folder); pdf = root / "presentation_frames.pdf"
            from pypdf import PdfWriter
            writer = PdfWriter(); writer.add_blank_page(width=200, height=200)
            with pdf.open("wb") as stream:
                writer.write(stream)
            with patch("src.batch.prepare_video", return_value=self.metadata()), \
                 patch("src.batch.crop_reference", return_value=([0, 0, 400, 240], 5)), \
                 patch("src.batch.detect", return_value=self.session(root)), \
                 patch("src.batch.export_pdf", return_value={"folder": str(root), "filename": pdf.name}) as exporter, \
                 patch("src.batch.finalize") as ocr:
                queue.run()
                ocr.assert_not_called()
                self.assertEqual(exporter.call_count, 2)
            restored = BatchQueue(queue.path, root / "output").snapshot()
            self.assertTrue(all(j["status"] == "completed" for j in restored["jobs"]))
            self.assertTrue(all(Path(j["outputs"][0]).is_file() for j in restored["jobs"]))
            self.assertFalse(restored["running"])

    def test_ocr_only_never_exports_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(folder, task="ocr")
            root = Path(folder)
            with patch("src.batch.prepare_video", return_value=self.metadata()), \
                 patch("src.batch.crop_reference", return_value=([0, 0, 400, 240], 5)), \
                 patch("src.batch.detect", return_value=self.session(root)), \
                 patch("src.batch.export_pdf") as pdf, \
                 patch("src.batch.finalize", return_value={"folder": str(root)}) as ocr:
                queue.run()
                pdf.assert_not_called()
                self.assertEqual(ocr.call_count, 2)
            self.assertTrue(all(j["status"] == "completed" for j in queue.snapshot()["jobs"]))

    def test_review_first_creates_no_exports_and_continues_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(folder, automatic_export=False)
            with patch("src.batch.prepare_video", return_value=self.metadata()), \
                 patch("src.batch.crop_reference", return_value=([0, 0, 400, 240], 5)), \
                 patch("src.batch.detect", return_value=self.session(Path(folder))), \
                 patch("src.batch.export_pdf") as pdf, patch("src.batch.finalize") as ocr:
                queue.run(); pdf.assert_not_called(); ocr.assert_not_called()
            self.assertTrue(all(j["status"] == "needs_review" for j in queue.snapshot()["jobs"]))

    def test_failure_does_not_block_next_file_and_retry_keeps_success(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(folder, automatic_export=False)
            with patch("src.batch.prepare_video", side_effect=[ValueError("Unreadable video"), self.metadata()]), \
                 patch("src.batch.crop_reference", return_value=([0, 0, 400, 240], 5)), \
                 patch("src.batch.detect", return_value=self.session(Path(folder))):
                queue.run()
            self.assertEqual([j["status"] for j in queue.snapshot()["jobs"]], ["failed", "needs_review"])
            queue.retry()
            self.assertEqual([j["status"] for j in queue.snapshot()["jobs"]], ["queued", "needs_review"])

    def test_pause_after_current_file_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(folder, automatic_export=False)
            def scan(*args, **kwargs):
                queue.pause()
                return self.session(Path(folder))
            with patch("src.batch.prepare_video", return_value=self.metadata()), \
                 patch("src.batch.crop_reference", return_value=([0, 0, 400, 240], 5)), \
                 patch("src.batch.detect", side_effect=scan):
                queue.run()
            self.assertEqual([j["status"] for j in queue.snapshot()["jobs"]], ["needs_review", "queued"])
            with patch("src.batch.prepare_video", return_value=self.metadata()), \
                 patch("src.batch.crop_reference", return_value=([0, 0, 400, 240], 5)), \
                 patch("src.batch.detect", return_value=self.session(Path(folder))):
                queue.run()
            self.assertTrue(all(j["status"] == "needs_review" for j in queue.snapshot()["jobs"]))

    def test_options_reject_invalid_values_before_queue_mutation(self):
        for options in [{"task": "wrong"}, {"sensitivity": 101}, {"crop_reference_seconds": float("nan")}, {"sample_seconds": -1}]:
            with self.assertRaises(ValueError):
                settings(options)
        self.assertTrue(presentation_content(np.full((90, 160, 3), 240, np.uint8)))
        self.assertFalse(presentation_content(np.zeros((90, 160, 3), np.uint8)))

    def test_appending_jobs_preserves_prior_job_options(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(folder, task="pdf")
            queue.enqueue([str(Path(folder) / "one.mp4")], {"task": "ocr", "sensitivity": 67})
            jobs = queue.snapshot()["jobs"]
            self.assertEqual(len(jobs), 3)
            self.assertEqual([j["settings"]["task"] for j in jobs], ["pdf", "pdf", "ocr"])
            self.assertEqual(jobs[-1]["settings"]["sensitivity"], 67)


if __name__ == "__main__":
    unittest.main()
