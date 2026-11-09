import tempfile, unittest
from unittest.mock import patch
from pathlib import Path
import cv2, numpy as np
from pypdf import PdfReader
from src.content import optimize_markers
from src.pdf_export import export_pdf, export_pdf_parts
from src.video.frame_extractor import extract_frames
from src.utils.hardware import HardwareConfig


class ExportTests(unittest.TestCase):
    def make_video(self, folder, images, fps=10):
        path = Path(folder) / "lecture.avi"
        h, w = images[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
        )
        for img in images:
            writer.write(img)
        writer.release()
        return {
            "video": str(path),
            "video_name": path.name,
            "folder": folder,
            "fps": fps,
            "frame_count": len(images),
            "width": w,
            "height": h,
            "content_roi": [0, 0, w - 100, h],
        }

    def test_outgoing_duplicate_is_removed_but_completed_new_content_and_manual_kept(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            a = np.full((180, 400, 3), 240, np.uint8)
            b = a.copy()
            cv2.putText(b, "x=2", (20, 60), 0, 1, (0, 0, 0), 2)
            s = self.make_video(folder, [a, a, b, b])
            marks = [
                {"frame_idx": i, "kind": kind, "included": True, "ai": False}
                for i, kind in enumerate(
                    ["opening", "pre_change", "pre_change", "manual"]
                )
            ]
            updated, _ = optimize_markers(s, marks, s["content_roi"])
            self.assertEqual(
                [m["included"] for m in updated], [True, False, True, True]
            )

    def test_detection_ignores_camera_outside_crop_but_saves_real_slide_change(self):
        with tempfile.TemporaryDirectory() as folder:
            images = []
            for i in range(30):
                img = np.full((180, 400, 3), 240 if i < 20 else 40, np.uint8)
                img[:, 300:] = 0 if i % 2 else 255
                images.append(img)
            s = self.make_video(folder, images)
            with patch("src.video.frame_extractor.log"):
                frames = extract_frames(
                    s["video"],
                    str(Path(folder) / "candidates"),
                    HardwareConfig("cpu", None, 0, "int8", None),
                    content_roi=s["content_roi"],
                )
            self.assertEqual([f.frame_idx for f in frames], [0, 20])

    def test_pdf_crop_timestamp_optional_and_exact_page_order(self):
        with tempfile.TemporaryDirectory() as folder:
            a = np.full((180, 400, 3), 240, np.uint8)
            s = self.make_video(folder, [a, a])
            marks = [
                {"frame_idx": 1, "kind": "manual"},
                {"frame_idx": 0, "kind": "opening"},
            ]
            for enabled in [True, False]:
                s["pdf_timestamps"] = enabled
                result = export_pdf(s, marks, lambda *a, **k: None)
                pdf = PdfReader(Path(result["folder"]) / result["filename"])
                self.assertEqual(len(pdf.pages), 2)
                first = pdf.pages[0].extract_text()
                if enabled:
                    self.assertIn("Video timestamp: 00:00:00.000", first)
                    self.assertIn(
                        "Video timestamp: 00:00:00.100", pdf.pages[1].extract_text()
                    )
                else:
                    self.assertNotIn("Video timestamp", first)
                # Lossless embedded slide matches crop dimensions; camera not embedded.
                self.assertEqual(pdf.pages[0].images[0].image.size, (300, 180))

    def test_interactive_export_returns_bounded_parts_and_preserves_master(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.full((180, 400, 3), 240, np.uint8)
            session = self.make_video(folder, [image] * 5)
            session["max_pdf_pages"] = 2
            result = export_pdf_parts(session, [{"frame_idx": n, "kind": "manual"} for n in range(5)], lambda *a, **k: None)
            self.assertEqual([p["pages"] for p in result["parts"]], [2, 2, 1])
            self.assertEqual(result["filename"], "lecture_01.pdf")
            self.assertTrue((Path(result["folder"]) / result["master_filename"]).is_file())
            text = [page.extract_text() for part in result["parts"] for page in PdfReader(part["file"]).pages]
            self.assertIn("Video timestamp: 00:00:00.400", text[-1])
