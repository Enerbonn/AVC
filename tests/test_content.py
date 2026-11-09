import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

from src.content import compare_content, comparison_image, crop_image, validate_roi
from src.pipeline import finalize


class ContentTests(unittest.TestCase):
    def image(self, text="x = 1"):
        image = np.full((240, 400, 3), 245, np.uint8)
        cv2.putText(
            image, text, (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 2
        )
        return image

    def test_changed_equation_is_kept(self):
        a = self.image("x = 1")
        b = self.image("x = 2")
        roi = [0, 0, 400, 240]
        self.assertFalse(
            compare_content(comparison_image(a, roi), comparison_image(b, roi))[
                "duplicate"
            ]
        )

    def test_webcam_motion_outside_content_is_ignored(self):
        a = self.image()
        b = a.copy()
        b[:, 320:] = 20
        roi = [0, 0, 300, 240]
        self.assertTrue(
            compare_content(comparison_image(a, roi), comparison_image(b, roi))[
                "duplicate"
            ]
        )
        self.assertFalse(
            compare_content(
                comparison_image(a, [0, 0, 400, 240]),
                comparison_image(b, [0, 0, 400, 240]),
            )["duplicate"]
        )

    def test_crop_cannot_escape_source(self):
        for roi in [[-1, 0, 100, 100], [0, 0, 401, 100], [0, 0, 10, 10]]:
            with self.assertRaises(ValueError):
                validate_roi(roi, 400, 240)

    def test_duplicate_text_removed_but_ai_frame_retained_and_images_all_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "lecture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1, (400, 240)
            )
            for _ in range(3):
                writer.write(self.image())
            writer.release()
            session = {
                "video": str(video),
                "folder": folder,
                "fps": 1,
                "frame_count": 3,
                "width": 400,
                "height": 240,
                "ocr_engine": "tesseract",
                "content_roi": [0, 0, 400, 240],
                "dedupe_ocr": True,
            }
            marks = [
                {"frame_idx": i, "kind": "burst_end", "ai": i == 2} for i in range(3)
            ]
            reports = []

            def report(message, **details):
                if details.get("frames_ready"):
                    self.assertEqual(
                        len(list(Path(details["export_folder"]).glob("*.png"))), 3
                    )
                reports.append(details)

            with (
                patch("src.pipeline.configure_ocr", return_value="test"),
                patch("src.pipeline.read_image", return_value=("x = 1", 98, [])),
            ):
                result = finalize(session, marks, report)
            self.assertEqual(result["frames"], 3)
            self.assertEqual(result["transcript_blocks"], 2)
            self.assertEqual(result["duplicates_skipped"], 1)
            self.assertTrue(result["entries"][2]["transcript_included"])
            text = (Path(result["folder"]) / "visual_transcript.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("[00:00:02.000] [VISUAL] | AI", text)
            self.assertNotIn("[00:00:01.000] [VISUAL]", text)
            self.assertTrue(
                any(
                    r.get("eta_seconds") is not None and r.get("stage") == "ocr"
                    for r in reports
                )
            )

    def test_new_visual_content_survives_even_when_ocr_text_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "lecture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1, (400, 240)
            )
            for image in [self.image("x = 1"), self.image("x = 2")]:
                writer.write(image)
            writer.release()
            session = {
                "video": str(video),
                "folder": folder,
                "fps": 1,
                "frame_count": 2,
                "width": 400,
                "height": 240,
                "ocr_engine": "tesseract",
                "content_roi": [0, 0, 400, 240],
                "dedupe_ocr": True,
            }
            with (
                patch("src.pipeline.configure_ocr", return_value="test"),
                patch(
                    "src.pipeline.read_image",
                    return_value=("OCR missed this equation", 98, []),
                ),
            ):
                result = finalize(
                    session,
                    [
                        {"frame_idx": 0, "kind": "opening"},
                        {"frame_idx": 1, "kind": "burst_end"},
                    ],
                    lambda *a, **k: None,
                )
            self.assertEqual(result["transcript_blocks"], 2)

    def test_optional_crop_preserves_exact_source_pixels_and_full_frame_option(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "lecture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1, (400, 240)
            )
            writer.write(self.image())
            writer.release()
            cap = cv2.VideoCapture(str(video))
            ok, source = cap.read()
            cap.release()
            self.assertTrue(ok)
            roi = [25, 30, 300, 180]
            for cropped in [False, True]:
                session = {
                    "video": str(video),
                    "folder": folder,
                    "fps": 1,
                    "frame_count": 1,
                    "width": 400,
                    "height": 240,
                    "ocr_engine": "tesseract",
                    "content_roi": roi,
                    "export_cropped": cropped,
                }
                with (
                    patch("src.pipeline.configure_ocr", return_value="test"),
                    patch(
                        "src.pipeline.read_image", return_value=("x = 1", 98, [])
                    ) as ocr,
                ):
                    result = finalize(
                        session,
                        [{"frame_idx": 0, "kind": "opening"}],
                        lambda *a, **k: None,
                    )
                output = cv2.imread(
                    str(Path(result["folder"]) / result["entries"][0]["filename"])
                )
                expected = crop_image(source, roi) if cropped else source
                self.assertTrue(np.array_equal(output, expected))
                self.assertEqual(ocr.call_args.args[0].shape, (180, 300, 3))


if __name__ == "__main__":
    unittest.main()
