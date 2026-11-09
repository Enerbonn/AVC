import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.batch import (
    BatchQueue,
    FrameDecodingError,
    NoSlideDetectedError,
    crop_reference,
)


class Capture:
    def __init__(self, image_at, opened=True, wrong_position=False):
        self.image_at = image_at
        self.opened = opened
        self.wrong_position = wrong_position
        self.index = 0
        self.attempts = []
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.index = value
        self.attempts.append(value)
        return True

    def read(self):
        image = self.image_at(self.index)
        return image is not None, image

    def get(self, key):
        return self.index + (2 if self.wrong_position else 1)

    def release(self):
        self.released = True


class CropTests(unittest.TestCase):
    def metadata(self, fps=10, count=4000):
        return dict(
            width=400, height=240, fps=fps, frame_count=count, duration=count / fps
        )

    def slide(self):
        image = np.full((240, 400, 3), 245, np.uint8)
        cv2.putText(
            image,
            "Example slide",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        return image

    def black(self):
        return np.zeros((240, 400, 3), np.uint8)

    def test_accepts_real_detector_full_frame_slide(self):
        cap = Capture(lambda _: self.slide())
        with patch("src.batch.cv2.VideoCapture", return_value=cap):
            roi, time = crop_reference("recording.mp4", self.metadata(), 17)
        self.assertEqual(roi, [0, 0, 400, 240])
        self.assertEqual(time, 17)
        self.assertEqual(cap.attempts, [170])
        self.assertTrue(cap.released)

    def test_delayed_slides_use_offsets_relative_to_initial_reference(self):
        offsets = [0, 5, 30, 120, 300]
        for offset in offsets:
            with self.subTest(offset=offset):
                frame = (17 + offset) * 10
                cap = Capture(
                    lambda index: self.slide() if index == frame else self.black()
                )
                with patch("src.batch.cv2.VideoCapture", return_value=cap):
                    roi, time = crop_reference("recording.mp4", self.metadata(), 17)
                self.assertEqual(time, 17 + offset)
                self.assertEqual(
                    cap.attempts,
                    [(17 + v) * 10 for v in offsets[: offsets.index(offset) + 1]],
                )

    def test_skips_out_of_duration_instead_of_clamping_to_final_frame(self):
        cap = Capture(lambda _: self.black())
        with (
            patch("src.batch.cv2.VideoCapture", return_value=cap),
            self.assertRaises(NoSlideDetectedError),
        ):
            crop_reference("recording.mp4", self.metadata(fps=1, count=15), 5)
        self.assertEqual(cap.attempts, [5, 10])

    def test_deduplicates_rounded_frame_indices(self):
        cap = Capture(lambda _: self.black())
        with (
            patch("src.batch.cv2.VideoCapture", return_value=cap),
            self.assertRaises(NoSlideDetectedError),
        ):
            crop_reference("recording.mp4", self.metadata(fps=0.05, count=20), 1)
        self.assertEqual(cap.attempts, [0, 2, 6, 15])

    def test_all_decoding_failures_have_distinct_error(self):
        cap = Capture(lambda _: None)
        with (
            patch("src.batch.cv2.VideoCapture", return_value=cap),
            self.assertRaises(FrameDecodingError),
        ):
            crop_reference("recording.mp4", self.metadata(), 5)
        self.assertTrue(cap.released)
        self.assertEqual(cap.attempts, [50, 100, 350, 1250, 3050])

    def test_later_reference_can_recover_after_failed_decode(self):
        cap = Capture(lambda index: None if index == 50 else self.slide())
        with patch("src.batch.cv2.VideoCapture", return_value=cap):
            _, time = crop_reference("recording.mp4", self.metadata(), 5)
        self.assertEqual(time, 10)

    def test_no_slide_with_partial_decode_failure_is_reported_as_no_slide(self):
        cap = Capture(lambda index: None if index == 50 else self.black())
        with (
            patch("src.batch.cv2.VideoCapture", return_value=cap),
            self.assertRaisesRegex(NoSlideDetectedError, "1 reference frame"),
        ):
            crop_reference("recording.mp4", self.metadata(), 5)

    def test_wrong_seek_position_is_decoding_failure(self):
        cap = Capture(lambda _: self.slide(), wrong_position=True)
        with (
            patch("src.batch.cv2.VideoCapture", return_value=cap),
            self.assertRaises(FrameDecodingError),
        ):
            crop_reference("recording.mp4", self.metadata(), 5)

    def test_closed_decoder_is_not_reported_as_no_slide(self):
        cap = Capture(lambda _: None, opened=False)
        with (
            patch("src.batch.cv2.VideoCapture", return_value=cap),
            self.assertRaises(FrameDecodingError),
        ):
            crop_reference("recording.mp4", self.metadata(), 5)
        self.assertTrue(cap.released)

    def test_no_valid_reference_does_not_decode_tail(self):
        with (
            patch("src.batch.cv2.VideoCapture") as capture,
            self.assertRaisesRegex(NoSlideDetectedError, "earlier reference"),
        ):
            crop_reference("recording.mp4", self.metadata(fps=1, count=3), 5)
        capture.assert_not_called()

    def test_explicit_manual_crop_bypasses_automatic_detector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "recording.mp4"
            video.write_bytes(b"fixture")
            roi = [10, 20, 350, 190]
            queue = BatchQueue(root / "queue.json", root / "output")
            queue.enqueue(
                [{"video": str(video), "content_roi": roi}],
                {"automatic_export": False, "slides_only": False},
            )
            (root / "session").mkdir()
            session = {
                "folder": str(root / "session"),
                "content_roi": roi,
                "markers": [],
            }
            with (
                patch("src.batch.prepare_video", return_value=self.metadata()),
                patch("src.batch.crop_reference") as auto,
                patch("src.batch.detect", return_value=session) as detect,
            ):
                queue.run()
            auto.assert_not_called()
            self.assertEqual(detect.call_args.kwargs["content_roi"], roi)
            job = queue.snapshot()["jobs"][0]
            self.assertEqual(job["content_roi"], roi)
            self.assertEqual(job["crop_source"], "manual")
            self.assertEqual(job["status"], "needs_review", job.get("error"))


if __name__ == "__main__":
    unittest.main()
