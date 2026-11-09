import tempfile, time, json
from pathlib import Path
from unittest.mock import patch
import unittest
import cv2, numpy as np
from src.video.adaptive_extractor import extract_adaptive
from src.selection import CameraFilter, thumbnail, difference, apply_detail
from src.diagnostics import TestRecorder


class AdaptiveTests(unittest.TestCase):
    def video(self, folder, images):
        path = Path(folder) / "cut.avi"
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
        for image in images:
            w.write(image)
        w.release()
        return path

    def test_refinement_preserves_exact_outgoing_and_brief_returning_slide(self):
        with tempfile.TemporaryDirectory() as folder:
            a = np.full((180, 320, 3), 245, np.uint8)
            b = np.full_like(a, 30)
            frames = [a if i < 17 or i >= 23 else b for i in range(40)]
            p = self.video(folder, frames)
            events = []
            result = extract_adaptive(
                p,
                Path(folder) / "candidates",
                [0, 0, 320, 180],
                0,
                40,
                10,
                lambda k, **v: events.append((k, v)),
                0.5,
                False,
            )
            indices = [r.frame_idx for r in result]
            for index in [0, 16, 17, 22, 23, 39]:
                self.assertIn(index, indices)
            complete = events[-1][1]
            self.assertEqual(complete["processed_frames"], 40)
            self.assertLess(complete["compared_frames"], 40)

    def test_camera_model_excludes_real_teacher_but_not_math_slide(self):
        root = Path(__file__).resolve().parents[1] / "output/9_9_26/exports/9_9_26"
        if not root.exists():
            self.skipTest("Local real lecture fixture not present")
        camera = CameraFilter()
        self.assertTrue(camera.check(cv2.imread(str(root / "02-10-04-170.png"))))
        self.assertFalse(camera.check(cv2.imread(str(root / "00-02-00-250.png"))))

    def test_changed_equation_is_not_pointer_noise(self):
        a = np.full((270, 480, 3), 245, np.uint8)
        b = a.copy()
        cv2.putText(a, "x = 1", (35, 100), 0, 1.2, (0, 0, 0), 2)
        cv2.putText(b, "x = 2", (35, 100), 0, 1.2, (0, 0, 0), 2)
        self.assertFalse(difference(thumbnail(a), thumbnail(b))["noise"])

    def test_detail_preserves_final_frame_manual_ai_and_explicit_exclusions(self):
        marks = []
        for i in range(7):
            marks.append(
                {
                    "frame_idx": i,
                    "kind": "burst_end",
                    "included": True,
                    "ai": i == 1,
                    "user_override": i == 2,
                    "visual_analysis": {
                        "page_group": 0 if i < 4 else 1,
                        "camera": i == 4,
                        "noise": i == 3,
                        "changed_fraction": 0.003,
                    },
                }
            )
        marks[2]["included"] = False
        marks[5]["kind"] = "manual"
        compact, stats = apply_detail(marks)
        self.assertEqual(
            [m["frame_idx"] for m in compact if m["included"]], [1, 3, 5, 6]
        )
        self.assertFalse(compact[4]["included"])
        self.assertFalse(compact[2]["included"])
        self.assertEqual(stats["camera_excluded"], 1)
        major, _ = apply_detail(marks, "major", 100)
        self.assertGreaterEqual(sum(m["included"] for m in major), stats["included"])

    def test_resource_sampler_records_instead_of_windows_path_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            rec = TestRecorder(folder, {"purpose": "unit sampler"})
            deadline = time.monotonic() + 3
            while rec.counts["resource_sample"] == 0 and time.monotonic() < deadline:
                time.sleep(0.03)
            rec.close()
            self.assertGreaterEqual(rec.counts["resource_sample"], 1)
            self.assertEqual(rec.counts["sampling_error"], 0)


if __name__ == "__main__":
    unittest.main()
