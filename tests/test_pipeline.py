"""Meaningful checks for timestamp identity, marking, and AI edit boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline import (
    frame_filename,
    time_label,
    timestamp_from_filename,
    validate_markers,
    enhancement_prompt,
    finalize,
)
from src.video.burst_grouper import group_bursts
from src.video.frame_extractor import ExtractedFrame


class PipelineTests(unittest.TestCase):
    def test_millisecond_carry_and_ai_filename(self):
        self.assertEqual(time_label(59.9999), "00:01:00.000")
        self.assertEqual(frame_filename(9181.3, True), "02-33-01-300__AI.png")
        self.assertNotEqual(frame_filename(1.033), frame_filename(1.067))
        self.assertNotIn("__AI", frame_filename(1.033))

    def test_ocr_can_recover_chronology_from_filenames(self):
        names = ["01-00-00-000.png", "00-00-10-067__AI.png", "00-00-10-033.png"]
        self.assertEqual(
            sorted(names, key=timestamp_from_filename), [names[2], names[1], names[0]]
        )
        with self.assertRaises(ValueError):
            timestamp_from_filename("00-62-00-000.png")

    def test_validate_orders_by_source_frame_and_ignores_client_timestamp(self):
        marks = validate_markers(
            [{"frame_idx": 31, "timestamp_s": 500, "ai": True}, {"frame_idx": 3}],
            {"fps": 30, "frame_count": 100},
        )
        self.assertEqual([m["frame_idx"] for m in marks], [3, 31])
        self.assertEqual(marks[1]["timestamp_s"], 31 / 30)
        self.assertTrue(marks[1]["ai"])

    def test_rejects_duplicate_or_outside_source(self):
        for marks in [
            [{"frame_idx": -1}],
            [{"frame_idx": 100}],
            [{"frame_idx": True}],
            [{"frame_idx": 2}, {"frame_idx": 2}],
        ]:
            with self.assertRaises(ValueError):
                validate_markers(marks, {"fps": 30, "frame_count": 100})

    def test_first_and_consecutive_slide_changes_survive_burst_grouping(self):
        sizes = [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
        classified = [
            {
                "frame": ExtractedFrame("unused", "00:00:00", i * 30),
                "is_meaningful": True,
                "content_change_frac": size,
            }
            for i, size in enumerate(sizes)
        ]
        result = group_bursts(classified, 30)
        self.assertEqual(
            [c["trigger"] for c in result],
            ["slide_change", "slide_change", "burst_end"],
        )
        self.assertEqual(result[-1]["frame"].frame_idx, 180)

    def test_prompt_lists_only_user_ai_frames(self):
        prompt = enhancement_prompt(
            [
                {
                    "filename": "00-00-01-000.png",
                    "timestamp": "00:00:01.000",
                    "ai": False,
                },
                {
                    "filename": "00-00-02-300__AI.png",
                    "timestamp": "00:00:02.300",
                    "ai": True,
                },
            ]
        )
        self.assertNotIn("00-00-01-000.png", prompt)
        self.assertIn("00-00-02-300__AI.png -> [00:00:02.300]", prompt)
        self.assertIn("Preserve every unflagged block verbatim", prompt)

    def test_empty_selection_never_starts_an_export(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("src.pipeline.configure_ocr", return_value="test"),
        ):
            with self.assertRaises(ValueError):
                finalize(
                    {
                        "folder": folder,
                        "video": "unused.mp4",
                        "fps": 30,
                        "frame_count": 20,
                    },
                    [],
                    lambda *a, **k: None,
                )
            self.assertFalse((Path(folder) / "exports").exists())


if __name__ == "__main__":
    unittest.main()
