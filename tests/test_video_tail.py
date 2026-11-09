import unittest
import cv2
from src.pipeline import verified_frame_count


class TailCapture:
    def __init__(self, actual, seek=True):
        self.actual = actual
        self.position = 0
        self.seek = seek
    def set(self, key, value):
        self.position = value
        return self.seek
    def get(self, key):
        return self.position
    def grab(self):
        if self.position >= self.actual:
            return False
        self.position += 1
        return True


class TailTests(unittest.TestCase):
    def test_small_terminal_overcount_corrected(self):
        self.assertEqual(verified_frame_count(TailCapture(375594), 375597, 60), 375594)
    def test_exact_count_preserved(self):
        self.assertEqual(verified_frame_count(TailCapture(600), 600, 60), 600)
    def test_large_shortfall_not_silently_truncated(self):
        self.assertEqual(verified_frame_count(TailCapture(490), 600, 60), 600)
    def test_unseekable_video_not_guessed(self):
        self.assertEqual(verified_frame_count(TailCapture(597, False), 600, 60), 600)


if __name__ == "__main__":
    unittest.main()
