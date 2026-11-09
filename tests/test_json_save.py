import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.pipeline import save_json


class JSONSaveTests(unittest.TestCase):
    def test_transient_windows_read_lock_retries_without_exposing_partial_json(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "queue.json"
            target.write_text('{"old":true}', encoding="utf-8")
            replace = Path.replace
            calls = []
            def locked(source, destination):
                calls.append(source)
                if len(calls) <= 2:
                    self.assertEqual(json.loads(target.read_text()), {"old": True})
                    raise PermissionError("Transient Windows read lock")
                return replace(source, destination)
            with patch.object(Path, "replace", locked), patch("src.pipeline.time.sleep"):
                save_json(target, {"new": [1, 2, 3]})
            self.assertEqual(json.loads(target.read_text()), {"new": [1, 2, 3]})
            self.assertEqual(len(calls), 3)

    def test_persistent_lock_preserves_original_and_recoverable_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "queue.json"
            target.write_text('{"old":true}', encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("Persistent lock")), patch("src.pipeline.time.sleep"):
                with self.assertRaises(PermissionError):
                    save_json(target, {"new": True})
            self.assertEqual(json.loads(target.read_text()), {"old": True})
            self.assertEqual(json.loads(next(Path(folder).glob('*.tmp')).read_text()), {"new": True})


if __name__ == "__main__":
    unittest.main()
