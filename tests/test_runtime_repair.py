import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.utils.logger import log

spec = importlib.util.spec_from_file_location("avc_installer", Path(__file__).resolve().parents[1] / "setup/install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class RuntimeRepairTests(unittest.TestCase):
    def test_legacy_console_cannot_abort_processing(self):
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        with patch("sys.stdout", stream):
            log("hardware", "Survey ── Tagalog / ∑")
        stream.flush()
        self.assertIn(b"Survey", buffer.getvalue())

    def test_failed_rebuild_restores_previous_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".venv").mkdir()
            (root / ".venv/original.txt").write_text("previous environment")
            def fail(command, **kwargs):
                (root / ".venv").mkdir()
                raise subprocess.CalledProcessError(1, command)
            with patch.object(installer, "ROOT", root), patch.object(installer, "choose_python", return_value="python"), patch.object(installer.subprocess, "run", side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    installer.main()
            self.assertTrue((root / ".venv/original.txt").is_file())
            self.assertEqual(len(list(root.glob(".venv-before-setup-*-failed"))), 1)
