import copy
import json
import tempfile
import threading
import unittest
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from src.app import Application, ThreadingHTTPServer, handler_for
from src.selection import apply_sensitivity, session_sensitivity, validate_sensitivity


def candidates():
    return [{"frame_idx": i, "kind": "burst_end", "included": True,
             "visual_analysis": {"page_group": 0, "camera": False,
                                 "noise": i == 5, "changed_fraction": delta}}
            for i, delta in enumerate([1, .01, .003, .001, .0004, .02])]


class SensitivityTests(unittest.TestCase):
    def test_full_range_is_monotonic_and_anchor_counts_are_distinct(self):
        marks = candidates()
        original = copy.deepcopy(marks)
        previous = set()
        counts = {}
        for value in range(101):
            result, stats = apply_sensitivity(marks, value)
            kept = {m["frame_idx"] for m in result if m["included"]}
            self.assertTrue(previous <= kept, value)
            self.assertEqual(len(kept), stats["included"])
            self.assertNotIn(5, kept, "Cursor-only candidate must never replace completed content")
            previous = kept
            counts[value] = len(kept)
        self.assertEqual([counts[n] for n in [0, 50, 100]], [1, 3, 5])
        self.assertEqual(marks, original)

    def test_unsorted_candidates_keep_the_last_useful_frame(self):
        result, _ = apply_sensitivity(list(reversed(candidates())), 0)
        self.assertEqual([m["frame_idx"] for m in result if m["included"]], [4])

    def test_manual_ai_overrides_without_analysis_and_camera_filter(self):
        marks = candidates() + [
            {"frame_idx": 6, "kind": "manual", "included": True},
            {"frame_idx": 7, "ai": True, "included": True},
            {"frame_idx": 8, "user_override": True, "included": False},
            {"frame_idx": 9, "included": True, "visual_analysis": {
                "page_group": 1, "camera": True, "noise": False, "changed_fraction": 1}},
        ]
        for value in [0, 37, 50, 67, 100]:
            result, _ = apply_sensitivity(marks, value)
            self.assertTrue(result[6]["included"])
            self.assertTrue(result[7]["included"])
            self.assertFalse(result[8]["included"])
            self.assertFalse(result[9]["included"])
        result, _ = apply_sensitivity(marks, 0, False)
        self.assertTrue(result[9]["included"])

    def test_old_reviews_translate_to_one_control(self):
        self.assertEqual(session_sensitivity({"detail_level": "completed", "detail_sensitivity": 99}), 0)
        self.assertEqual(session_sensitivity({"detail_level": "major", "detail_sensitivity": 50}), 50)
        self.assertEqual(session_sensitivity({"detail_level": "steps", "detail_sensitivity": 100}), 100)
        self.assertEqual(session_sensitivity({"sensitivity": 37, "detail_level": "completed"}), 37)

    def test_invalid_values_and_missing_analysis(self):
        for value in [True, False, -1, 101, 2.5, "67", None, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                validate_sensitivity(value)
        with self.assertRaises(ValueError):
            apply_sensitivity([{"frame_idx": 0, "kind": "opening"}], 50)

    def test_browser_and_export_selection_agree_for_every_slider_value(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node unavailable for cross-language parity check")
        marks = candidates() + [{"frame_idx": 9, "kind": "manual", "included": False}]
        marks[1]['visual_analysis']['motion']={'version':1,'summary':{'force_keep':True,'preserve_min_sensitivity':50,'reason':'temporary emphasis'}}
        marks[2]['visual_analysis']['motion']={'version':1,'summary':{'suppressed':True,'reason':'repeated loop'}}
        expected = [[m["frame_idx"] for m in apply_sensitivity(marks, n)[0] if m["included"]]
                    for n in range(101)]
        module = Path(__file__).resolve().parents[1] / "web/selection.js"
        code = "const s=require(process.argv[1]);let t='';process.stdin.on('data',d=>t+=d);process.stdin.on('end',()=>{const m=JSON.parse(t);console.log(JSON.stringify(Array.from({length:101},(_,n)=>s.select(m,n).filter(m=>m.included).map(m=>m.frame_idx))));});"
        result = subprocess.run([node, "-e", code, str(module)], input=json.dumps(marks),
                                text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), expected)

    def test_server_save_roundtrip_and_invalid_input_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            app = Application(folder)
            session = {"video": "not-needed.avi", "folder": folder, "markers": candidates(),
                       "fps": 1, "frame_count": 10, "width": 400, "height": 240,
                       "content_roi": [0, 0, 400, 240], "selection_analysis_roi": [0, 0, 400, 240],
                       "sensitivity": 50, "revision": 0, "ocr_engine": "tesseract"}
            app.state.update(session=session, phase="review")
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            app.base_url = f"http://127.0.0.1:{server.server_port}"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch("src.app.available_engines", return_value=["tesseract"]):
                    def post(value, marks):
                        req = Request(app.base_url + "/api/save", data=json.dumps({
                            "markers": marks, "sensitivity": value}).encode(),
                            headers={"Content-Type": "application/json", "X-TransVision-Token": app.token})
                        return urlopen(req)
                    before = copy.deepcopy(session)
                    with self.assertRaises(HTTPError) as error:
                        post(101, [])
                    error.exception.close()
                    self.assertEqual(session, before)
                    marks, _ = apply_sensitivity(candidates(), 67)
                    with post(67, marks) as response:
                        self.assertEqual(response.status, 200)
                    saved = json.loads((Path(folder) / "session.json").read_text())
                    self.assertEqual(saved["sensitivity"], 67)
                    self.assertEqual(saved["markers"], session["markers"])
                    self.assertNotIn("detail_level", saved)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
