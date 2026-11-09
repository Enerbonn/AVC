"""Local API access boundaries using temporary synthetic state."""

import http.client
import tempfile
import threading
import unittest
from pathlib import Path

from src.app import Application, ThreadingHTTPServer, handler_for


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.app = Application(
            self.root / "output", batch_path=self.root / "queue.json"
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.app))
        self.app.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.tmp.cleanup()

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        body = "{}" if method == "POST" else None
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        data = response.read()
        result = response.status, dict(response.getheaders()), data
        connection.close()
        return result

    def test_private_routes_require_token_and_correct_host(self):
        self.assertEqual(self.request("/api/status")[0], 403)
        self.assertEqual(self.request("/api/status?token=wrong")[0], 403)
        self.assertEqual(
            self.request("/", headers={"Host": "attacker.example"})[0], 403
        )
        self.assertEqual(
            self.request(
                "/api/status", headers={"X-TransVision-Token": self.app.token}
            )[0],
            200,
        )

    def test_foreign_origin_cannot_mutate_state(self):
        self.assertEqual(self.request("/api/reset", "POST")[0], 403)
        headers = {
            "X-TransVision-Token": self.app.token,
            "Origin": "https://attacker.example",
        }
        self.assertEqual(self.request("/api/reset", "POST", headers)[0], 403)
        headers["Origin"] = self.app.base_url
        self.assertEqual(self.request("/api/reset", "POST", headers)[0], 200)

    def test_browser_security_headers(self):
        for route in ["/", "/manual", "/logo.png", "/editor.js", "/learn", "/learn.js", "/learn.css"]:
            status, headers, _ = self.request(route)
            self.assertEqual(status, 200)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(headers["Referrer-Policy"], "no-referrer")
            self.assertEqual(headers["X-Frame-Options"], "DENY")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_cloud_routes_require_auth_and_prepare_download(self):
        import json
        from src.learning import Learning
        self.app.learning = Learning(self.root/'cloud')
        self.assertEqual(self.request('/api/cloud/jobs')[0],403)
        self.assertEqual(self.request('/api/local/status')[0],403)
        self.assertEqual(self.request('/api/local/start','POST')[0],403)
        self.assertEqual(self.request('/api/local/stop','POST')[0],403)
        self.assertEqual(self.request('/api/cloud/prepare','POST')[0],403)
        headers={'X-TransVision-Token':self.app.token,'Origin':self.app.base_url}
        self.assertEqual(self.request('/api/cloud/jobs',headers=headers)[0],200)
        video=self.root/'demo.mp4'
        video.write_bytes(b'synthetic')
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        connection.request('POST','/api/cloud/prepare',json.dumps({'videos':[str(video)]}),headers)
        response=connection.getresponse()
        self.assertEqual(response.status,200)
        job=json.loads(response.read());connection.close()
        # An old job's saved notebook must not bypass application runtime fixes.
        (Path(job['folder'])/'transvision-learning.ipynb').write_text('{}',encoding='utf-8')
        status,download_headers,body=self.request('/api/cloud/notebook?job_id='+job['job_id'],headers=headers)
        self.assertEqual(status,200)
        self.assertIn('attachment',download_headers['Content-Disposition'])
        self.assertEqual(json.loads(body)['nbformat'],4)
        downloaded=json.loads(body)
        self.assertEqual(downloaded['metadata']['avc_runtime'],'cuda12-cudnn9-v2')
        code='\n'.join(''.join(c['source']) for c in downloaded['cells'] if c['cell_type']=='code')
        self.assertIn('nvidia-cublas-cu12',code)
        self.assertIn('src.cloud_runtime',code)
        self.assertEqual(self.request('/api/cloud/notebook?job_id=../private',headers=headers)[0],404)

    def test_large_authenticated_cached_analysis_request(self):
        import json
        headers={"X-TransVision-Token":self.app.token,"Origin":self.app.base_url}
        connection=http.client.HTTPConnection("127.0.0.1",self.server.server_port)
        connection.request("POST","/api/reset",json.dumps({"padding":"x"*8000001}),headers)
        response=connection.getresponse()
        self.assertEqual(response.status,200)
        response.read();connection.close()

    def test_export_preferences_require_auth_and_batch_start_uses_destination(self):
        import json
        from unittest.mock import patch

        self.assertEqual(self.request("/api/export-settings")[0], 403)
        headers = {"X-TransVision-Token": self.app.token, "Origin": self.app.base_url}
        destination = self.root / "chosen"
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("POST", "/api/export-settings",
                           json.dumps({"folder": str(destination), "mode": "saved"}), headers)
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["folder"], str(destination))
        connection.close()
        video = self.root / "demo.mp4"
        video.write_bytes(b"synthetic")
        self.app.batch.enqueue([str(video)], {})
        with patch.object(self.app.batch, "start") as start:
            self.assertEqual(self.request("/api/batch/start", "POST", headers)[0], 200)
        start.assert_called_once()
        self.assertEqual(self.app.batch.snapshot()["output_root"], str(destination))
        self.assertEqual(self.app.batch.snapshot()["jobs"][0]["output_root"], str(destination))

    def test_export_path_cannot_escape_output_folder(self):
        folder = self.root / "export"
        folder.mkdir()
        (self.root / "private.txt").write_text("Synthetic private sentinel")
        self.app.state["session"] = {
            "video": str(self.root / "video.avi"),
            "folder": str(self.root),
            "frame_count": 1,
            "result": {"folder": str(folder)},
            "pdf_result": {"folder": str(folder)},
        }
        headers = {"X-TransVision-Token": self.app.token}
        for prefix in ["/output/", "/pdf/"]:
            status, _, data = self.request(prefix + "..%2Fprivate.txt", headers=headers)
            self.assertEqual(status, 403)
            self.assertNotIn(b"Synthetic private sentinel", data)


if __name__ == "__main__":
    unittest.main()
