import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cloud_runtime import library_path, launch, preflight
from src.learning import notebook, validate_settings


class CloudRuntimeTests(unittest.TestCase):
    def test_library_discovery_keeps_existing_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lib = root / 'nvidia/cublas/lib'
            lib.mkdir(parents=True)
            value = library_path([root, root], 'existing-libs')
            self.assertEqual(value, str(lib) + os.pathsep + 'existing-libs')

    def test_launch_sets_loader_path_before_starting_child(self):
        with patch('src.cloud_runtime.library_path', return_value='cuda-libs'), patch('src.cloud_runtime.subprocess.run') as run:
            launch({'settings': {}}, '/engine')
            self.assertEqual(run.call_args.kwargs['env']['LD_LIBRARY_PATH'], 'cuda-libs')
            self.assertEqual(run.call_args.kwargs['cwd'], '/engine')
            self.assertIn('src.cloud_runtime', run.call_args.args[0])
            self.assertTrue(run.call_args.kwargs['check'])

    def test_missing_gpu_library_fails_before_model_download(self):
        with patch('ctranslate2.get_cuda_device_count', return_value=1), patch('src.cloud_runtime.ctypes.CDLL', side_effect=OSError('missing')):
            with self.assertRaisesRegex(RuntimeError, 'updated notebook'):
                preflight({'exports': ['srt'], 'device': 'auto'})

    def test_cpu_and_visual_only_skip_gpu_libraries(self):
        with patch('src.cloud_runtime.ctypes.CDLL') as load:
            preflight({'exports': ['pdf'], 'device': 'auto'})
            preflight({'exports': ['srt'], 'device': 'cpu'})
            load.assert_not_called()

    def test_notebooks_install_gpu_libraries_only_when_needed(self):
        for settings, expected in [({'exports':['srt']}, True), ({'exports':['pdf']}, False), ({'exports':['srt'],'device':'cpu'}, False)]:
            n = notebook({'schema':1, 'job_id':'a'*32, 'videos':[], 'settings':validate_settings(settings)})
            cells = [''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
            self.assertEqual('nvidia-cublas-cu12' in cells[0], expected)
            for code in cells:
                compile(code, '<cloud-notebook>', 'exec')
