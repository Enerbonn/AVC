"""Real visual engine -> portable cloud archive -> verified AVC import."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from pypdf import PdfReader

from src.learning import Learning
from src.portable_cloud import run


class CloudRoundtripTests(unittest.TestCase):
    def test_visual_export_archive_and_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root/'input'; inputs.mkdir()
            working = root/'working'; working.mkdir()
            destination = root/'imported'; destination.mkdir()
            video = inputs/'Board lesson.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 4, (320,180))
            for i in range(12):
                image = np.full((180,320,3), 245, np.uint8)
                cv2.putText(image,'x = 2',(30,80),0,1,(0,0,0),2)
                writer.write(image)
            writer.release()
            service = Learning(root/'registry')
            job = service.prepare({'videos':[str(video)], 'settings':{'exports':['pdf'], 'device':'cpu', 'preview_seconds':2}})
            config = json.loads((Path(job['folder'])/'job.json').read_text(encoding='utf-8'))['config']
            # Redirect only provider mount locations; the media/export/ZIP/import
            # engines are real, including checksums and original job identity.
            mounts = {'/kaggle/working':working, '/kaggle/input':inputs}
            with patch('src.portable_cloud.Path', side_effect=lambda name: mounts[name]):
                run(config)
            archive = working/'learning-results.zip'
            self.assertTrue(archive.is_file())
            imported = service.import_results({'job_id':job['job_id'], 'archive':str(archive), 'destination':str(destination)})
            outputs = Path(imported['folder'])/job['videos'][0]['id']
            pdfs = list(outputs.glob('*.pdf'))
            self.assertTrue(pdfs)
            self.assertGreater(len(PdfReader(pdfs[0]).pages),0)
            self.assertIn('2.000',(outputs/'SOURCES.md').read_text(encoding='utf-8'))
            self.assertEqual(service.jobs()[0]['status'],'imported')
            self.assertTrue(video.is_file())
