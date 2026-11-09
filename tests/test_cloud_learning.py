import base64
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.cloud_learning import run, stamp, write_srt, speech, visuals
from src.learning import Learning, engine_payload, validate_settings


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/'Class 01.mp4'
        self.video.write_bytes(b'fixture-media')
        self.service = Learning(self.root/'jobs')

    def prepare(self):
        return self.service.prepare({'videos':[str(self.video)], 'settings':{'task':'speech'}})

    def archive(self, job, tamper=False, traversal=False, failed=False, missing=False):
        registry = json.loads((Path(job['folder'])/'job.json').read_text(encoding='utf-8'))
        vid = registry['config']['videos'][0]['id']
        content = {vid+'/SOURCES.md':b'guide', vid+'/Online Class.srt':b'1\r\n00:00:00,000 --> 00:00:01,000\r\nHello\r\n\r\n'}
        if missing:
            del content[vid+'/Online Class.srt']
        files = [dict(path=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for name,value in content.items()]
        report = dict(schema=1,job_id=job['job_id'],settings=registry['config']['settings'],
                      results=[dict(id=vid,status='failed' if failed else 'complete')],files=files)
        archive = self.root/'results.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name,value in content.items():
                z.writestr(name, value if not tamper else b'x'*len(value))
            z.writestr('learning-report.json',json.dumps(report))
            if traversal:
                z.writestr('../escape.md','evil')
        return dict(job_id=job['job_id'],archive=str(archive),destination=str(self.root))

    def test_notebook_has_no_local_paths_and_compiles(self):
        job = self.prepare()
        notebook = json.loads((Path(job['folder'])/'transvision-learning.ipynb').read_text(encoding='utf-8'))
        self.assertNotIn(str(self.root), json.dumps(notebook))
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),'<notebook>','exec')
        self.assertEqual(self.video.read_bytes(),b'fixture-media')
        self.assertEqual(self.service.jobs()[0]['status'],'prepared')

    def test_payload_is_public_engine_only(self):
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(engine_payload()))) as z:
            self.assertIn('src/cloud_learning.py',z.namelist())
            self.assertIn('NOTICE',z.namelist())
            for name in z.namelist():
                self.assertTrue(name.startswith(('src/','models/')) or name in ('LICENSE','NOTICE'))
                self.assertNotIn('.local',name)

    def test_import_integrity_and_no_replacement(self):
        job = self.prepare()
        data = self.archive(job)
        output = self.service.import_results(data)
        self.assertTrue((Path(output['folder'])/job['videos'][0]['id']/'Online Class.srt').exists())
        self.assertTrue(Path(data['archive']).exists())
        with self.assertRaisesRegex(ValueError,'already imported'):
            self.service.import_results(data)

    def test_reject_tampered_output_without_partial_import(self):
        job = self.prepare()
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.service.import_results(self.archive(job,tamper=True))
        self.assertFalse(list(self.root.glob('.transvision-import-*')))
        self.assertFalse(list(self.root.glob('AVC Context*')))

    def test_reject_traversal_failed_or_missing_output(self):
        job = self.prepare()
        for option in ('traversal','failed','missing'):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.service.import_results(self.archive(job,**{option:True}))

    def test_reject_duplicate_inputs_and_bad_settings(self):
        with self.assertRaises(ValueError):
            self.service.prepare({'videos':[str(self.video),str(self.video)]})
        for value in ({'model':'invalid-model'},{'sensitivity':True},{'language':'made-up'}):
            with self.assertRaises(ValueError):
                validate_settings(value)

    def test_cloud_only_and_srt_hour_boundary(self):
        with self.assertRaisesRegex(RuntimeError,'only on Kaggle'):
            run({})
        self.assertEqual(stamp(3599.9999),'01:00:00,000')
        path=self.root/'speech.srt'
        write_srt(path,[dict(start=3599.9999,end=3601,text='  Kumusta class  ')])
        self.assertIn(b'01:00:00,000 --> 01:00:01,000\r\nKumusta class',path.read_bytes())
        with self.assertRaises(ValueError):
            write_srt(path,[dict(start=2,end=1,text='bad')])

    def test_speech_boundary_and_checkpoint_reuse_without_real_model(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import wave
        settings=validate_settings({})
        settings.update(chunk_seconds=2,overlap_seconds=1)
        output=self.root/'speech';output.mkdir()
        calls=[]
        class FakeModel:
            def transcribe(inner,waveform,**kwargs):
                calls.append(kwargs)
                # Second chunk begins at source second 1. Word at local 0.2
                # belongs to the previous core and must be discarded.
                words=([SimpleNamespace(start=0.1,end=0.5,word=' First')]
                       if len(calls)==1 else [SimpleNamespace(start=0.2,end=0.7,word=' Duplicate'),
                                             SimpleNamespace(start=1.1,end=1.5,word=' Next')])
                return iter([SimpleNamespace(words=words)]),None
        def fake_ffmpeg(command,**kwargs):
            with wave.open(command[-1],'wb') as audio:
                audio.setparams((1,2,16000,0,'NONE','not compressed'))
                audio.writeframes(b'\0\0'*16000)
        cache=self.root/'cache'
        with patch('src.cloud_learning.subprocess.run',side_effect=fake_ffmpeg):
            self.assertEqual(speech(self.video,output,cache,FakeModel(),settings,4),2)
            self.assertEqual(speech(self.video,output,cache,FakeModel(),settings,4),2)
        self.assertEqual(len(calls),2)
        text=(output/'Online Class.srt').read_text(encoding='utf-8')
        self.assertNotIn('Duplicate',text)
        self.assertIn('00:00:02,100',text)

    def test_real_visual_adapter_exports_timestamped_pdf(self):
        import cv2
        import numpy as np
        from pypdf import PdfReader
        video=self.root/'board.avi'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),4,(480,270))
        self.assertTrue(writer.isOpened())
        for frame in range(40):
            image=np.full((270,480,3),35,np.uint8)
            cv2.putText(image,'x + y = 2',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(240,240,240),2)
            if frame>=20:
                cv2.putText(image,'x = 1',(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(240,240,240),2)
            writer.write(image)
        writer.release()
        output=self.root/'visual-result';output.mkdir()
        names=visuals(video,output,self.root/'visual-cache',validate_settings({'task':'visual'}))
        self.assertTrue(names)
        pdf=PdfReader(output/names[0])
        self.assertGreater(len(pdf.pages),0)
        self.assertIn('00:00:', '\n'.join(p.extract_text() for p in pdf.pages))


if __name__=='__main__':
    unittest.main()
