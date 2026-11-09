import tempfile, unittest, wave, json, time
from pathlib import Path
from unittest.mock import patch
from src.learning import validate_settings, Learning, notebook
from src.processing import process, LocalJobs

class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.audio=self.root/'speech.wav'
        with wave.open(str(self.audio),'wb') as f:
            f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(b'\0\0'*64000)
    def test_audio_only_export_no_model(self):
        out=self.root/'out';settings=validate_settings({'exports':['audio']})
        process([self.audio],out,settings)
        folder=out/'01-speech'
        with wave.open(str(folder/'Audio.wav')) as f:self.assertEqual(f.getnframes(),64000)
        self.assertNotIn('Online Class.srt',(folder/'SOURCES.md').read_text(encoding='utf-8'))
        self.assertFalse((folder/'.work').exists())
        self.assertTrue(self.audio.exists())
    def test_audio_rejects_visual_before_start(self):
        with self.assertRaisesRegex(ValueError,'Audio-only'):
            LocalJobs(self.root).start(dict(videos=[str(self.audio)],destination=str(self.root)))
        with self.assertRaisesRegex(ValueError,'Audio-only'):
            Learning(self.root/'cloud').prepare(dict(videos=[str(self.audio)]))
    def test_visual_trial_real_trim_and_timestamp_guide(self):
        import cv2,numpy as np
        video=self.root/'board.avi';w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),4,(320,180))
        for i in range(24):
            image=np.full((180,320,3),230,np.uint8)
            cv2.putText(image,'x = 2',(30,60),0,1,(0,0,0),2);w.write(image)
        w.release()
        settings=validate_settings({'exports':['pdf'],'preview_seconds':2})
        result=process([video],self.root/'trial',settings)
        self.assertEqual(result[0]['visual_seconds'],2)
        folder=self.root/'trial/01-board'
        self.assertTrue((folder/'Visual Transcript.pdf').exists())
        self.assertIn('2.000 seconds',(folder/'SOURCES.md').read_text(encoding='utf-8'))
        self.assertFalse((folder/'.work').exists())
    def test_settings_reject_invalid_and_accept_cpu_audio(self):
        for settings in ({'exports':[]},{'device':'metal'},{'preview_seconds':-1},{'preview_seconds':float('nan')}):
            with self.assertRaises(ValueError):validate_settings(settings)
        self.assertEqual(validate_settings({'exports':['audio'],'device':'cpu'})['device'],'cpu')
    def test_async_local_job_and_unique_output(self):
        jobs=LocalJobs(self.root)
        jobs.start(dict(videos=[str(self.audio)],destination=str(self.root),settings={'exports':['audio']}))
        jobs.thread.join(15)
        self.assertEqual(jobs.snapshot()['status'],'complete',jobs.snapshot())
        self.assertTrue(Path(jobs.snapshot()['folder']).exists())
    def test_generated_cloud_notebook_portable(self):
        n=notebook(dict(schema=1,job_id='a'*32,videos=[],settings=validate_settings({'provider':'colab'})))
        for c in n['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
        self.assertIn('cloud_runtime',json.dumps(n))

    def test_device_selection_without_model_download(self):
        for count,device,kind in [(0,'cpu','int8'),(1,'cuda','float16')]:
            with self.subTest(count=count), patch('ctranslate2.get_cuda_device_count',return_value=count), patch('faster_whisper.WhisperModel') as model, patch('src.processing.speech',return_value=0):
                process([self.audio],self.root/f'device-{count}',validate_settings({'exports':['srt']}))
                self.assertEqual(model.call_args.kwargs['device'],device)
                self.assertEqual(model.call_args.kwargs['compute_type'],kind)
        with patch('ctranslate2.get_cuda_device_count',return_value=0),self.assertRaisesRegex(ValueError,'CUDA'):
            process([self.audio],self.root/'forced-cuda',validate_settings({'exports':['srt'],'device':'cuda'}))

if __name__=='__main__':unittest.main()
