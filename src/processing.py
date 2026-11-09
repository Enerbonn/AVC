"""Shared local/cloud media execution. Model downloads occur only for SRT requests."""
from pathlib import Path
import json, shutil, subprocess, threading, uuid, traceback
from src.cloud_learning import duration, speech, visuals, write_json
from src.cancellation import cancellable, check_cancel, JobCancelled

AUDIO = {'.wav','.mp3','.m4a','.flac','.ogg','.aac','.opus'}

def process(paths, output, settings, report=lambda text:None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('Install FFmpeg and ffprobe and add them to PATH, then restart AVC.')
    model=None;device='not needed';results=[]
    if 'srt' in settings['exports']:
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ValueError('Speech requires faster-whisper. Run Setup AVC.cmd first.') from e
        cuda=ctranslate2.get_cuda_device_count()>0
        choice=settings['device']
        if choice=='cuda' and not cuda: raise ValueError('CUDA is unavailable. Choose Auto or CPU.')
        device='cuda' if cuda and choice!='cpu' else 'cpu'
        report('Loading Whisper on '+device+'; first use may download the model.')
        model=WhisperModel(settings['model'],device=device,compute_type='float16' if device=='cuda' else 'int8',
                           download_root=str(Path(__file__).resolve().parents[1]/'.local/models'))
    for index,source in enumerate(map(Path,paths),1):
        check_cancel();report(f'Processing {index}/{len(paths)}: {source.name}')
        folder=output/f'{index:02d}-{source.stem[:70]}'
        folder.mkdir();cache=folder/'.work';cache.mkdir()
        total=duration(source);visual_seconds=total
        try:
            if 'srt' in settings['exports']:
                speech(source,folder,cache/'speech',model,settings,total)
            if 'audio' in settings['exports']:
                subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(source),'-vn','-ac','1','-ar','16000',
                                '-c:a','pcm_s16le',str(folder/'Audio.wav')],check=True)
            if any(e in settings['exports'] for e in ('pdf','ocr')):
                check_cancel();visual_source=source
                if settings['preview_seconds']:
                    visual_seconds=min(total,settings['preview_seconds'])
                    visual_source=cache/'visual-trial.mkv'
                    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(source),'-t',str(visual_seconds),
                                    '-map','0:v:0','-an','-c:v','ffv1',str(visual_source)],check=True)
                visuals(visual_source,folder,cache/'visual',settings)
            if (folder/'Visual OCR.txt').exists():
                ocr=(folder/'Visual OCR.txt').read_text(encoding='utf-8')
                ocr=ocr.replace('Source video: '+visual_source.name,'Source recording: '+source.name)
                ocr='\n'.join(line for line in ocr.splitlines() if not line.startswith('[IMAGE:'))+'\n'
                (folder/'Visual OCR.txt').write_text(ocr,encoding='utf-8')
            check_cancel()
            artifacts=[p for p in folder.iterdir() if p.is_file()]
            role={'.srt':'timestamped speech recognition draft','.pdf':'selected timestamped visual frames',
                  '.txt':'OCR text with source timestamps; check equations against images','.json':'structured OCR with timestamps, confidence and text regions','.wav':'16 kHz mono extracted audio'}
            guide=f'# {source.name}\n\nOriginal duration: {total:.3f} seconds. Speech covers the whole recording when selected.\n'
            guide+=f'Visual coverage requested: 0â€“{visual_seconds:.3f} seconds. Visual trial: {bool(settings["preview_seconds"])}.\n'
            guide+='All offsets start at the original recording start. Visuals are selected frames, not exhaustive coverage.\n\n'
            from urllib.parse import quote
            guide+='\n'.join(f'- [{p.name}]({quote(p.name)}): {role.get(p.suffix,"context resource")}' for p in artifacts)
            guide+='\n\nUse matching speech and visual timestamps for source tracing. OCR and speech may misread names, Taglish and equations. No summary is generated.\n'
            (folder/'SOURCES.md').write_text(guide,encoding='utf-8')
            results.append(dict(source=source.name,folder=folder.name,device=device,status='complete',
                                seconds=total,visual_seconds=visual_seconds))
        finally:
            # Delete only this job's own intermediate directory, never originals.
            assert cache.parent==folder and cache.name=='.work'
            shutil.rmtree(cache,ignore_errors=True)
    write_json(output/'processing-report.json',dict(settings=settings,results=results))
    return results

class LocalJobs:
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.Lock();self.state={'status':'idle'};self.cancel=threading.Event()
    def snapshot(self):
        with self.lock:return dict(self.state)
    def update(self,**values):
        with self.lock:self.state.update(values)
    def start(self,data):
        from src.learning import validate_settings, MEDIA
        settings=validate_settings(data.get('settings',{}))
        paths=[Path(p).resolve() for p in data.get('videos',[])]
        if not 1<=len(paths)<=30 or any(not p.is_file() or p.suffix.lower() not in MEDIA for p in paths):
            raise ValueError('Select existing video or audio files.')
        if any(p.suffix.lower() in AUDIO for p in paths) and any(e in settings['exports'] for e in ('pdf','ocr')):
            raise ValueError('Audio-only files cannot produce visual exports. Choose SRT and/or audio.')
        if len(set(paths))!=len(paths):raise ValueError('A recording was selected twice.')
        target=Path(data.get('destination','')).resolve()
        if not data.get('destination') or not target.is_dir():raise ValueError('Choose an existing output folder.')
        with self.lock:
            if self.state['status'] in ('running','stopping'):raise ValueError('Stop or finish the current local job first.')
            folder=target/('AVC-'+uuid.uuid4().hex[:12]);self.cancel=threading.Event()
            self.state=dict(status='running',folder=str(folder),message='Starting local processing',settings=settings)
        def worker():
            try:
                with cancellable(self.cancel.is_set):
                    process(paths,folder,settings,lambda t:self.update(message=t))
                self.update(status='complete',message='Local exports complete.')
            except JobCancelled:self.update(status='stopped',message='Stopped; completed files remain. Retry creates a new folder.')
            except Exception as e:self.update(status='failed',message=str(e))
        self.thread=threading.Thread(target=worker,daemon=True)
        self.thread.start()
        return self.snapshot()
    def stop(self):
        self.cancel.set()
        if self.snapshot()['status']=='running':self.update(status='stopping',message='Stopping at the next safe boundaryâ€¦')
        return self.snapshot()
