"""Offline preparation and verified import for the cloud learning workflow."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import threading
import uuid
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus'}
SCHEMA = 1
CLOUD_RUNTIME_REVISION = 'cuda12-cudnn9-v2'


def slug(value):
    return re.sub(r'[^a-zA-Z0-9]+', '-', value).strip('-')[:70] or 'lecture'


def validate_settings(value):
    model = value.get('model', 'large-v3')
    language = value.get('language', 'tl')
    task = value.get('task', 'both')
    crop_mode = value.get('crop_mode', 'full')
    sensitivity = value.get('sensitivity', 50)
    if model not in ('tiny','base','small','medium','large-v3', 'large-v3-turbo'):
        raise ValueError('Choose Large V3 or Large V3 Turbo.')
    if language not in ('tl', 'en', 'auto'):
        raise ValueError('Choose Taglish / Filipino, English, or automatic language detection.')
    if task not in ('both', 'speech', 'visual'):
        raise ValueError('Choose speech, visuals, or both.')
    if crop_mode not in ('full', 'auto'):
        raise ValueError('Choose full-frame capture or automatic slide crop.')
    if isinstance(sensitivity, bool) or not isinstance(sensitivity, int) or not 0 <= sensitivity <= 100:
        raise ValueError('Visual detail must be between 0 and 100.')
    roi=value.get('crop_roi')
    if roi is not None and (not isinstance(roi,list) or len(roi)!=4 or any(isinstance(v,bool) or not isinstance(v,int) for v in roi) or min(roi[:2])<0 or min(roi[2:])<=0):
        raise ValueError('Crop must contain x, y, width and height in pixels.')
    exports=value.get('exports', ['srt','pdf'] if task=='both' else ['srt'] if task=='speech' else ['pdf'])
    if not isinstance(exports,list) or not exports or len(set(exports))!=len(exports) or any(e not in ('srt','pdf','ocr','audio') for e in exports):
        raise ValueError('Choose SRT, visual PDF, OCR text and/or extracted audio.')
    device=value.get('device','auto');provider=value.get('provider','kaggle')
    preview=value.get('preview_seconds',0)
    if device not in ('auto','cpu','cuda') or provider not in ('kaggle','colab'):
        raise ValueError('Choose a supported device and cloud provider.')
    if isinstance(preview,bool) or not isinstance(preview,(float,int)) or not 0<=preview<=86400:
        raise ValueError('Visual trial must be 0 (whole video) to 86400 seconds.')
    return dict(crop_roi=roi,exports=exports,device=device,provider=provider,preview_seconds=preview,
                model=model, language=language, task=task, sensitivity=sensitivity, crop_mode=crop_mode,
                slides_only=bool(value.get('slides_only', False)),
                motion_mode='summary', chunk_seconds=900, overlap_seconds=2)


def engine_payload():
    """Explicit application-source allowlist: no output, accounts or local settings."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT/'src').rglob('*.py')):
            if path.name in ('app.py', 'learning.py'):
                continue
            z.write(path, path.relative_to(ROOT).as_posix())
        for name in ('LICENSE', 'NOTICE', 'models/face_detection_yunet_2023mar.onnx', 'models/YUNET-LICENSE.txt'):
            z.write(ROOT/name, name)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def notebook(config):
    setup = '''from pathlib import Path
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'faster-whisper==1.2.1', 'opencv-python-headless>=4.8,<5',
    'scikit-image>=0.21,<1', 'pytesseract>=0.3.10,<1', 'rapidocr==3.9.2',
    'psutil>=7,<8', 'reportlab>=4,<6', 'pypdf>=5,<7', 'onnxruntime>=1.20,<2'], check=True)
'''
    if 'srt' in config['settings']['exports'] and config['settings']['device'] != 'cpu':
        setup += "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nvidia-cublas-cu12>=12,<13', 'nvidia-cudnn-cu12>=9,<10'], check=True)\n"
    runtime = f'''import base64, io, json, sys, zipfile
from pathlib import Path
engine = Path('/kaggle/temp/avc-engine') if Path('/kaggle/working').exists() else Path('/content/avc-engine')
engine.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({engine_payload()!r}))) as archive:
    archive.extractall(engine)
sys.path.insert(0, str(engine))
from src.cloud_runtime import launch
launch(json.loads({json.dumps(config, ensure_ascii=False)!r}), engine)
'''
    def cell(kind, text):
        value = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        return value
    return dict(nbformat=4, nbformat_minor=5,
                metadata={'avc_runtime': CLOUD_RUNTIME_REVISION, 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                cells=[cell('markdown', '# AVC — Audio Visual Context\n\nKeep notebook and input dataset private. Kaggle: attach the listed recordings, enable Internet, and Save & Run All. Colab: create /content/avc-input and upload the listed recordings there, then Run all. Select an available GPU for faster speech; CPU is supported. Download learning-results.zip when finished. All timestamps refer to the original recording start.\n'),
                       cell('markdown', 'For Colab: upload input files using its Files sidebar into /content/avc-input (create the folder first), or copy them there from your mounted Drive. Select a GPU runtime for speech acceleration. For Kaggle: attach a private dataset. Free GPU availability is not guaranteed.\n'), cell('code', setup), cell('code', runtime)])


class Learning:
    def __init__(self, state_root):
        self.root = Path(state_root).resolve()
        self.lock = threading.RLock()

    def current_notebook(self, job_id):
        """Downloads always use the installed app's engine, including older jobs."""
        if not re.fullmatch('[a-f0-9]{32}', job_id):
            raise ValueError('Choose a prepared cloud job.')
        with self.lock:
            record = json.loads((self.root/job_id/'job.json').read_text(encoding='utf-8'))
            return json.dumps(notebook(record['config']), ensure_ascii=False).encode('utf-8')

    def prepare(self, data):
        with self.lock:
            values = data.get('videos', [])
            if not isinstance(values, list) or not 1 <= len(values) <= 30:
                raise ValueError('Select between 1 and 30 recordings.')
            settings = validate_settings(data.get('settings', {}))
            paths = [Path(v).expanduser().resolve() for v in values]
            from src.processing import AUDIO
            if any(p.suffix.lower() in AUDIO for p in paths) and any(e in settings['exports'] for e in ('pdf','ocr')):
                raise ValueError('Audio-only inputs cannot produce visual exports.')
            if len({str(p).casefold() for p in paths}) != len(paths):
                raise ValueError('A recording was selected twice.')
            if len({p.name.casefold() for p in paths}) != len(paths):
                raise ValueError('Recordings need distinct filenames for Kaggle upload.')
            for p in paths:
                if not p.is_file() or p.suffix.lower() not in MEDIA:
                    raise ValueError('Select existing video recordings.')
            job_id = uuid.uuid4().hex
            folder = self.root/job_id
            folder.mkdir(parents=True)
            videos = [dict(id=f'{i:02d}-{slug(p.stem)}', name=p.name, bytes=p.stat().st_size)
                      for i, p in enumerate(paths, 1)]
            config = dict(schema=SCHEMA, job_id=job_id, settings=settings, videos=videos)
            # Local paths belong only in the private registry, never in the notebook.
            registry = dict(config=config, local_sources=[str(p) for p in paths], status='prepared')
            (folder/'job.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
            (folder/'transvision-learning.ipynb').write_text(json.dumps(notebook(config), ensure_ascii=False), encoding='utf-8')
            (folder/'UPLOAD.txt').write_text('Upload these original recordings to one PRIVATE Kaggle dataset:\n\n'
                +'\n'.join(str(p) for p in paths)
                +'\n\nImport transvision-learning.ipynb into a private notebook. Attach your dataset.\n'
                +'Select GPU T4 x2 and Internet on; use Save & Run All.\n'
                +'Download learning-results.zip from Output and import it in TransVision.\n', encoding='utf-8')
            provider=settings['provider']
            instructions=('# AVC cloud instructions: '+provider+'\n\n'
                +'1. Sign into your '+provider+' account and open/import transvision-learning.ipynb.\n'
                +('2. Create a private dataset with the original selected recordings and attach it.\n3. Enable Internet, choose an available GPU for speech, and Save & Run All.\n' if provider=='kaggle' else '2. Create /content/avc-input using the Colab Files sidebar; upload the original selected recordings there (or copy from mounted Drive).\n3. Choose a GPU runtime if available and Run all. Keep the session connected.\n')
                +'4. Download learning-results.zip and import it into the matching AVC job.\n\nFree compute has variable availability and limits. No account credentials are stored.\n')
            (folder/'INSTRUCTIONS.md').write_text(instructions,encoding='utf-8')
            return dict(job_id=job_id, folder=str(folder), videos=videos, status='prepared')

    def jobs(self):
        result = []
        with self.lock:
            for p in sorted(self.root.glob('*/job.json'), key=lambda p:p.stat().st_mtime, reverse=True):
                record = json.loads(p.read_text(encoding='utf-8'))
                result.append(dict(job_id=record['config']['job_id'], videos=record['config']['videos'],
                                   status=record['status'], folder=str(p.parent), imported=record.get('imported')))
        return result

    def import_results(self, data):
        with self.lock:
            job_id = data.get('job_id', '')
            if not re.fullmatch('[a-f0-9]{32}', job_id):
                raise ValueError('Choose a prepared cloud job.')
            job_path = self.root/job_id/'job.json'
            record = json.loads(job_path.read_text(encoding='utf-8'))
            if record['status'] == 'imported':
                raise ValueError('This job was already imported. Open its existing results.')
            source = Path(data.get('archive', '')).expanduser().resolve()
            target = Path(data.get('destination', '')).expanduser().resolve()
            if not data.get('destination') or not target.is_dir():
                raise ValueError('Choose an existing destination folder.')
            destination = target/('AVC Context - '+job_id[:8])
            if destination.exists():
                raise ValueError('The result folder already exists; nothing was replaced.')
            with zipfile.ZipFile(source) as z:
                entries = z.infolist()
                if len(entries) > 10000 or sum(e.file_size for e in entries) > 2*1024**3:
                    raise ValueError('Result archive is too large.')
                names = [e.filename for e in entries]
                if len(names) != len(set(names)) or len(names) != len({n.casefold() for n in names}):
                    raise ValueError('Archive contains duplicate paths.')
                for name in names:
                    p = PurePosixPath(name)
                    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                        raise ValueError('Unsafe archive path.')
                if 'learning-report.json' not in names:
                    raise ValueError('This is not an AVC result archive.')
                if z.getinfo('learning-report.json').file_size > 1024*1024:
                    raise ValueError('Invalid learning report size.')
                report = json.loads(z.read('learning-report.json'))
                if report.get('schema') != SCHEMA or report.get('job_id') != job_id:
                    raise ValueError('Results belong to a different job.')
                if report.get('settings') != record['config']['settings']:
                    raise ValueError('Cloud settings do not match the prepared notebook.')
                expected = {v['id'] for v in record['config']['videos']}
                results = report.get('results', [])
                if len(results) != len(expected) or {v['id'] for v in results} != expected:
                    raise ValueError('Results do not match all selected recordings.')
                if any(v.get('status') != 'complete' for v in results):
                    raise ValueError('Cloud job has incomplete recordings. Read the cloud report before retrying.')
                files = report.get('files', [])
                if not files or len({f['path'] for f in files}) != len(files):
                    raise ValueError('Invalid output file list.')
                for item in files:
                    p = PurePosixPath(item['path'])
                    if p.parts[0] not in expected or p.suffix.lower() not in ('.pdf', '.srt', '.md', '.json', '.txt', '.wav'):
                        raise ValueError('Unexpected output type or lesson folder.')
                    if item['path'] not in names or z.getinfo(item['path']).file_size != item['bytes']:
                        raise ValueError('An output is missing or has a different size.')
                for video in results:
                    paths = {f['path'] for f in files if f['path'].startswith(video['id']+'/')}
                    if video['id']+'/SOURCES.md' not in paths:
                        raise ValueError('A recording has no source guide.')
                    task = record['config']['settings']['task']
                    if 'srt' in record['config']['settings'].get('exports', ['srt'] if task!='visual' else []) and video['id']+'/Online Class.srt' not in paths:
                        raise ValueError('A recording has no completed transcript.')
                    if 'pdf' in record['config']['settings'].get('exports', ['pdf'] if task!='speech' else []) and not any(p.endswith('.pdf') for p in paths):
                        raise ValueError('A recording has no visual transcript.')
                for video in results:
                    requested=record['config']['settings'].get('exports',[])
                    for export,name in [('audio','Audio.wav'),('ocr','Visual OCR.txt'),('ocr','Visual OCR.json')]:
                        if export in requested and video['id']+'/'+name not in {f['path'] for f in files}:
                            raise ValueError('Missing requested '+export+' export.')
                # Write in an exclusive staging folder, verify streams, then atomically publish.
                staging = target/('.transvision-import-'+uuid.uuid4().hex)
                staging.mkdir()
                try:
                    for item in files:
                        out = staging/item['path']
                        out.parent.mkdir(parents=True, exist_ok=True)
                        h = hashlib.sha256()
                        with z.open(item['path']) as inp, out.open('xb') as stream:
                            for block in iter(lambda: inp.read(1024*1024), b''):
                                h.update(block)
                                stream.write(block)
                        if h.hexdigest() != item['sha256']:
                            raise ValueError('Output integrity check failed.')
                    (staging/'learning-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                    staging.rename(destination)
                except Exception:
                    # Only this operation's verified private staging path is removed.
                    assert staging.parent == target and staging.name.startswith('.transvision-import-')
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            record.update(status='imported', imported=str(destination))
            job_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            return dict(folder=str(destination), status='imported', recordings=len(results))
