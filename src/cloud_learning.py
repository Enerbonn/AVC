"""Kaggle-only execution: Whisper speech + the existing TransVision visual engine."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue


def stamp(seconds):
    milliseconds = round(seconds*1000)
    h, milliseconds = divmod(milliseconds, 3600000)
    m, milliseconds = divmod(milliseconds, 60000)
    s, milliseconds = divmod(milliseconds, 1000)
    return f'{h:02}:{m:02}:{s:02},{milliseconds:03}'


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def write_srt(path, segments):
    temp = path.with_suffix('.srt.tmp')
    previous = -1
    with temp.open('w', encoding='utf-8', newline='\r\n') as f:
        for i, segment in enumerate(segments, 1):
            start, end, text = segment['start'], segment['end'], segment['text'].strip()
            if not text or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end or start < previous:
                raise ValueError('Invalid subtitle cue.')
            previous = start
            if stamp(start) == stamp(end):
                raise ValueError('Subtitle is shorter than one millisecond.')
            f.write(f'{i}\n{stamp(start)} --> {stamp(end)}\n{text}\n\n')
    temp.replace(path)


def duration(path):
    value = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], text=True).strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Recording has no valid duration.')
    return value


def speech(source, output, cache, model, settings, total):
    import numpy as np
    cache.mkdir(parents=True, exist_ok=True)
    segments = []
    chunk_size, overlap = settings['chunk_seconds'], settings['overlap_seconds']
    signature = dict(settings=settings, name=source.name, bytes=source.stat().st_size, seconds=total)
    for chunk in range(math.ceil(total/chunk_size)):
        from src.cancellation import check_cancel
        check_cancel()
        core_start = chunk*chunk_size
        core_end = min(total, core_start+chunk_size)
        checkpoint = cache/f'{chunk:04}.json'
        cached = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
        if cached and cached.get('signature') == signature:
            cues = cached['segments']
        else:
            begin, end = max(0, core_start-overlap), min(total, core_end+overlap)
            audio = cache/f'{chunk:04}.wav'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                '-ss', str(begin), '-i', str(source), '-t', str(end-begin), '-map', '0:a:0',
                '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-y', str(audio)], check=True)
            with wave.open(str(audio), 'rb') as reader:
                if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1,2,16000):
                    raise ValueError('Expected mono 16 kHz PCM16 audio.')
                waveform = np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').astype(np.float32)/32768.0
            generated, _ = model.transcribe(waveform, language=None if settings['language']=='auto' else settings['language'],
                task='transcribe', beam_size=5, word_timestamps=True, vad_filter=True,
                vad_parameters={'min_silence_duration_ms':1000, 'speech_pad_ms':400},
                condition_on_previous_text=False, hallucination_silence_threshold=2.0)
            cues = []
            for segment in generated:
                words = [w for w in (segment.words or []) if core_start <= begin+(w.start+w.end)/2 < core_end]
                if not words:
                    continue
                text = ''.join(w.word for w in words).strip()
                start, stop = max(core_start, begin+words[0].start), min(core_end, begin+words[-1].end)
                if text and round(stop*1000)>round(start*1000):
                    cues.append(dict(start=start, end=stop, text=text))
            write_json(checkpoint, dict(signature=signature, segments=cues))
            audio.unlink(missing_ok=True)
        segments.extend(cues)
        write_srt(output/'Online Class.partial.srt', segments)
        print(f'{source.name}: speech {core_end/60:.1f}/{total/60:.1f} minutes', flush=True)
    write_srt(output/'Online Class.srt', segments)
    (output/'Online Class.partial.srt').unlink(missing_ok=True)
    return len(segments)


def visuals(source, output, cache, settings):
    from src.batch import BatchQueue
    cache.mkdir(parents=True, exist_ok=True)
    queue = BatchQueue(cache/'queue.json', cache/'visuals')
    if not queue.snapshot()['jobs']:
        entry = {'video': str(source)}
        if settings.get('crop_roi'):
            entry['content_roi']=settings['crop_roi']
        elif settings['crop_mode'] == 'full':
            import cv2
            capture = cv2.VideoCapture(str(source))
            try:
                width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if not capture.isOpened() or width <= 0 or height <= 0:
                    raise ValueError('Cannot read the video dimensions.')
                entry['content_roi'] = [0,0,width,height]
            finally:
                capture.release()
        queue.enqueue([entry], dict(task='both' if 'ocr' in settings.get('exports',[]) and 'pdf' in settings.get('exports', ['pdf']) else 'ocr' if 'ocr' in settings.get('exports',[]) else 'pdf', sensitivity=settings['sensitivity'],
            automatic_export=True, pdf_timestamps=True, slides_only=settings['slides_only'],
            motion_mode=settings['motion_mode']))
    else:
        queue.retry()
    queue.run()
    jobs = queue.snapshot()['jobs']
    if len(jobs) != 1 or jobs[0]['status'] != 'completed':
        raise RuntimeError('Visual extraction failed: '+str(jobs[0].get('message') if jobs else 'no job'))
    import shutil
    paths = jobs[0]['outputs']
    if not paths:
        raise RuntimeError('Visual extraction produced no PDF.')
    result = []
    for i, path in enumerate(paths, 1):
        name = 'Visual OCR.txt' if Path(path).suffix=='.txt' else 'Visual Transcript.pdf' if len([p for p in paths if Path(p).suffix=='.pdf'])==1 else f'Visual Transcript - Part {i:02d}.pdf'
        shutil.copyfile(path, output/name)
        result.append(name)
        if Path(path).suffix=='.txt':
            structured=Path(path).parent/'selected_frames.json'
            if structured.exists():
                entries=json.loads(structured.read_text(encoding='utf-8'))
                # Raw cache paths are not portable; retain evidence and timestamps.
                for entry in entries:
                    for key in list(entry):
                        if 'path' in key or key in ('file','filename'):entry.pop(key,None)
                write_json(output/'Visual OCR.json',entries)
                result.append('Visual OCR.json')
    print(source.name+': visuals complete', flush=True)
    return result


def source_guide(video, output, total, settings, pdfs):
    text = f"# {video['name']}\n\nOriginal recording: `{video['name']}`. Duration: {total/60:.1f} minutes.\n\n"
    text += 'All timestamps refer to the original recording start; no trim offset was applied.\n\n'
    if settings['task'] in ('both','speech'):
        text += f"- [Online Class.srt](Online%20Class.srt): spoken transcript, multilingual Whisper {settings['model']}.\n"
    for name in pdfs:
        text += f'- [{name}]({name.replace(" ", "%20")}): selected visual frames with original timestamps.\n'
    text += '\nUse the spoken transcript for explanations and the visual PDFs for equations, slides and on-screen work. '
    text += 'Match SRT cue times to PDF timestamps before attributing a statement to a frame. '
    text += 'Add instructor notes for formal definitions and practice reviewers for study priorities.\n\n'
    text += 'These automated exports can miss speech or visuals and can misrecognize Taglish, names and equations. '
    text += 'They supplement the recording and instructor notes; they do not guarantee complete coverage.\n\n'
    text += 'To study with review(er), provide this folder and your learning goal. The skill should read this guide first.\n'
    (output/'SOURCES.md').write_text(text, encoding='utf-8')


def run(config):
    # Guard before model imports or expensive work. Never silently transcribe on the laptop.
    if not Path('/kaggle/working').is_dir() or not Path('/kaggle/input').is_dir():
        raise RuntimeError('Cloud learning runs only on Kaggle.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required. Enable GPU T4 x2.')
    settings = config['settings']
    output = Path('/kaggle/working/learning-results')
    cache = Path('/kaggle/temp/transvision-learning')/config['job_id']
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    totals = {}
    for video in config['videos']:
        matches = [p for p in Path('/kaggle/input').rglob('*') if p.is_file() and p.name==video['name']]
        if len(matches) != 1 or matches[0].stat().st_size != video['bytes']:
            raise RuntimeError('Attach exactly the selected original recording: '+video['name'])
        files[video['id']] = matches[0]
        totals[video['id']] = duration(matches[0])
    count = min(2, torch.cuda.device_count())
    models = []
    if settings['task'] != 'visual':
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        model_path = download_model(settings['model'], cache_dir=str(cache/'models'))
        models = [WhisperModel(model_path, device='cuda', device_index=i, compute_type='float16', cpu_threads=2)
                  for i in range(count)]
    pending = Queue()
    for video in sorted(config['videos'], key=lambda v:totals[v['id']], reverse=True):
        pending.put(video)
    def worker(gpu):
        results = []
        while True:
            try:
                video = pending.get_nowait()
            except Empty:
                return results
            dest = output/video['id']
            dest.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                cues, pdfs = 0, []
                if settings['task'] != 'visual':
                    cues = speech(files[video['id']], dest, cache/video['id']/'speech', models[gpu], settings, totals[video['id']])
                if settings['task'] != 'speech':
                    pdfs = visuals(files[video['id']], dest, cache/video['id']/'visual', settings)
                source_guide(video, dest, totals[video['id']], settings, pdfs)
                results.append(dict(id=video['id'], status='complete', cues=cues, pdfs=len(pdfs),
                    seconds=totals[video['id']], processing_seconds=time.monotonic()-started))
            except Exception:
                error = traceback.format_exc()
                print('FAILED '+video['name']+'\n'+error, flush=True)
                results.append(dict(id=video['id'], status='failed', error=error))
            finally:
                pending.task_done()
    with ThreadPoolExecutor(max_workers=count) as pool:
        results = [r for group in pool.map(worker, range(count)) for r in group]
    generated = []
    for video in config['videos']:
        for path in sorted((output/video['id']).rglob('*')):
            if path.is_file():
                h = hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024), b''):
                        h.update(block)
                generated.append(dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size, sha256=h.hexdigest()))
    report = dict(schema=config['schema'], job_id=config['job_id'], settings=settings, results=results, files=generated)
    write_json(output/'learning-report.json', report)
    archive = Path('/kaggle/working/learning-results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(output/'learning-report.json', 'learning-report.json')
        for item in generated:
            z.write(output/item['path'], item['path'])
    print('DOWNLOAD: '+str(archive), flush=True)
    if any(r['status']!='complete' for r in results):
        raise RuntimeError('Some recordings failed. Inspect learning-report.json; successful outputs were preserved.')
    print('ALL RECORDINGS COMPLETE', flush=True)
