"""Launch cloud inference with NVIDIA libraries visible before Python starts."""
import ctypes
import json
import os
from pathlib import Path
import site
import subprocess
import sys


def library_path(roots, existing=""):
    libraries = []
    for root in roots:
        libraries.extend(str(p) for p in sorted((Path(root) / "nvidia").glob("*/lib")) if p.is_dir())
    return os.pathsep.join(dict.fromkeys(libraries + ([existing] if existing else [])))


def launch(config, engine):
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = library_path(site.getsitepackages() + [site.getusersitepackages()],
                                         env.get('LD_LIBRARY_PATH', ''))
    # Linux's dynamic loader reads this path at process start, not after a
    # notebook kernel has already imported inference libraries.
    subprocess.run([sys.executable, '-u', '-m', 'src.cloud_runtime', json.dumps(config)],
                   cwd=str(engine), env=env, check=True)


def preflight(settings):
    if 'srt' not in settings['exports'] or settings['device'] == 'cpu':
        return
    import ctranslate2
    if ctranslate2.get_cuda_device_count() == 0:
        print('No CUDA GPU detected; Whisper will use CPU.', flush=True)
        return
    for name in ('libcublasLt.so.12', 'libcublas.so.12', 'libcudnn.so.9'):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError('AVC GPU runtime check failed for ' + name +
                               '. Run both cells of the updated notebook with Internet enabled. '
                               'CUDA 12 cuBLAS and cuDNN 9 are required.') from exc
    print('AVC GPU runtime check passed: CUDA 12 cuBLAS and cuDNN 9.', flush=True)


if __name__ == '__main__':
    config = json.loads(sys.argv[1])
    preflight(config['settings'])
    from src.portable_cloud import run
    run(config)
