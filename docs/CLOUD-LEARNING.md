# AVC cloud notebooks

Select Cloud notebook, choose Kaggle or Colab, select your exports and prepare.
The saved job folder includes INSTRUCTIONS.md and transvision-learning.ipynb.
The notebook embeds a snapshot of the processing engine and its licenses, plus
input filenames/sizes and requested settings. Local paths and credentials are excluded.

## Kaggle

1. Sign into your account. Create a private dataset with the selected originals.
2. Import the notebook, keep it private, and attach that dataset.
3. Enable Internet. Select an available GPU to accelerate speech; visual-only
   processing and CPU speech are also supported. Choose Save & Run All.
4. Download learning-results.zip from the saved run's Output tab and import it into
   the matching prepared AVC job.

## Colab

1. Sign into Colab and import the notebook.
2. Create /content/avc-input using its Files sidebar and upload the originals there.
   For larger files, copy them from your mounted Drive into that directory.
3. Select an available GPU runtime for speech, then Run all. CPU also works.
4. Keep the session connected. Download /content/learning-results.zip before the
   runtime expires, then import it into the matching AVC job.

There is no automatic upload, account switching, quota workaround, API submission,
or cloud status polling. Free resources are not guaranteed; respect each platform's
limits. Uploads and visual processing may dominate elapsed time. Failed runs can
leave partial files in the output directory; a complete importable ZIP is generated
only on success. Cross-session checkpoint restore is not implemented.

Use Visual trial only for a short visual check before running long recordings.
It skips speech and audio exports. The ordinary run button applies the visual
trial limit only to visuals. All timestamps retain the original start point.

Sources: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
[Colab FAQ](https://research.google.com/colaboratory/faq.html),
[faster-whisper CPU/CUDA options](https://github.com/SYSTRAN/faster-whisper).
# GPU runtime repair (2026-11-09)

If an older notebook fails with `libcublas.so.12` missing, replace it with a newly
generated notebook and run both code cells. New speech notebooks install CUDA 12
cuBLAS and cuDNN 9 and start inference in a fresh process with their library paths
set before Python starts. A preflight checks library loading before downloading
Whisper. Keep the same input dataset attached and Internet enabled.

The Hugging Face unauthenticated-download warning is informational; it was not
the cause of this failure. GPU initialization repair still needs a live Kaggle
run to verify on that provider's current container.

Reference: [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu).
