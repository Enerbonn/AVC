# AVC - Audio Visual Context

Convert video or audio into context an AI can follow: timed speech, visual frames,
OCR and source guides. No summarization step is applied. AVC remains separate
from TransVision Beta.

## Use

Run **Start AVC.cmd**. For a fresh installation, run **Setup AVC.cmd** first.
Install FFmpeg (including ffprobe) and make it available on PATH.

If startup reports broken dependencies, run **Setup AVC.cmd** to repair them.
Setup builds a fresh environment rather than mixing compiled packages from different
Python versions. It restores the previous environment if installation fails and
preserves the previous environment on success. Python 3.12 is recommended.

1. Choose video or audio recordings and **Local** or **Cloud notebook**.
2. Select SRT, visual PDF, OCR text/JSON, extracted WAV audio, or a combination.
   Audio-only recordings accept SRT/audio exports.
3. For local processing, choose an existing output folder and start. Auto selects
   NVIDIA CUDA for Whisper when available; otherwise it uses CPU int8. CUDA needs
   compatible NVIDIA drivers/CUDA runtime libraries. Choose CPU explicitly if needed.
4. For cloud processing, choose Kaggle or Colab and prepare the notebook. The
   generated folder contains the notebook and INSTRUCTIONS.md. Upload and start
   the run yourself, download learning-results.zip, then import it here.

Whisper and OCR may download models on first use. After models are cached, local
processing does not require uploading your recordings. Tiny/Base are useful for
CPU trials; Large V3 is much heavier. Visual detection uses CPU/OpenCV; CUDA
acceleration currently applies to Whisper, not the entire pipeline.

## Check visuals before a full run

Set **Sample length in seconds** to 180 for the first three minutes, or choose another
number of seconds. Click **Test visuals only**: this exports visual PDF (and OCR
if selected) without launching full speech transcription or WAV extraction.
Inspect the PDF, then choose **Whole recording** under **Visual coverage for the
main run** for the full visual run. The test button always uses the sample length;
the main button uses the selected coverage. Testing does not change that selection.

Use **Visual editor** to load the same single video and adjust its crop. Back on
AVC's main page, enable **Use the crop from the visual editor**. Both local jobs
and prepared cloud notebooks retain that crop. For multiple recordings, use
full-frame capture or automatic crop instead.

The ordinary Start button respects every selected export. A visual time limit
only limits visuals; selected SRT and WAV exports still cover the full recording.
All exports start at the original recording's time zero. Custom start offsets
are not implemented.

The interface explains each export and setting beside its control. **Whole frame**
keeps the full picture; automatic crop is intended for light slides. The visual
change slider controls how many writing steps become checkpoints, not image
resolution or OCR accuracy. Start at 50, then inspect a sample. Speech settings
appear only when the transcript is selected; visual settings appear only for PDF
or OCR. **What will happen** shows outputs, execution location and coverage before
you start. Cloud instructions update for the selected provider.

## Results and limits

Each recording has its own folder and SOURCES.md. OCR JSON retains timestamps,
text, confidence and regions. Pair SRT cue times with PDF timestamps. Recognition
and frame selection can miss or misread material, especially equations and Taglish.
No output certifies complete coverage. Existing media and course files are preserved.

Stopping is cooperative at safe boundaries; an active model call, FFmpeg command
or visual extraction may take time to return. Local status is for the current app
session. Interrupted outputs are retained; restarting creates a fresh destination.

Cloud availability and free quotas vary. Account login, uploading and launching
remain manual; direct API submission and live cloud polling are not implemented.
For a failed or older cloud job, switch to Cloud notebook, select the prepared
job, and choose **Download updated notebook for this job**. Downloads always
include the currently installed AVC engine and CUDA setup, while retaining the
job identity so its results can still be imported. Replace the notebook on your
cloud provider and run both code cells; keep its dataset attached.
Generated notebooks support Kaggle and Colab, but a live combined run on both
providers remains to be verified. See [cloud guide](docs/CLOUD-LEARNING.md).

## Credits

Visual engine derived from [TransVision](https://github.com/Enerbonn/TransVision),
based on FrameRead. Speech uses faster-whisper and multilingual Whisper models.
[LICENSE](LICENSE), [NOTICE](NOTICE), and model notices remain applicable.
