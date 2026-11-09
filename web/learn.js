'use strict';
const $ = id => document.getElementById(id);
const outputNames = {srt: 'Speech transcript (.srt)', pdf: 'Visual notes (.pdf)', ocr: 'Screen text (.txt + .json)', audio: 'Separate audio (.wav)'};
async function api(route, data) {
  const response = await fetch(route, {method: data ? 'POST' : 'GET', headers: {'Content-Type': 'application/json', 'X-TransVision-Token': window.TRANSVISION_TOKEN}, body: data ? JSON.stringify(data) : undefined});
  const value = await response.json();
  if (!response.ok) throw Error(value.error || 'Request failed');
  return value;
}
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
async function perform(button, work) {
  button.disabled = true;
  try { await work(); } catch (error) { message(error.message, true); $('message').scrollIntoView({block: 'nearest'}); }
  finally { button.disabled = false; }
}
function notebookLink() {
  const link = $('currentNotebook'), id = $('job').value;
  link.hidden = !id;
  if (id) link.href = `/api/cloud/notebook?job_id=${encodeURIComponent(id)}&token=${encodeURIComponent(window.TRANSVISION_TOKEN)}`;
  else link.removeAttribute('href');
}
$('job').onchange = notebookLink;
async function jobs(selectId) {
  const rows = await api('/api/cloud/jobs'), select = $('job'); select.replaceChildren();
  for (const row of rows) select.append(new Option(`${row.videos.map(v => v.name).join(', ')} · ${row.status}`, row.job_id));
  if (selectId) select.value = selectId;
  if (!rows.length) select.append(new Option('Prepare a notebook first', ''));
  notebookLink();
}
function selectedOutputs() { return Object.keys(outputNames).filter(id => $(id).checked); }
function recordings() { return $('videos').value.split(/\r?\n/).map(v => v.trim().replace(/^"|"$/g, '')).filter(Boolean); }
function durationLabel(seconds) { return seconds % 60 === 0 ? `${seconds / 60} minute${seconds === 60 ? '' : 's'}` : `${seconds} seconds`; }
function sampleSeconds() {
  const seconds = Number($('preview').value);
  if (!Number.isInteger(seconds) || seconds < 1 || seconds > 86400) { $('preview').focus(); throw Error('Enter a sample length from 1 to 86400 seconds. For example, 180 is 3 minutes.'); }
  return seconds;
}
function requestData(testVisuals = false) {
  const exports = testVisuals ? ($('ocr').checked ? ['pdf', 'ocr'] : ['pdf']) : selectedOutputs();
  const visuals = exports.some(id => ['pdf', 'ocr'].includes(id));
  const paths = recordings();
  if (!paths.length) { $('videos').focus(); throw Error('Choose at least one video or audio file first.'); }
  if (!exports.length) { $('srt').focus(); throw Error('Choose at least one output: speech, visual notes, screen text or audio.'); }
  const destination = $('destination').value.trim().replace(/^"|"$/g, '');
  if ($('mode').value === 'local' && !destination) { $('destination').focus(); throw Error('Choose a folder to save the local results in.'); }
  return {use_editor_crop: visuals && $('editorCrop').checked, videos: paths, destination, settings: {
    exports, task: visuals ? (exports.includes('srt') ? 'both' : 'visual') : 'speech',
    model: $('model').value, language: $('language').value, crop_mode: $('crop').value,
    sensitivity: Number($('sensitivity').value), slides_only: visuals && $('slides').checked,
    device: $('device').value, provider: $('provider').value,
    preview_seconds: visuals && (testVisuals || $('visualRange').value === 'sample') ? sampleSeconds() : 0
  }};
}
function planRow(label, value) {
  const row = document.createElement('div'), term = document.createElement('dt'), description = document.createElement('dd');
  term.textContent = label; description.textContent = value; row.append(term, description); $('runPlan').append(row);
}
function updateConfiguration() {
  const cloud = $('mode').value === 'cloud', outputs = selectedOutputs();
  const visuals = outputs.some(id => ['pdf', 'ocr'].includes(id)), speech = outputs.includes('srt');
  const seconds = Number($('preview').value), validSeconds = Number.isInteger(seconds) && seconds >= 1 && seconds <= 86400;
  const level = Number($('sensitivity').value), detail = level < 34 ? 'fewer checkpoints' : level > 66 ? 'more writing steps' : 'balanced';
  $('detail').textContent = `${level} / 100 · ${detail}`; $('sensitivity').setAttribute('aria-valuetext', `${level} out of 100, ${detail}`);
  $('prepare').textContent = cloud ? 'Create cloud notebook' : 'Start local processing';
  $('providerField').hidden = !cloud; $('provider').disabled = !cloud;
  $('cloudResults').hidden = !cloud; $('cloudGuide').hidden = !cloud;
  $('speechSettings').hidden = !speech; $('visualSettings').hidden = !visuals;
  $('modeHelp').textContent = cloud ? 'Your recording is processed on Kaggle or Colab. AVC prepares a notebook here; you upload files and start it there.' : 'Processing uses this computer. First use may download recognition models; your recordings are not uploaded.';
  $('trial').textContent = `Test visuals only · first ${validSeconds ? durationLabel(seconds) : 'sample'}`;
  $('trialHelp').textContent = `${cloud ? 'Creates a cloud notebook for' : 'Exports'} a visual PDF${$('ocr').checked ? ' and screen text' : ''} only. No speech transcript or WAV. Check the result, then use Whole recording for the complete visual run.`;
  const colab = $('provider').value === 'colab';
  $('uploadStep').textContent = colab ? 'In Colab, create /content/avc-input using the Files sidebar. Upload the original recordings there, or copy them from your mounted Drive.' : 'Upload the original recordings to a private Kaggle dataset, then attach it to the notebook.';
  $('runStep').textContent = colab ? 'Choose an available GPU runtime for speech, then Run all. Keep the session connected.' : 'Enable Internet, choose an available GPU for speech, then Save & Run All.';
  $('runPlan').replaceChildren();
  planRow('Processing', cloud ? `Cloud notebook for ${colab ? 'Google Colab' : 'Kaggle'}` : 'This computer');
  planRow('Recordings', pathsLabel(recordings().length));
  planRow('Output files', outputs.length ? outputs.map(id => outputNames[id]).join('; ') : 'Choose at least one output');
  if (speech) planRow('Speech coverage', `Whole recording · ${$('model').selectedOptions[0].textContent.split(' — ')[0]}`);
  if (outputs.includes('audio')) planRow('Audio coverage', 'Whole recording');
  if (visuals) {
    planRow('Visual coverage', $('visualRange').value === 'full' ? 'Whole recording' : validSeconds ? `First ${durationLabel(seconds)}` : 'Enter a valid sample length');
    planRow('Screen area', $('editorCrop').checked ? 'Saved editor crop (same single video)' : $('crop').value === 'full' ? 'Whole frame' : 'Automatic light-slide crop');
  }
  $('configurationHint').textContent = cloud ? 'Creating a notebook does not upload files or start a cloud run.' : speech && ['large-v3','medium'].includes($('model').value) && $('device').value !== 'cuda' ? 'Large speech models can be slow on CPU. Consider a cloud GPU for long recordings.' : 'Results are saved in a new folder; source recordings stay where they are.';
}
function pathsLabel(count) { return count ? `${count} selected` : 'None selected yet'; }
for (const id of ['mode','provider','model','language','device','crop','sensitivity','srt','pdf','ocr','audio','visualRange','preview','editorCrop','slides','videos']) {
  $(id).addEventListener('input', updateConfiguration); $(id).addEventListener('change', updateConfiguration);
}
$('browse').onclick = () => perform($('browse'), async () => { const result = await api('/api/cloud/browse', {kind:'videos'}); if (result.paths.length) $('videos').value = result.paths.join('\n'); updateConfiguration(); });
for (const [id,kind,target] of [['pickZip','archive','archive'],['pickFolder','destination','destination']]) $(id).onclick = () => perform($(id), async () => { const result = await api('/api/cloud/browse', {kind}); if (result.path) $(target).value = result.path; });
async function submit(data, testVisuals = false) {
  if ($('mode').value === 'local') {
    await api('/api/local/start', data);
    message(testVisuals ? `Visual test started: first ${durationLabel(data.settings.preview_seconds)}. Speech and WAV extraction are skipped.` : 'Local processing started. Speech models may download on first use.');
    await localStatus(); return;
  }
  message('Creating cloud notebook…');
  const result = await api('/api/cloud/prepare', data), panel = $('prepared'); panel.hidden = false; panel.replaceChildren();
  const link = document.createElement('a'); link.className = 'download'; link.textContent = testVisuals ? 'Download visual-test notebook' : 'Download cloud notebook';
  link.href = `/api/cloud/notebook?job_id=${result.job_id}&token=${encodeURIComponent(window.TRANSVISION_TOKEN)}`; panel.append(link);
  const text = document.createElement('p'); text.textContent = `Saved in ${result.folder}. Follow the cloud steps below to upload and run.`; panel.append(text);
  await jobs(result.job_id); message('Notebook ready. Download it, upload your recordings to the selected platform and run both code cells.');
}
$('prepare').onclick = () => perform($('prepare'), () => submit(requestData()));
$('trial').onclick = () => perform($('trial'), () => submit(requestData(true), true));
async function localStatus() {
  const result = await api('/api/local/status');
  const text = result.status === 'idle' ? 'No local job running. Choose files to begin.' : [result.status,result.message,result.folder].filter(Boolean).join(' · ');
  if ($('localStatus').textContent !== text) $('localStatus').textContent = text;
  $('stopLocal').disabled = !['running','stopping'].includes(result.status);
}
$('stopLocal').onclick = () => perform($('stopLocal'), async () => { await api('/api/local/stop', {}); await localStatus(); });
$('import').onclick = () => perform($('import'), async () => {
  message('Checking results and importing…');
  const result = await api('/api/cloud/import', {job_id:$('job').value, archive:$('archive').value.trim().replace(/^"|"$/g,''), destination:$('destination').value.trim().replace(/^"|"$/g,'')});
  $('imported').hidden = false; $('imported').textContent = `Your learning resources: ${result.folder}`; await jobs(); message(`${result.recordings} recording(s) imported. Give the folder and your learning goal to your AI study tool.`);
});
updateConfiguration(); localStatus().catch(() => {}); setInterval(() => localStatus().catch(() => {}), 2000);
jobs().catch(error => message(error.message, true));
const windowClient = crypto.randomUUID();
function presence(closed = false) { fetch('/api/window', {method:'POST', keepalive:true, headers:{'Content-Type':'application/json','X-TransVision-Token':window.TRANSVISION_TOKEN}, body:JSON.stringify({client:windowClient,closed})}).catch(() => {}); }
presence(); setInterval(() => presence(), 15000); window.addEventListener('pagehide', () => presence(true)); window.addEventListener('pageshow', () => presence());
