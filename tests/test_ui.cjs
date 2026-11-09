// Test the actual browser controller with a minimal DOM, without starting media jobs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/learn.html'), 'utf8');
const elements = {};
function element() { return {value:'', checked:false, hidden:false, disabled:false, textContent:'', selectedOptions:[{textContent:'Large V3 — quality default'}], children:[], classList:{toggle(){}}, addEventListener(){}, setAttribute(){}, removeAttribute(){}, focus(){}, scrollIntoView(){}, append(...items){this.children.push(...items);}, replaceChildren(...items){this.children=items;}}; }
for (const match of html.matchAll(/\bid="([^"]+)"/g)) elements[match[1]] = element();
for (const match of html.matchAll(/<input\b([^>]+)>/g)) {
  const id = match[1].match(/\bid="([^"]+)"/)?.[1]; if (!id) continue;
  elements[id].checked = /\bchecked\b/.test(match[1]); elements[id].value = match[1].match(/\bvalue="([^"]*)"/)?.[1] || '';
}
Object.assign(elements.mode, {value:'local'}); Object.assign(elements.provider, {value:'kaggle'});
for (const [id,value] of Object.entries({model:'large-v3',language:'tl',device:'auto',crop:'full',visualRange:'full'})) elements[id].value=value;
const context = vm.createContext({document:{getElementById:id=>elements[id], createElement:element}, window:{TRANSVISION_TOKEN:'test', addEventListener(){}}, crypto:{randomUUID:()=> 'test'}, Option:function(text,value){return {textContent:text,value};}, fetch:async route=>({ok:true,json:async()=>route.endsWith('/jobs')?[]:{status:'idle'}}), setInterval(){}, console});
vm.runInContext(fs.readFileSync(path.join(root,'web/learn.js'),'utf8'),context);
const evaluate = code => vm.runInContext(code,context);
elements.videos.value = 'example.mp4'; elements.destination.value='results';
let data = JSON.parse(evaluate('JSON.stringify(requestData())'));
assert.equal(data.settings.preview_seconds,0,'Defaults cover the whole recording despite sample length 180');
assert.deepEqual(data.settings.exports,['srt','pdf']);
data = JSON.parse(evaluate('JSON.stringify(requestData(true))'));
assert.equal(data.settings.preview_seconds,180); assert.deepEqual(data.settings.exports,['pdf']);
elements.ocr.checked=true;
assert.deepEqual(JSON.parse(evaluate('JSON.stringify(requestData(true).settings.exports)')),['pdf','ocr']);
elements.visualRange.value='sample'; elements.preview.value='60';
data = JSON.parse(evaluate('JSON.stringify(requestData())'));
assert.equal(data.settings.preview_seconds,60); assert.ok(data.settings.exports.includes('srt'));
elements.preview.value=''; assert.throws(()=>evaluate('requestData(true)'),/sample length/);
elements.preview.value='180'; elements.srt.checked=false; elements.pdf.checked=false; elements.ocr.checked=false;
assert.throws(()=>evaluate('requestData()'),/at least one output/);
elements.audio.checked=true; elements.editorCrop.checked=true;
data = JSON.parse(evaluate('JSON.stringify(requestData())'));
assert.equal(data.use_editor_crop,false); assert.equal(data.settings.preview_seconds,0);
elements.mode.value='cloud'; elements.provider.value='colab'; evaluate('updateConfiguration()');
assert.equal(elements.providerField.hidden,false); assert.equal(elements.speechSettings.hidden,true); assert.equal(elements.visualSettings.hidden,true);
assert.match(elements.uploadStep.textContent,/\/content\/avc-input/);
elements.srt.checked=true; elements.device.value='cpu';
assert.equal(JSON.parse(evaluate('JSON.stringify(requestData())')).settings.device,'cpu');
elements.mode.value='local'; elements.destination.value=''; assert.throws(()=>evaluate('requestData()'),/folder/);
console.log('UI controller checks passed: defaults, visual-only test, custom range, output visibility, CPU choice, validation and provider guidance.');
