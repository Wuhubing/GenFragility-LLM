// Run: node website/tests/audit.cjs [output-directory]
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const root = path.resolve(__dirname, '..');
const out = process.argv[2];
const elements = new Map();
function element(key) {
  if (!elements.has(key)) elements.set(key, {innerHTML:'', disabled:false, classList:{add(){}}, addEventListener(){}, querySelectorAll(){return [];}});
  return elements.get(key);
}
const payload = JSON.parse(fs.readFileSync(path.join(root,'data/entities.json')));
let exportedBlob;
const context = vm.createContext({console, setTimeout, performance, URL:{createObjectURL(blob){exportedBlob=blob;return 'blob:test';},revokeObjectURL(){}}, Blob,
  document:{addEventListener(){}, querySelector:element,body:{appendChild(){}},createElement(){return {click(){},remove(){}};}},
  fetch:async()=>({ok:true,json:async()=>payload})});
const appSource=process.env.AUDIT_BASELINE
 ? require('node:child_process').execFileSync('git',['show','21bf8a3:website/app.js'],{cwd:root,encoding:'utf8'})
 : fs.readFileSync(path.join(root,'app.js'),'utf8');
vm.runInContext(appSource, context);
const seeds = [
 ['geography','The capital of France is Paris.',['France','Paris']],
 ['geography','London and Berlin are cities.',['London','Berlin']],
 ['geography','I visited New York City.',['New York City']],
 ['geography','I live in the United States.',['United States']],
 ['company','Apple is a big company.',['Apple Inc.']],
 ['company','Apple Inc. is headquartered in Cupertino.',['Apple Inc.','Cupertino']],
 ['company','Google and Microsoft are companies.',['Google','Microsoft']],
 ['fruit','An apple is a fruit.',['Apple']],
 ['fruit','I ate an apple.',['Apple']],
 ['fruit','Apple is a fruit, and Microsoft is a company.',['Apple','Microsoft']],
 ['entity','Apple Inc.',['Apple Inc.']],
 ['entity','Paris',['Paris']],
 ['entity','United States',['United States']],
 ['punctuation','France—Paris',['France','Paris']],
 ['punctuation','Paris, Paris, Paris!',['Paris']],
 ['negative','We may be reading this today.',[]],
 ['negative','This is a good idea.',[]],
 ['negative','Hello world!',[]],
 ['negative','zzzxqv987654',[]],
 ['negative','1234567890',[]],
 ['negative','',[]],
 ['negative','   ',[]],
 ['negative','<script>alert(1)</script>',[]],
];
const variants = [s=>s,s=>s.toLowerCase(),s=>s.toUpperCase(),s=>[...s].map((c,i)=>i%2?c.toUpperCase():c.toLowerCase()).join('')];
const cases = seeds.flatMap(([category,text,expected],i)=>variants.map((v,k)=>({id:`case-${i+1}-${k}`,category,text:v(text),expected})));
const sorted = a=>[...new Set(a)].sort();
async function main(){
 await vm.runInContext('loadIndex()',context);
 const start=performance.now();
 const results=cases.map(c=>{
  context.input=c.text;
  const actual=vm.runInContext('findEntities(input, false).matches.map(r=>r[0])',context);
  return {...c,actual:[...actual],pass:JSON.stringify(sorted(actual))===JSON.stringify(sorted(c.expected))};
 });
 const checks=[];
 const check=(name,pass,detail='')=>checks.push({name,pass,detail});
 async function upload(name,text){context.file={name,size:Buffer.byteLength(text),text:async()=>text};await vm.runInContext('analyzeDataset(file)',context);return element('#dataset-results').innerHTML;}
 const texts=['The capital of France is Paris.','apple is a big company.','Apple Inc.','zzzxqv987654'];
 const formats={
  'sample.txt':texts.join('\n'),
  'sample.csv':'text\n'+texts.map(t=>'"'+t.replaceAll('"','""')+'"').join('\n'),
  'sample.json':JSON.stringify(texts),
  'sample-records.json':JSON.stringify(texts.map((text,i)=>({id:i,text,source:'London'})))
 };
 for(const [name,text] of Object.entries(formats)){
  const html=await upload(name,text);
  const counts=vm.runInContext('state.lastDatasetResults.map(x=>[x.record[0],x.count])',context);
  check(name, JSON.stringify([...counts].sort())===JSON.stringify([['Apple Inc.',2],['France',1],['Paris',1]].sort()),JSON.stringify(counts));
 }
 let html=await upload('bad.json','{invalid');check('invalid JSON rejected',html.includes('Could not read'));
 html=await upload('bad.csv','text\n"unclosed');check('invalid CSV rejected',html.includes('Could not read'));
 html=await upload('bad.exe','Paris');check('unsupported type rejected',html.includes('Could not read'));
 context.file={name:'big.txt',size:5242881,text:async()=>{throw Error('must not read');}};
 await vm.runInContext('analyzeDataset(file)',context);check('5 MB limit',element('#dataset-results').innerHTML.includes('larger than 5 MB'));
 html=await upload('large.txt',Array(5001).fill('Paris').join('\n'));check('row limit disclosed',/truncat|first 5,000|first 5000/i.test(html));
 check('HTML escaping',vm.runInContext('escapeHtml("<img src=x onerror=alert(1)>")',context).startsWith('&lt;'));
 const names=new Map(payload.entities.map(r=>[r[0],r]));
 check('known degrees',names.get('Apple')[2]===4&&names.get('Apple Inc.')[2]===467&&names.get('United States')[2]===17027);
 check('index count',payload.entities.length===payload.meta.entity_count);
 check('degree sum',payload.entities.reduce((s,r)=>s+r[2],0)===payload.meta.forward_edge_count);
 html=await upload('bom.csv','\uFEFFtext,source\r\n"Paris",London\r\n"France",Berlin');
 check('CSV BOM and metadata exclusion',!html.includes('entity-name">London') && html.includes('entity-name">Paris'));
 html=await upload('multiline.csv','text\n"France\nand Paris"');
 check('quoted multiline CSV',html.includes('entity-name">France')&&html.includes('entity-name">Paris'));
 html=await upload('empty.txt','\n   \n');check('empty dataset',html.includes('0</strong>'));
 html=await upload('long.txt','x'.repeat(20001));check('long row rejected',html.includes('Could not read'));
 context.input='x'.repeat(20001);vm.runInContext('renderSentenceResult(input)',context);
 check('long sentence rejected',element('#sentence-results').innerHTML.includes('20,000'));
 await upload('export.txt','Apple Inc.\nParis\nApple Inc.');
 vm.runInContext('downloadDatasetResults()',context);
 const exportText=await exportedBlob.text();
 check('CSV export values',exportText.includes('"Apple Inc.","Q312","467","Super hub","2"')&&exportText.includes('"Paris","Q90","1031"'));
 await upload('invalid.json','{bad');
 check('failed upload clears previous export',vm.runInContext('state.lastDatasetResults.length',context)===0);
 const summary={cases:results.length,passed:results.filter(r=>r.pass).length,checks:checks.length,checksPassed:checks.filter(r=>r.pass).length,elapsedMs:Math.round(performance.now()-start)};
 // Separate evaluation set: do not fold these into the bug-fix regression score.
 const challengeSeeds=[
  ['I work at Apple and eat an apple.',['Apple Inc.','Apple']],
  ['Apple sells phones.',['Apple Inc.']],
  ['Amazon sells books.',['Amazon.com']],
  ['Python is a programming language.',['Python (programming language)']],
  ['Java is a programming language.',['Java (programming language)']],
  ['NASA studies space.',['Nasa']],
  ['IBM makes computers.',['Ibm']],
  ['苹果公司位于美国。',['Apple Inc.','United States']],
  ['I visited Pariss.',['Paris']],
  ['May is a month.',['May']],
  ['Reading is a town.',['Reading']],
  ['I enjoy reading books.',[]],
  ['London is in the United Kingdom.',['London','United Kingdom']],
  ['Berlin is in Germany.',['Berlin','Germany']],
  ['Paris and France are mentioned here.',['Paris','France']],
  ['Microsoft makes software.',['Microsoft']],
  ['Apple makes the iPhone.',['Apple Inc.','iPhone']],
  ['I live in Cupertino.',['Cupertino']],
  ['An apple grows in an orchard.',['Apple']],
  ['No named entities are present.',[]]
 ];
 const challenge=challengeSeeds.flatMap(([text,expected],i)=>variants.map((v,k)=>{
   const input=v(text); context.input=input;
   const actual=[...vm.runInContext('findEntities(input,false).matches.map(r=>r[0])',context)];
   return {id:`challenge-${i+1}-${k}`,text:input,expected,actual,pass:JSON.stringify(sorted(expected))===JSON.stringify(sorted(actual))};
 }));
 const broad=payload.entities.filter((_,i)=>i%200===0).map(record=>{
   const outputs=variants.map(v=>{context.input=`Mention: ${v(record[0])}.`;return [...vm.runInContext('findEntities(input,false).matches.map(r=>r[0])',context)];});
   return {entity:record[0],expected:record[0],outputs,caseConsistent:outputs.every(x=>JSON.stringify(x)===JSON.stringify(outputs[0])),targetRetrieved:outputs.every(x=>x.includes(record[0]))};
 });
 summary.challenge={total:challenge.length,passed:challenge.filter(x=>x.pass).length};
 summary.broad={entities:broad.length,queries:broad.length*4,caseConsistent:broad.filter(x=>x.caseConsistent).length,targetRetrieved:broad.filter(x=>x.targetRetrieved).length};
 console.log(JSON.stringify({summary,failed:results.filter(r=>!r.pass),checks},null,2));
 if(out){fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({summary,results,checks,challenge,broad},null,2));
  fs.writeFileSync(path.join(out,'expected.json'),JSON.stringify(cases,null,2));
  for(const [name,text] of Object.entries(formats)) fs.writeFileSync(path.join(out,name),text);
  fs.writeFileSync(path.join(out,'case-matrix.txt'),cases.map(c=>c.text).join('\n'));
  const csv=rows=>rows.map(row=>row.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\n');
  fs.writeFileSync(path.join(out,'review.csv'),csv([['id','text','expected','actual','pass'],...[...results,...challenge].map(c=>[c.id,c.text,c.expected.join(' | '),c.actual.join(' | '),c.pass])]));
  fs.writeFileSync(path.join(out,'challenge.csv'),csv([['text'],...challenge.map(c=>[c.text])]));
  fs.writeFileSync(path.join(out,'case-matrix.csv'),csv([['text'],...cases.map(c=>[c.text])]));
  fs.writeFileSync(path.join(out,'case-matrix.json'),JSON.stringify(cases.map(c=>({text:c.text})),null,2));
 }
 if (!process.env.AUDIT_BASELINE && (summary.passed !== summary.cases || summary.checksPassed !== summary.checks)) process.exitCode=1;
}
main().catch(e=>{console.error(e);process.exitCode=1;});
