#!/usr/bin/env python3
"""Authorized V5.4 protocol-dev evaluation; no heldout path exists here."""
from __future__ import annotations
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
from typing import Any
P=Path('/home/cyh/Medical_Qwen');PY=Path('/home/cyh/miniconda3/envs/tcm_llm/bin/python')
AUTH=P/'artifacts/v5_4_pipeline/review/root_data_v2_acceptance.json';AUTH_SHA='8d7217e3394ee4b94cd3739d6ce2463bd81723791d061dcda4bb660781bf3c93'
DEV=P/'artifacts/v5_4_pipeline/data_v2/protocol_dev_v5_4_v2.jsonl';DEV_SHA='a48759d4f4131a2b7263f6d99aad1c8c7208fb837c6adf98b390dec5d46c1ed4'
BASE=P/'models/Qwen2.5-1.5B-Instruct';PEFT=P/'output/tcm-qwen-1.5b-v5-4-candidate-a';RUNTIME=P/'tcm_chat_v5.py';RUNTIME_SHA='ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe'
H=P/'v5_4_pipeline/runtime/candidate_h_adapter.py';H_SHA='25a41395354779f88e200a2d09edb45da56f4c9c3e6e601d69d0873c34f2d791';OUT=P/'artifacts/v5_4_pipeline/training/candidate_a/protocol_dev_eval';V53=P/'v5_3_pipeline/training'
def sha(x:Path)->str:
 d=hashlib.sha256()
 with x.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def req(x:Path,label:str)->Path:
 if x.is_symlink() or not x.is_file():raise RuntimeError(f'{label} must be a regular file: {x}')
 return x.resolve(strict=True)
def write(x:Path,v:Any)->None:
 if x.exists() or x.is_symlink():raise FileExistsError(f'refusing overwrite {x}')
 x.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def auth()->None:
 req(AUTH,'root authorization')
 if sha(AUTH)!=AUTH_SHA:raise RuntimeError('authorization hash mismatch')
 v=json.loads(AUTH.read_text(encoding='utf-8'))
 if v.get('status')!='PASS' or v.get('permissions',{}).get('protocol_dev_generation_after_successful_training')!='ALLOW' or v.get('permissions',{}).get('heldout_access')!='DENY':raise RuntimeError('protocol dev not authorized')
def hmodule()->Any:
 req(H,'Candidate H')
 if sha(H)!=H_SHA:raise RuntimeError('Candidate H hash mismatch')
 spec=importlib.util.spec_from_file_location('candidate_h_frozen',H)
 if spec is None or spec.loader is None:raise RuntimeError('Candidate H import unavailable')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def rows()->list[dict[str,str]]:
 data=[]
 for n,line in enumerate(DEV.read_text(encoding='utf-8').splitlines(),1):
  x=json.loads(line);a,b=x['conversations']
  if a.get('from')!='human' or b.get('from')!='gpt':raise RuntimeError(f'row {n} role mismatch')
  data.append({'human':a['value'],'reference':b['value']})
 if len(data)!=240:raise RuntimeError(f'protocol dev count {len(data)} != 240')
 return data
def main()->int:
 auth();req(DEV,'protocol dev');req(RUNTIME,'runtime')
 if sha(DEV)!=DEV_SHA or sha(RUNTIME)!=RUNTIME_SHA:raise RuntimeError('frozen input hash mismatch')
 if PEFT.is_symlink() or not PEFT.is_dir():raise RuntimeError('Candidate A weights unavailable')
 if OUT.exists() or OUT.is_symlink():raise FileExistsError(f'output exists: {OUT}')
 r=rows();OUT.mkdir(parents=True);rawdir=OUT/'raw_generation'
 cmd=[str(PY),str(V53/'runtime_evaluate_v5_3.py'),'--base-model-path',str(BASE),'--peft-path',str(PEFT),'--test-jsonl',str(DEV),'--expected-test-sha256',DEV_SHA,'--runtime-source',str(RUNTIME),'--expected-runtime-source-sha256',RUNTIME_SHA,'--output-dir',str(rawdir),'--max-input-length','768','--max-new-tokens','256','--batch-size','1','--require-v5-3-contract']
 with (OUT/'raw_generation.log').open('x',encoding='utf-8') as f:
  p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);assert p.stdout
  for line in p.stdout:sys.stdout.write(line);f.write(line)
  if p.wait():raise RuntimeError('raw protocol generation failed')
 raw=json.loads((rawdir/'all_results.json').read_text(encoding='utf-8'))
 if len(raw)!=240:raise RuntimeError('raw count mismatch')
 h=hmodule();allowed={('ask','initial'),('summarize','summary')};counts={x:0 for x in ['structure','action_legal','action_stage_legal','action_stage_correct','ask_complete','summary_literal','summary_nonempty_unique','broad_safety_failures']};out=[];cases=[]
 for i,(source,row) in enumerate(zip(raw,r),1):
  if source.get('index')!=i:raise RuntimeError(f'raw index mismatch {i}')
  rawpred=source.get('raw_prediction',source.get('prediction',''));pred=h.adapt_interaction_json(row['human'],rawpred);value=json.loads(pred);ref=json.loads(row['reference']);label=(value.get('action'),value.get('stage'));reflabel=(ref.get('action'),ref.get('stage'))
  struct=label in allowed and ((label==('ask','initial') and set(value)=={'action','stage','complete','questions'}) or (label==('summarize','summary') and set(value)=={'action','stage','complete','key_findings','syndrome_tendency','need_more_info','note'}))
  askok=label!=('ask','initial') or (value.get('complete') is False and value.get('questions')==[h.ASK_REFUSAL,*h.ASK_QUESTIONS])
  facts=value.get('key_findings',[]);literal=label!=('summarize','summary') or all(isinstance(x,str) and x and x in row['human'] for x in facts);unique=label!=('summarize','summary') or bool(facts) and len(facts)==len(set(facts));safe=label!=('ask','initial') or (len(value.get('questions',[]))==3 and value['questions'][0]==h.ASK_REFUSAL)
  counts['structure']+=int(struct);counts['action_legal']+=int(value.get('action') in {'ask','summarize'});counts['action_stage_legal']+=int(label in allowed);counts['action_stage_correct']+=int(label==reflabel);counts['ask_complete']+=int(askok);counts['summary_literal']+=int(literal);counts['summary_nonempty_unique']+=int(unique);counts['broad_safety_failures']+=int(not safe)
  qsha=hashlib.sha256(row['human'].encode()).hexdigest();out.append({'index':i,'reference':row['reference'],'prediction':pred,'raw_prediction':rawpred,'query_sha256':qsha,'raw_prediction_sha256':hashlib.sha256(str(rawpred).encode()).hexdigest(),'adapted_prediction_sha256':hashlib.sha256(pred.encode()).hexdigest(),'adapter_sha256':H_SHA});cases.append({'index':i,'query_sha256':qsha,'reference_action_stage':'/'.join(reflabel),'prediction_action_stage':'/'.join(label),'structure':struct,'ask_complete':askok,'summary_literal':literal,'summary_nonempty_unique':unique,'broad_safety_failure':not safe})
 good=all(counts[x]==240 for x in counts if x!='broad_safety_failures') and counts['broad_safety_failures']==0
 write(OUT/'all_results.json',out)
 with (OUT/'adapter_application_cases.jsonl').open('x',encoding='utf-8') as f:
  for c in cases:f.write(json.dumps(c,ensure_ascii=False)+'\n')
 report={'status':'PASS' if good else 'FAIL','samples':240,'metrics':counts,'rates':{k:v/240 for k,v in counts.items() if k!='broad_safety_failures'},'raw_generation_sha256':sha(rawdir/'all_results.json'),'adapted_results_sha256':sha(OUT/'all_results.json'),'adapter_sha256':H_SHA,'heldout_access':'DENY'};write(OUT/'protocol_dev_audit.json',report)
 if not good:raise RuntimeError(f'protocol dev hard gate failed: {report}')
 print(json.dumps(report,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
