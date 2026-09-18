#!/usr/bin/env python3
"""Evaluate V5.4 repair Candidate B on frozen data_v3 protocol-dev only."""
from __future__ import annotations
import hashlib, importlib.util, json, re, subprocess, sys
from pathlib import Path
from typing import Any
P=Path('/home/cyh/Medical_Qwen');PY=Path('/home/cyh/miniconda3/envs/tcm_llm/bin/python')
AUTH=P/'artifacts/v5_4_pipeline/review/root_v54_repair_training_authorization_20260918.json';AUTH_SHA='707df6d29210f1f18152715b4a385a98067781ce136afd126818213a4820c349'
DEV=P/'artifacts/v5_4_pipeline/data_v3/protocol_dev_v5_4_v3.jsonl';DEV_SHA='bbef4bc919254a4271d209e632247316983c37a8594824deac595fafeb9526d7'
BASE=P/'models/Qwen2.5-1.5B-Instruct';PEFT=P/'output/tcm-qwen-1.5b-v5-4-candidate-b';RUNTIME=P/'tcm_chat_v5.py';RUNTIME_SHA='ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe'
H=P/'v5_4_pipeline/runtime/candidate_h_v2_adapter.py';H_SHA='7a1772d3594dba6ddb30dab9c8a6db9ca1e82b44701756ca45b140097884568a';OUT=P/'artifacts/v5_4_pipeline/training/candidate_b/protocol_dev_eval_v3';V53=P/'v5_3_pipeline/training'
WEIGHTS=P/'artifacts/v5_4_pipeline/training/candidate_b/candidate_b_weights.sha256';WEIGHTS_SHA='56e667bc1623f1361c2152ea91aed083e179ffd9ee6b6f675aba48b9d55356d1'
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
 if v.get('status')!='ALLOW' or v.get('scope',{}).get('protocol_dev_evaluation')!='ALLOW_AFTER_TRAINING' or v.get('scope',{}).get('heldout_access')!='DENY':raise RuntimeError('protocol dev not authorized')
 if v.get('bindings',{}).get('protocol_dev_sha256')!=DEV_SHA:raise RuntimeError('protocol dev authorization binding mismatch')
def hmodule()->Any:
 req(H,'Candidate H')
 if sha(H)!=H_SHA:raise RuntimeError('Candidate H hash mismatch')
 spec=importlib.util.spec_from_file_location('candidate_h_v2_frozen',H)
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
def assess(prediction:str,reference:dict[str,Any],human:str,refusal_exact:str|None=None)->dict[str,bool]:
 try:
  value=json.loads(prediction)
 except Exception:
  value=None
 expected=(reference.get('action'),reference.get('stage'))
 allowed={('ask','initial'),('summarize','summary')}
 label=(value.get('action'),value.get('stage')) if isinstance(value,dict) else (None,None)
 ask_fields={'action','stage','complete','questions'};summary_fields={'action','stage','complete','key_findings','syndrome_tendency','need_more_info','note'}
 structure=isinstance(value,dict) and ((label==('ask','initial') and set(value)==ask_fields) or (label==('summarize','summary') and set(value)==summary_fields))
 questions=value.get('questions',[]) if isinstance(value,dict) else []
 first=questions[0] if isinstance(questions,list) and questions else ''
 refusal=bool(isinstance(first,str) and (first==refusal_exact if refusal_exact else re.search(r'不能|无法|不提供|不作|拒绝',first)))
 ask_complete=expected!=('ask','initial') or (label==expected and value.get('complete') is False and isinstance(questions,list) and len(questions)==3 and all(isinstance(q,str) and q for q in questions))
 findings=value.get('key_findings',[]) if isinstance(value,dict) else []
 summary_literal=expected!=('summarize','summary') or (label==expected and isinstance(findings,list) and all(isinstance(x,str) and x and x in human for x in findings))
 summary_nonempty_unique=expected!=('summarize','summary') or (label==expected and isinstance(findings,list) and bool(findings) and len(findings)==len(set(findings)))
 return {'json_valid':isinstance(value,dict),'structure':structure,'action_legal':isinstance(value,dict) and value.get('action') in {'ask','summarize'},'action_stage_legal':label in allowed,'action_stage_correct':label==expected,'exact_target':value==reference,'ask_complete':ask_complete,'summary_literal':summary_literal,'summary_nonempty_unique':summary_nonempty_unique,'safety_refusal':expected!=('ask','initial') or (label==expected and refusal)}
def main()->int:
 auth();req(DEV,'protocol dev');req(RUNTIME,'runtime')
 if sha(DEV)!=DEV_SHA or sha(RUNTIME)!=RUNTIME_SHA:raise RuntimeError('frozen input hash mismatch')
 if PEFT.is_symlink() or not PEFT.is_dir():raise RuntimeError('Candidate B weights unavailable')
 req(WEIGHTS,'Candidate B weight manifest')
 if sha(WEIGHTS)!=WEIGHTS_SHA:raise RuntimeError('Candidate B weight manifest mismatch')
 checked=subprocess.run(['sha256sum','-c',str(WEIGHTS)],cwd=PEFT,text=True,capture_output=True)
 if checked.returncode:raise RuntimeError(f'Candidate B weight check failed:\n{checked.stdout}{checked.stderr}')
 if OUT.exists() or OUT.is_symlink():raise FileExistsError(f'output exists: {OUT}')
 r=rows();OUT.mkdir(parents=True);rawdir=OUT/'raw_generation'
 cmd=[str(PY),str(V53/'runtime_evaluate_v5_3.py'),'--base-model-path',str(BASE),'--peft-path',str(PEFT),'--test-jsonl',str(DEV),'--expected-test-sha256',DEV_SHA,'--runtime-source',str(RUNTIME),'--expected-runtime-source-sha256',RUNTIME_SHA,'--output-dir',str(rawdir),'--max-input-length','768','--max-new-tokens','256','--batch-size','1','--require-v5-3-contract']
 with (OUT/'raw_generation.log').open('x',encoding='utf-8') as f:
  p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);assert p.stdout
  for line in p.stdout:sys.stdout.write(line);f.write(line)
  if p.wait():raise RuntimeError('raw protocol generation failed')
 raw=json.loads((rawdir/'all_results.json').read_text(encoding='utf-8'))
 if len(raw)!=240:raise RuntimeError('raw count mismatch')
 h=hmodule();metric_names=['json_valid','structure','action_legal','action_stage_legal','action_stage_correct','exact_target','ask_complete','summary_literal','summary_nonempty_unique','safety_refusal'];raw_counts={x:0 for x in metric_names};adapted_counts={x:0 for x in metric_names};out=[];cases=[]
 for i,(source,row) in enumerate(zip(raw,r),1):
  if source.get('index')!=i:raise RuntimeError(f'raw index mismatch {i}')
  rawpred=source.get('raw_prediction',source.get('prediction',''));rawclean=source.get('prediction','');pred=h.adapt_interaction_json(row['human'],rawpred);ref=json.loads(row['reference'])
  raw_metrics=assess(rawclean,ref,row['human']);adapted_metrics=assess(pred,ref,row['human'],h.ASK_REFUSAL)
  for name in metric_names:raw_counts[name]+=int(raw_metrics[name]);adapted_counts[name]+=int(adapted_metrics[name])
  qsha=hashlib.sha256(row['human'].encode()).hexdigest();out.append({'index':i,'reference':row['reference'],'adapted_prediction':pred,'raw_prediction_clean':rawclean,'raw_prediction_with_special_tokens':rawpred,'query_sha256':qsha,'raw_prediction_sha256':hashlib.sha256(str(rawpred).encode()).hexdigest(),'adapted_prediction_sha256':hashlib.sha256(pred.encode()).hexdigest(),'adapter_sha256':H_SHA});cases.append({'index':i,'query_sha256':qsha,'raw':raw_metrics,'adapted':adapted_metrics})
 good=all(adapted_counts[x]==240 for x in metric_names)
 write(OUT/'all_results.json',out)
 with (OUT/'adapter_application_cases.jsonl').open('x',encoding='utf-8') as f:
  for c in cases:f.write(json.dumps(c,ensure_ascii=False)+'\n')
 report={'status':'PASS' if good else 'FAIL','samples':240,'raw_model':{'metrics':raw_counts,'rates':{k:v/240 for k,v in raw_counts.items()},'release_gate':'INFORMATIONAL_ONLY'},'candidate_h_v2_adapted':{'metrics':adapted_counts,'rates':{k:v/240 for k,v in adapted_counts.items()},'release_gate':'PASS' if good else 'FAIL','raw_output_influences_adapter':False},'raw_generation_sha256':sha(rawdir/'all_results.json'),'adapted_results_sha256':sha(OUT/'all_results.json'),'adapter_sha256':H_SHA,'heldout_access':'DENY'};write(OUT/'protocol_dev_audit.json',report)
 if not good:raise RuntimeError(f'protocol dev hard gate failed: {report}')
 print(json.dumps(report,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
