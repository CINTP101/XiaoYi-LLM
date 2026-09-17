#!/usr/bin/env python3
"""Authorized V5.4 Candidate A training; no heldout path exists here."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
from typing import Any

PROJECT = Path('/home/cyh/Medical_Qwen')
PYTHON = Path('/home/cyh/miniconda3/envs/tcm_llm/bin/python')
AUTH = PROJECT / 'artifacts/v5_4_pipeline/review/root_data_v2_acceptance.json'
AUTH_SHA = '8d7217e3394ee4b94cd3739d6ce2463bd81723791d061dcda4bb660781bf3c93'
TRAIN = PROJECT / 'artifacts/v5_4_pipeline/data_v2/train_v5_4_v2.jsonl'
TRAIN_SHA = 'd07d2bfdbe002473ccc4c5fbd3b87ff2888873cbda9a7e60790019a6755f4677'
BASE = PROJECT / 'models/Qwen2.5-1.5B-Instruct'
PEFT = PROJECT / 'output/tcm-qwen-1.5b-v5-3'
OUTPUT = PROJECT / 'output/tcm-qwen-1.5b-v5-4-candidate-a'
RUNTIME = PROJECT / 'tcm_chat_v5.py'
RUNTIME_SHA = 'ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe'
ARTIFACT = PROJECT / 'artifacts/v5_4_pipeline/training/candidate_a'
PREFLIGHT = PROJECT / 'artifacts/v5_4_pipeline/training/candidate_a_preflight.json'
V53 = PROJECT / 'v5_3_pipeline/training'
PROTECTED = (
 ('v5_1', PROJECT/'output/tcm-qwen-1.5b-v5-1', PROJECT/'artifacts/v5_2_pipeline/v5-1_baseline.sha256'),
 ('v5_2', PROJECT/'output/tcm-qwen-1.5b-v5-2', PROJECT/'artifacts/v5_3_pipeline/stage0/v5-2_baseline.sha256'),
 ('v5_3', PEFT, PROJECT/'artifacts/v5_4_pipeline/stage0/v5-3_baseline.sha256'),
 ('candidate_c', PROJECT/'output/tcm-qwen-1.5b-v5-3-candidate-hardened-from-v5-2', PROJECT/'artifacts/v5_3_pipeline/training/candidates/hardened_from_v5_2/candidate_weights.sha256'),
 ('candidate_d', PROJECT/'output/tcm-qwen-1.5b-v5-3-candidate-precision-from-c', PROJECT/'artifacts/v5_3_pipeline/training/candidates/precision_from_c/candidate-d_weights.sha256'),
 ('candidate_e', PROJECT/'output/tcm-qwen-1.5b-v5-3-candidate-e-minimal-from-d', PROJECT/'artifacts/v5_3_pipeline/training/candidates/minimal_correction_from_d/candidate-e_weights.sha256'),
)
def sha(p: Path) -> str:
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
 return d.hexdigest()
def regular_file(p:Path,label:str)->Path:
 if p.is_symlink() or not p.is_file(): raise RuntimeError(f'{label} must be a regular file: {p}')
 return p.resolve(strict=True)
def regular_dir(p:Path,label:str)->Path:
 if p.is_symlink() or not p.is_dir(): raise RuntimeError(f'{label} must be a regular directory: {p}')
 return p.resolve(strict=True)
def check(name:str, root:Path, manifest:Path)->dict[str,Any]:
 regular_dir(root,name); regular_file(manifest,name+' manifest')
 entries=[x for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
 r=subprocess.run(['sha256sum','-c',str(manifest)],cwd=root,text=True,capture_output=True)
 if r.returncode: raise RuntimeError(f'{name} hash check failed:\n{r.stdout}{r.stderr}')
 return {'name':name,'entries':len(entries),'manifest':str(manifest),'status':'PASS'}
def write_new(p:Path,v:Any)->None:
 if p.exists() or p.is_symlink(): raise FileExistsError(f'refusing overwrite: {p}')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def preflight()->dict[str,Any]:
 regular_file(AUTH,'authorization')
 if sha(AUTH)!=AUTH_SHA: raise RuntimeError('authorization SHA mismatch')
 auth=json.loads(AUTH.read_text(encoding='utf-8'))
 if auth.get('status')!='PASS' or auth.get('decision')!='GO_V5_4_CANDIDATE_A_TRAINING' or auth.get('permissions',{}).get('training')!='ALLOW' or auth.get('permissions',{}).get('heldout_access')!='DENY': raise RuntimeError('authorization not valid for candidate training')
 if auth.get('accepted_data',{}).get('train_path')!=str(TRAIN) or auth.get('accepted_data',{}).get('train_sha256')!=TRAIN_SHA: raise RuntimeError('accepted train differs')
 regular_file(TRAIN,'training input'); regular_file(RUNTIME,'runtime'); regular_dir(BASE,'base');regular_dir(PEFT,'V5.3 adapter')
 if sha(TRAIN)!=TRAIN_SHA or sha(RUNTIME)!=RUNTIME_SHA: raise RuntimeError('frozen input hash mismatch')
 if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError(f'output already exists: {OUTPUT}')
 protected=[check(*item) for item in PROTECTED]
 free=int(subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True).stdout.splitlines()[0].strip())
 if free<6000: raise RuntimeError(f'GPU memory too low: {free} MiB')
 return {'status':'PASS','authorization_sha256':AUTH_SHA,'train':str(TRAIN),'train_sha256':TRAIN_SHA,'runtime_sha256':RUNTIME_SHA,'peft_start':str(PEFT),'output':str(OUTPUT),'gpu_free_mib':free,'protected_hash_checks':protected,'hyperparameters':{'epochs':1,'learning_rate':3e-6,'train_batch':1,'eval_batch':1,'gradient_accumulation_steps':8,'max_length':768,'validation_percent':5,'warmup_ratio':0.03,'seed':42,'data_seed':42,'resume_from_checkpoint':False,'logging_steps':1,'eval_steps':25,'save_steps':25},'heldout_access':'DENY'}
def run(command:list[str],log:Path)->None:
 with log.open('x',encoding='utf-8') as f:
  p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);assert p.stdout
  for line in p.stdout: sys.stdout.write(line);f.write(line)
  code=p.wait()
 if code: raise RuntimeError(f'training exit code {code}; resume was not attempted')
def evidence()->dict[str,Any]:
 summary=json.loads((OUTPUT/'v5_3_runtime_training_summary.json').read_text(encoding='utf-8'))
 curve=(OUTPUT/'train_curve.csv').read_text(encoding='utf-8').splitlines()[1:]
 evals=(OUTPUT/'eval_curve.csv').read_text(encoding='utf-8').splitlines()[1:]
 losses=[float(r.split(',')[1]) for r in curve if len(r.split(','))>1 and r.split(',')[1]]
 steps=math.ceil((1600-math.ceil(1600*.05))/8)
 if int(summary.get('global_step',-1))!=steps or len(losses)!=steps: raise RuntimeError('training step/loss evidence incomplete')
 n=min(10,len(losses));initial=sum(losses[:n])/n;final=sum(losses[-n:])/n
 if final>=initial: raise RuntimeError('loss decline gate failed')
 return {'status':'PASS','expected_global_step':steps,'actual_global_step':int(summary['global_step']),'per_step_losses':len(losses),'internal_eval_records':len(evals),'initial_loss_window_mean':initial,'final_loss_window_mean':final,'loss_decline_verified':True}
def weight_manifest()->dict[str,Any]:
 files=sorted(p for p in OUTPUT.rglob('*') if p.is_file() and not p.is_symlink())
 if not files: raise RuntimeError('Candidate A has no regular files')
 path=ARTIFACT/'candidate_a_weights.sha256';path.write_text(''.join(f'{sha(p)}  {p.relative_to(OUTPUT)}\n' for p in files),encoding='utf-8')
 return {'status':'PASS','path':str(path),'sha256':sha(path),'files':len(files)}
def main()->int:
 a=argparse.ArgumentParser();a.add_argument('--preflight-only',action='store_true');args=a.parse_args();p=preflight()
 if args.preflight_only: write_new(PREFLIGHT,p);print(json.dumps(p,ensure_ascii=False,indent=2));return 0
 if ARTIFACT.exists() or ARTIFACT.is_symlink(): raise FileExistsError(f'artifact exists: {ARTIFACT}')
 ARTIFACT.mkdir(parents=True);write_new(ARTIFACT/'training_preflight.json',p)
 subprocess.run([str(PYTHON),str(V53/'verify_v5_3_prompt_parity.py'),'--jsonl',str(TRAIN),'--expected-sha256',TRAIN_SHA,'--base-model-path',str(BASE),'--runtime-source',str(RUNTIME),'--expected-runtime-source-sha256',RUNTIME_SHA,'--output',str(ARTIFACT/'training_runtime_prompt_parity.json')],check=True)
 parity=json.loads((ARTIFACT/'training_runtime_prompt_parity.json').read_text(encoding='utf-8'))
 if parity.get('total')!=1600 or not parity.get('all_exact'): raise RuntimeError('all-1600 prompt parity gate failed')
 cmd=[str(PYTHON),str(V53/'runtime_sft_v5_3.py'),'--base-model-path',str(BASE),'--peft-path',str(PEFT),'--train-jsonl',str(TRAIN),'--expected-train-sha256',TRAIN_SHA,'--runtime-source',str(RUNTIME),'--expected-runtime-source-sha256',RUNTIME_SHA,'--output-dir',str(OUTPUT),'--model-max-length','768','--validation-split-percentage','5','--per-device-train-batch-size','1','--per-device-eval-batch-size','1','--gradient-accumulation-steps','8','--num-train-epochs','1','--learning-rate','3e-6','--warmup-ratio','0.03','--logging-steps','1','--eval-steps','25','--save-steps','25','--seed','42','--data-seed','42','--report-to','tensorboard']
 run(cmd,ARTIFACT/'train.log');write_new(ARTIFACT/'training_evidence.json',evidence());write_new(ARTIFACT/'candidate_a_weight_manifest.json',weight_manifest());write_new(ARTIFACT/'postflight.json',{'status':'PASS','protected_hash_checks':[check(*i) for i in PROTECTED]});print(json.dumps({'status':'PASS','artifact':str(ARTIFACT),'output':str(OUTPUT)},ensure_ascii=False));return 0
if __name__=='__main__': raise SystemExit(main())
