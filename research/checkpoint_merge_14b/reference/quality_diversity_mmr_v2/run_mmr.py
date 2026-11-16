"""Re-estimate the historical MMR/TIES recipe from the complete dev mean@8 matrix."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent
ROOT = PROJECT.parent
V3 = PROJECT/'quality_diversity_v3'
V4 = PROJECT/'quality_diversity_v4'
V5 = PROJECT/'quality_diversity_v5'
NORMAL = V4/'test_normal'
BASE = Path('/mnt/public/hf_models/Qwen3-8B')
sys.path.insert(0,str(V3))
from run_stage1 import MODEL_ROOT, EVAL, SOURCE, digest, now, write_json, environment
from analyze import read_run, paired_comparison
sys.path.insert(0,str(V5))
from run_pilot import free_gpus
sys.path.insert(0,str(NORMAL))
import queue_test_normal as normal
sys.path.insert(0,str(ROOT/'agent_merge/research/model_merge'))
import weighted_merge as wm


def select_bank(vectors, k=5, lam=.7, tau=.02):
    pool = sorted(vectors)
    quality = {s:float(vectors[s].mean()) for s in pool}
    similarity = {i:{j:float(vectors[i]@vectors[j]/(np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j])))
                     if np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j]) else 0. for j in pool} for i in pool}
    selected = [max(pool,key=quality.get)]
    trace = []
    while len(selected)<k:
        scores = {i:lam*quality[i]-(1-lam)*max(similarity[i][j] for j in selected) for i in pool if i not in selected}
        best = max(scores,key=scores.get)
        trace.append({'selected_before':list(selected),'scores':scores,'chosen':best})
        selected.append(best)
    bank = sorted(selected)
    values = np.array([quality[s] for s in bank])
    weights = np.exp((values-values.max())/tau)
    weights /= weights.sum()
    return {'quality':quality,'cosine':similarity,'greedy_order':selected,'trace':trace,
            'weights':dict(zip(bank,map(float,weights))),
            'same_bank_as_quality_only':set(bank)==set(sorted(pool,key=quality.get,reverse=True)[:k])}


def identity(path):
    source = wm.ModelSource(wm.ModelEntry('audit',Path(path),1.))
    shards = {}
    for name in sorted({spec.file_name for spec in source.specs.values()}):
        file = Path(path)/name
        with file.open('rb') as stream:
            prefix = stream.read(8)
            size = struct.unpack('<Q',prefix)[0]
            if size > 100_000_000:
                raise ValueError('Invalid safetensors header')
            header = stream.read(size)
        offsets = [v['data_offsets'][1] for k,v in json.loads(header).items() if k!='__metadata__']
        if file.stat().st_size != 8+size+max(offsets):
            raise ValueError('Invalid shard length')
        shards[name] = {'bytes':file.stat().st_size,'mtime_ns':file.stat().st_mtime_ns,
                        'header_sha256':hashlib.sha256(prefix+header).hexdigest()}
    return {'path':str(path),'shards':shards,'tensor_count':len(source.specs),
            'assets':{str(p):digest(p) for p in Path(path).iterdir() if p.is_file() and (p.suffix in ('.json','.jinja'))},
            'hash_scope':'Source payload identity uses header/size/mtime, not full tensor SHA256.'}


def prepare():
    if (WORK/'protocol.json').exists():
        return json.loads((WORK/'protocol.json').read_text())
    manifest = json.loads((V3/'task_manifest.json').read_text())['tasks']
    paths = {s:V3/f'sweep8_240_300/runs/ck{s}_mean8/validation_log/0.jsonl' for s in (240,250,260,270,280,300)}
    vectors = {s:np.array([sum(scores[t])/8 for t in sorted(manifest)])
               for s,path in paths.items() for scores in [read_run(path,8,manifest)]}
    selection = select_bank(vectors)
    write_json(WORK/'selection.json',selection)
    config = {'models':[{'model':str(MODEL_ROOT/f'checkpoint-{s}'),'parameters':{'weight':w,'density':1.}}
                        for s,w in selection['weights'].items()],
              'merge_method':'ties','dtype':'float32','base_model':str(BASE),
              'parameters':{'normalize':True,'int8_mask':False,'lambda':1.}}
    (WORK/'merge.yml').write_text(yaml.safe_dump(config,sort_keys=False))
    model_paths = [BASE]+[MODEL_ROOT/f'checkpoint-{s}' for s in selection['weights']]
    sources = [wm.ModelSource(wm.ModelEntry(str(i),p,1.)) for i,p in enumerate(model_paths)]
    reference = sources[0].specs
    for source in sources[1:]:
        if set(source.specs)!=set(reference) or any(source.specs[k].shape!=reference[k].shape for k in reference):
            raise ValueError('Base/checkpoint tensor shapes differ')
    wm._validate_tokenizer_assets(sources[1:])
    wm._load_and_validate_config(sources[1:])
    seed = int.from_bytes(hashlib.sha256(b'mmr-mean8-20261023').digest()[:4],'big')%(2**31-8)
    import mergekit.merge_methods.generalized_task_arithmetic as gta
    scripts = [Path(__file__),WORK/'merge.yml',WORK/'selection.json',V3/'run_stage1.py',V3/'analyze.py',
               V3/'python_seeded.sh',V3/'task_manifest.json',V5/'run_pilot.py',V4/'study.py',
               NORMAL/'queue_test_normal.py',NORMAL/'protocol.json',NORMAL/'validate_test_normal.py',
               EVAL/'rollout_appworld.sh',EVAL/'validate_and_summarize.py',EVAL/'prepare_source.py',
               EVAL/'runtime_manifest.json',Path(gta.__file__),ROOT/'agent_merge/research/model_merge/weighted_merge.py']
    predecessor = json.loads((NORMAL/'protocol.json').read_text())['predecessor']
    p = {'study':'mmr-mean8-20261023','created_utc':now(),'selection':selection,
         'lambda':.7,'K':5,'tau':.02,'excluded':{'290':'Unresolved historical weight anomaly'},
         'job':{'id':'mmr_mean8','model':str(WORK/'model'),'engine_seed':seed},
         'eval':{'split':'dev','tasks':57,'samples':8,'rollouts':456,'temperature':.5,'dtype':'float32','gpus':8},
         'models':[identity(path) for path in model_paths],'predecessor':predecessor,
         'files':{str(path):digest(path) for path in scripts},'behavior_inputs':{str(p):digest(p) for p in paths.values()},
         'merge':'Original weighted-TIES, raw pretrained base, density=1, normalize=true; no outer anchor interpolation.',
         'assets':'Copy checkpoint300 config/tokenizer/template/generation assets after merge to match the current evaluation contract.',
         'interpretation':'Re-estimated recipe, not a byte-identical old-model repeat. Dev reused. MMR bank equals quality-only when flagged.',
         'queue_order':['V5 pilot','new MMR dev@8','linear280 test_normal@8'],
         'failure_policy':'Stop and retain logs; no automatic retries or seed selection; downstream test_normal remains queued on MMR failure.'}
    write_json(WORK/'protocol.json',p)
    return p


def verify(p):
    for path,expected in {**p['files'],**p['behavior_inputs']}.items():
        if digest(Path(path))!=expected:
            raise ValueError('Frozen input changed: '+path)
    if digest(V5/'protocol.json')!=p['predecessor']['protocol_sha256']:
        raise ValueError('V5 protocol changed')
    for expected in p['models']:
        if identity(Path(expected['path']))!=expected:
            raise ValueError('Source model changed: '+expected['path'])


def status(state, **extra):
    write_json(WORK/'status.json',{'state':state,'updated_utc':now(),**extra})


def merge_command(config, destination):
    return [str(ROOT/'venv/mergekit/bin/mergekit-yaml'),str(config),str(destination),
            '--device','cpu','--no-cuda','--no-read-to-gpu','--no-multi-gpu','--num-threads','8',
            '--out-shard-size','1B','--no-copy-tokenizer','--no-write-model-card']


def build(p):
    destination = Path(p['job']['model'])
    stage = WORK/'model.building'
    if destination.exists() or stage.exists():
        raise ValueError('Existing model requires inspection')
    if shutil.disk_usage(WORK).free<50_000_000_000:
        raise ValueError('Need 50 GB free')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
    with (WORK/'build.log').open('x') as log:
        subprocess.run(merge_command(WORK/'merge.yml',stage),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    wm._copy_hf_assets(MODEL_ROOT/'checkpoint-300',stage)
    source = wm.ModelSource(wm.ModelEntry('output',stage,1.))
    anchor = wm.ModelSource(wm.ModelEntry('anchor',MODEL_ROOT/'checkpoint-300',1.))
    wm._validate_tensor_specs([anchor,source])
    wm._validate_tokenizer_assets([anchor,source])
    with ExitStack() as stack:
        source.attach(stack)
        for key in source.specs:
            tensor = source.read(key,None)
            if tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite/non-fp32 merged tensor: '+key)
    shards = [{'file':name,'bytes':(stage/name).stat().st_size,'sha256':wm._sha256_file(stage/name)}
              for name in sorted({s.file_name for s in source.specs.values()})]
    write_json(stage/'EXPORT_READY.json',{'weights':shards})
    write_json(stage/'merge_audit.json',{'selection':p['selection'],'base':str(BASE),'operator':'ties',
               'density':1.,'normalize':True,'tensor_count':len(source.specs),'weights':shards,
               'assets_source':str(MODEL_ROOT/'checkpoint-300')})
    stage.rename(destination)
    write_json(WORK/'build_status.json',{'state':'complete','completed_utc':now()})


def eval_command(p, dry=False):
    output = WORK/('preflight' if dry else 'run')
    args = ['bash',str(EVAL/'rollout_appworld.sh'),'--source',str(SOURCE),'--python',str(V3/'python_seeded.sh'),
            '--model-path',p['job']['model'],'--output-dir',str(output),'--samples','8','--temperature','.5',
            '--dtype','float32','--devices','0,1,2,3,4,5,6,7','--train-batch-size','8',
            '--gpu-memory-utilization','.85','--ray-temp-dir','/tmp/mmr_mean8']
    if dry: args.append('--dry-run')
    return args,output


def summarize(p):
    from run_pilot import validate_config
    validate_config(WORK/'run',p['job'])
    config = yaml.safe_load((WORK/'run/resolved_config.yaml').read_text())
    if config['data'].get('val_type') in ('test_normal','test_challenge'):
        raise ValueError('Unexpected evaluation split')
    manifest = json.loads((V3/'task_manifest.json').read_text())['tasks']
    scores = read_run(WORK/'run/validation_log/0.jsonl',8,manifest)
    successes = int(sum(map(sum,scores.values())))
    result = {'successes':successes,'mean_at_8':successes/456,'pass_at_8':sum(any(v) for v in scores.values())/57,
              'tasks':57,'rollouts':456,'selection':p['selection'],'comparisons':[]}
    for label,path in [('linear280_v5_repeat',V5/'runs/linear280_repeat/validation_log/0.jsonl'),
                       ('ck300_historical',V3/'sweep8_240_300/runs/ck300_mean8/validation_log/0.jsonl')]:
        ref = read_run(path,8,manifest)
        result['comparisons'].append({'reference':label,**paired_comparison(scores,ref,manifest)})
    write_json(WORK/'results.json',result)
    (WORK/'RESULTS.md').write_text('# MMR Re-estimated From Dev Mean@8\n\n'
        f'Completed UTC: {now()}\n\n| Model | Successes | Mean@8 | Pass@8 |\n|---|---:|---:|---:|\n'
        f'| MMR/TIES mean8 | {successes}/456 | {result["mean_at_8"]:.6f} | {result["pass_at_8"]:.6f} |\n\n'
        '57 dev tasks x 8, T=0.5, fp32, eight GPUs.\n'
        'Reused dev tasks, one session, exploratory unadjusted intervals in results.json.\n'
        'No simultaneous fresh ck300 control. Same selected bank as quality-only; no isolated diversity benefit claim.\n')
    return result


def run(p):
    with (WORK/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify(p)
        while True:
            previous = json.loads((V5/'status.json').read_text())
            normal.predecessor_ready(p,previous)
            build_status = V5/'build_status.json'
            if build_status.exists():
                value = json.loads(build_status.read_text())
                if value['state']=='failed': raise RuntimeError('V5 construction failed')
                if value['state']=='complete': break
            status('waiting_for_v5_builder',predecessor_state=previous)
            time.sleep(30)
        status('building_cpu')
        build(p)
        args,output = eval_command(p,dry=True)
        with (WORK/'preflight.log').open('x') as log:
            subprocess.run(args,env=environment(p['job']),stdout=log,stderr=subprocess.STDOUT,check=True)
        from run_pilot import validate_config
        validate_config(output,p['job'])
        write_json(WORK/'preflight_passed.json',{'completed_utc':now()})
        while not normal.predecessor_ready(p,json.loads((V5/'status.json').read_text())):
            status('waiting_for_v5_evaluations')
            time.sleep(30)
        with (V3/'queue.lock').open('a') as shared:
            status('waiting_for_shared_queue')
            fcntl.flock(shared,fcntl.LOCK_EX)
            verify(p)
            free_gpus()
            status('running_dev8',job=p['job'])
            args,output = eval_command(p)
            with (WORK/'evaluation.log').open('x') as log:
                subprocess.run(args,env=environment(p['job']),stdout=log,stderr=subprocess.STDOUT,check=True)
            result = summarize(p)
            write_json(output/'session_metadata.json',{**p['job'],'completed_utc':now(),
                       'engine_replica_seeds':[p['job']['engine_seed']+i for i in range(8)]})
            status('dev8_complete',mean_at_8=result['mean_at_8'])
        normal.verify(json.loads((NORMAL/'protocol.json').read_text()))
        with (NORMAL/'queue_after_mmr.log').open('x') as log:
            process = subprocess.Popen([sys.executable,'-u',str(NORMAL/'queue_test_normal.py'),'run'],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(NORMAL/'queue_process.json',{'pid':process.pid,'started_utc':now(),'scheduled_after':'MMR dev@8'})
        status('complete',mean_at_8=result['mean_at_8'],test_normal_pid=process.pid)


def launch(p):
    verify(p)
    if (WORK/'queue.log').exists() or (WORK/'schedule_amendment.json').exists():
        raise ValueError('Queue already launched or amended; inspect before retrying')
    previous = json.loads((NORMAL/'status.json').read_text())
    record = json.loads((NORMAL/'queue_process.json').read_text())
    pid = record['pid']
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if previous['state']!='waiting_for_v5' or str(NORMAL/'queue_test_normal.py').encode() not in cmd:
        raise ValueError('test_normal is not the expected idle waiting process')
    start = normal.process_start(pid)
    amendment = {'created_utc':now(),'user_request':'Insert re-estimated MMR dev@8 before linear280 test_normal',
                 'previous_test_normal_process':record,'previous_state':previous,
                 'original_test_normal_protocol_sha256':digest(NORMAL/'protocol.json'),
                 'new_order':p['queue_order'],'test_normal_protocol_unchanged':True}
    write_json(WORK/'schedule_amendment.json',amendment)
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if normal.process_start(pid)!=start: break
        time.sleep(.1)
    else: raise RuntimeError('Waiting process did not exit; no new queue launched')
    normal.status('waiting_for_mmr',mmr_protocol=str(WORK/'protocol.json'),queue_order=p['queue_order'])
    with (WORK/'queue.log').open('x') as log:
        process = subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'run'],stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(WORK/'queue_process.json',{'pid':process.pid,'started_utc':now()})
    print('MMR queue PID',process.pid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','launch','run'])
    mode = parser.parse_args().mode
    p = prepare()
    if mode=='prepare': print(json.dumps(p['selection'],indent=2))
    elif mode=='launch': launch(p)
    else:
        try: run(p)
        except Exception as exc:
            status('failed',reason=str(exc))
            normal.status('blocked_by_mmr_failure',reason=str(exc))
            raise
