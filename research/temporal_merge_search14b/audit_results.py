"""Independent round-1 JSONL, resolved protocol and service-range audit.

Does not import the search runner, rollout receipt reader, merge or statistics
implementation. Completed cells are audited without trusting their score vector.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import yaml


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def audit(root, require_complete=False, service_path=None):
    inputs = read(root/'inputs.json')
    recipes = read(root/'recipes.json')
    environment = read(root/'environment.json')
    execution = inputs['config']
    amendment_file = root/'execution-amendment.json'
    amendment = read(amendment_file) if amendment_file.exists() else None
    if amendment:
        require(amendment['original_inputs_sha256']==sha_bytes((root/'inputs.json').read_bytes()),'Wrong amendment inputs')
        execution = amendment['execution_config']
        require(execution==dict(inputs['config'],max_num_batched_tokens=4096,
                                gpu_memory_utilization=.80,ray_namespace=310),'Undeclared execution changes')
        require(amendment['independent_auditor_sha256']==sha_bytes(Path(__file__).read_bytes()),'Amended auditor changed')
    require(sha_bytes((root/'recipes.json').read_bytes())==inputs['recipes_sha256'], 'Recipes changed')
    for file, expected in inputs['code_sha256'].items():
        require(sha_bytes(Path(file).read_bytes())==expected, f'Protected code changed: {file}')
    service = Path(service_path or '/mnt/infini/zym-tmp/merge/appworld-service.log')
    events = re.compile(r'Creating instance with env_type: appworld, task_id: ([^,\s]+), instance_id: ([0-9a-f]{32})')
    results, pending, seen_instances = {}, [], set()
    all_task_keys = None
    instructions = environment['appworld']['task_instructions']
    for name in recipes['order']:
        directory = root/'evaluations'/name
        receipt = directory/'CELL_AUDIT.json'
        if not receipt.is_file():
            pending.append(name)
            continue
        recorded = read(receipt)
        if amendment:
            proof = read(directory/'EXECUTION_AMENDMENT.json')
            require(proof['amendment_sha256']==sha_bytes(amendment_file.read_bytes()),'Wrong cell execution amendment')
        payload = (directory/'validation_log/0.jsonl').read_bytes()
        groups, prompts = defaultdict(list), {}
        for line in payload.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            require(type(row.get('score')) in [int,float] and row['score'] in [0,1], 'Invalid binary score')
            prompt = row.get('input')
            require(isinstance(prompt,str) and bool(prompt), 'Missing initial input')
            key = sha_bytes(prompt.encode())
            groups[key].append(row['score'])
            prompts[key] = prompt
        require(len(groups)==57 and sum(map(len,groups.values()))==456 and
                all(len(v)==8 for v in groups.values()), f'Incomplete raw sample matrix: {name}')
        keys = sorted(groups)
        if all_task_keys is None:
            all_task_keys = keys
        require(keys==all_task_keys,'Different task prompts across methods')
        vector = [sum(groups[k])/8 for k in keys]
        families = []
        for key in keys:
            matches = {tid.split('_')[0] for tid,instruction in instructions.items() if instruction in prompts[key]}
            require(len(matches)==1,'Ambiguous task-family mapping')
            families.append(matches.pop())
        result = {'task_keys':keys,'vector':vector,'mean_at_8':sum(sum(v) for v in groups.values())/456,
                  'pass_at_8':sum(max(v)>0 for v in groups.values())/57,'rollouts':456,
                  'task_grouping':'initial_input_sha256','log_sha256':sha_bytes(payload),'task_families':families}
        expected = read(directory/'VERIFIED.json')
        require(expected==recorded['result'], 'Runner receipts disagree')
        for key,value in result.items():
            if key in ['mean_at_8','pass_at_8']:
                require(abs(expected[key]-value)<1e-15,'Recomputed metric differs')
            else:
                require(expected[key]==value,f'Recomputed receipt differs: {name}:{key}')

        config_bytes = (directory/'resolved_config.yaml').read_bytes()
        require(sha_bytes(config_bytes)==recorded['resolved_config_sha256'],'Resolved config changed')
        config = yaml.safe_load(config_bytes)
        rollout = config['actor_rollout_ref']['rollout']
        model = (Path('/mnt/infini/zym-tmp/merge/runs/qwen3-14b-dev8-v2/models/tsg_tv')
                 if name=='tsg_tv' else root/'models'/name)
        require(Path(config['actor_rollout_ref']['model']['path']).resolve()==model.resolve(),'Wrong actual model')
        for key,value in {'dtype':'float32','seed':20261110,'n':8,'temperature':.9,'top_p':1.,'top_k':-1,
                          'context_template':'linear_tokens','use_qwen3':False,'max_model_len':25580,
                          'prompt_length':20480,'response_length':4096,'max_num_batched_tokens':execution['max_num_batched_tokens'],
                          'tensor_model_parallel_size':1,'gpu_memory_utilization':execution['gpu_memory_utilization']}.items():
            require(rollout[key]==value,f'Protocol differs: {name}:{key}')
        require(rollout['multi_turn']['max_steps']==30,'Wrong action budget')
        require(rollout['engine_kwargs']['vllm']['attention_backend']=='TRITON_ATTN','Wrong attention backend')
        require(all(rollout['val_kwargs'].get(k)==v for k,v in
                    dict(n=8,do_sample=True,temperature=.9,top_p=1.,top_k=-1).items()),
                'Wrong validation sampling')
        for key,value in {'val_only':True,'val_before_train':True,'resume_mode':'disable',
                          'n_gpus_per_node':8,'nnodes':1}.items():
            require(config['trainer'][key]==value,f'Wrong trainer protocol: {key}')
        require(config['data']['seed']==2027 and config['data']['max_prompt_length']==4000 and
                config['data']['max_response_length']==21580,'Wrong data protocol')
        require(config['exp_manager']['val_rollout_mode']=='woexp','Experience augmentation enabled')

        stat = service.stat()
        require([stat.st_dev,stat.st_ino]==recorded['service_log_identity'],'Service log identity changed')
        start,end = recorded['service_start_offset'],recorded['service_end_offset']
        require(0<=start<=end<=stat.st_size,'Invalid service log byte range')
        with service.open('rb') as stream:
            stream.seek(start)
            chunk = stream.read(end-start)
        instances = {}
        for match in events.finditer(chunk.decode(errors='replace')):
            task,instance = match.groups()
            require(instance not in instances or instances[instance]==task,'Service ID task changed')
            instances[instance] = task
        counts = Counter(instances.values())
        require(counts==Counter({t:8 for t in environment['appworld']['task_ids']}),'Service task counts differ')
        require(not (seen_instances & set(instances)),'Service instances reused across cells')
        seen_instances.update(instances)
        results[name] = {'result':result,'resolved_config_sha256':sha_bytes(config_bytes),
                         'service_instances':len(instances),'service_range_sha256':sha_bytes(chunk),
                         'service_start_offset':start,'service_end_offset':end}
    require(not require_complete or not pending, f'Incomplete round: {pending}')
    return {'checked_at':datetime.now(timezone.utc).isoformat(),'complete':list(results),'pending':pending,
            'cells':results,'trajectories':sum(r['result']['rollouts'] for r in results.values()),
            'unique_service_instances':len(seen_instances),'inputs_sha256':sha_bytes((root/'inputs.json').read_bytes()),
            'scope':'Independent raw binary scores, prompt/family alignment, frozen code, actual protocol and service-instance byte ranges; no significance or source tensor audit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get(
        'BCM_RUN_ROOT', Path(__file__).resolve().parents[2].parent/'runs/temporal-search14b-r1')))
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = audit(args.root,args.require_complete)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':
    main()
