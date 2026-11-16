#!/usr/bin/env python3
"""Check the resolved eval contract and reject incomplete AppWorld results."""
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import sys


def check_config(args):
    import yaml
    output, model, samples, temperature, gpus, tp, env_url = args
    output = Path(output)
    lines = (output / 'config_stdout.txt').read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('model_engine:'))
    text = '\n'.join(lines[start:]) + '\n'
    config = yaml.safe_load(text)
    rollout = config['actor_rollout_ref']['rollout']
    trainer = config['trainer']
    assert config['actor_rollout_ref']['model']['path'] == model
    assert rollout['context_template'] == 'linear_tokens' and rollout['use_qwen3'] is False
    expected = dict(n=int(samples), do_sample=True, temperature=float(temperature), top_p=1.0, top_k=-1)
    assert all(rollout['val_kwargs'].get(key) == value for key, value in expected.items())
    assert rollout['n'] == int(samples) and rollout['temperature'] == float(temperature)
    assert rollout['top_p'] == 1.0 and rollout['top_k'] == -1
    assert rollout['multi_turn']['max_steps'] == 30
    assert rollout['max_model_len'] == 25580 and rollout['prompt_length'] == 20480 and rollout['response_length'] == 4096
    assert rollout['tensor_model_parallel_size'] == int(tp)
    assert config['data']['max_prompt_length'] == 4000 and config['data']['max_response_length'] == 21580
    assert config['task_manager']['mixture']['use_original_tasks'] is True
    assert config['task_manager']['mixture']['synthetic_data_ratio'] == 0
    assert config['exp_manager']['val_rollout_mode'] == 'woexp'
    assert config['env_service']['env_url'] == env_url
    assert trainer['val_only'] is True and trainer['val_before_train'] is True and trainer['resume_mode'] == 'disable'
    assert trainer['n_gpus_per_node'] == int(gpus)
    assert trainer['validation_data_dir'] == str(output / 'validation_log')
    (output / 'resolved_config.yaml').write_text(text)
    print('Verified validation-only config: linear_tokens, temperature='+temperature+', samples='+samples)


def summarize(args):
    output, model, samples, temperature = args
    output, samples = Path(output), int(samples)
    rows = [json.loads(line) for line in (output / 'validation_log/0.jsonl').read_text().splitlines() if line.strip()]
    # The original evaluated runtime writes input/output/score without task_id.
    # Its initial input is identical across repeats and unique for each dev task.
    have_ids = all(row.get('task_id') for row in rows)
    grouping = 'task_id' if have_ids else 'initial_input_sha256'
    counts = Counter(str(row['task_id']) if have_ids else hashlib.sha256(row['input'].encode()).hexdigest() for row in rows)
    assert len(counts) == 57 and len(rows) == 57 * samples, (len(counts), len(rows))
    assert set(counts.values()) == {samples}, counts
    scores = [float(row['score']) for row in rows]
    assert all(math.isfinite(score) and 0 <= score <= 1 for score in scores), 'Invalid score'
    result = dict(model_path=model, split='appworld_dev', tasks=57, samples_per_task=samples,
                  rollouts=len(rows), task_grouping=grouping, temperature=float(temperature), mean_at_n=sum(scores)/len(scores),
                  rollout_file=str(output / 'validation_log/0.jsonl'))
    (output / 'rollout_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    {'config': check_config, 'summary': summarize}[sys.argv[1]](sys.argv[2:])
