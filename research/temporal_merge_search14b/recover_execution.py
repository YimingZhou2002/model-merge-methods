"""Apply a recorded common execution amendment without editing frozen operators."""
import fcntl
import json
from pathlib import Path
import time

from . import run as original


def main():
    root = original.ROOT
    frozen = json.loads((root/'inputs.json').read_text())
    amendment = json.loads((root/'execution-amendment.json').read_text())
    if amendment['original_inputs_sha256'] != original.sha(root/'inputs.json'):
        raise ValueError('Execution amendment refers to different original inputs')
    if amendment['recovery_runner_sha256'] != original.sha(Path(__file__)):
        raise ValueError('Recovery runner changed after amendment freeze')
    config = amendment['execution_config']
    expected = dict(frozen['config'], max_num_batched_tokens=4096,
                    gpu_memory_utilization=.80, ray_namespace=310)
    if config != expected:
        raise ValueError('Only the declared common memory/batch/namespace amendment is allowed')
    with (root/'study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.verify_frozen(frozen)
        results = {}
        for name in original.ORDER:
            original.verify_frozen(frozen)
            cell_config = dict(config, ray_namespace=config['ray_namespace']+original.ORDER.index(name))
            model = original.OLD/'models/tsg_tv' if name=='tsg_tv' else root/'models'/name
            if name!='tsg_tv':
                started = time.monotonic()
                while not (root/'independent-export-audit.json').exists():
                    for filename in ['export.log','post-export-audit.log']:
                        if 'Traceback (most recent call last)' in (root/filename).read_text():
                            raise RuntimeError(f'Export/audit failed: {filename}')
                    if time.monotonic()-started>14400:
                        raise TimeoutError('Independent export readiness not reached')
                    original.study.state(config,'waiting_for_export_audit',job=name)
                    time.sleep(30)
                export_audit = json.loads((root/'independent-export-audit.json').read_text())
                if export_audit['inputs_sha256'] != original.sha(root/'inputs.json') or export_audit['recipes_sha256'] != frozen['recipes_sha256']:
                    raise ValueError('Export audit identities differ')
            receipt = root/'evaluations'/name/'CELL_AUDIT.json'
            execution_receipt = receipt.parent/'EXECUTION_AMENDMENT.json'
            if receipt.exists():
                result = original.study.evaluate(cell_config,name,model,0)
                audit = json.loads(receipt.read_text())
                if result!=audit['result'] or audit['resolved_config_sha256']!=original.sha(receipt.parent/'resolved_config.yaml'):
                    raise ValueError('Completed amended evaluation changed')
                if json.loads(execution_receipt.read_text())['amendment_sha256'] != original.sha(root/'execution-amendment.json'):
                    raise ValueError('Completed cell used a different execution amendment')
            else:
                info = original.SERVICE.stat()
                offset, identity = info.st_size,(info.st_dev,info.st_ino)
                original.write_new({'start_offset':offset,'service_log_identity':list(identity),
                                    'model':str(model),'seed':config['seed'],'started_at':original.study.now(),
                                    'execution_amendment_sha256':original.sha(root/'execution-amendment.json')},
                                   root/f'cell-start-{name}.json')
                result = original.study.evaluate(cell_config,name,model,0)
                original.write_new(original.audit_cell(name,offset,identity),receipt)
                original.write_new({'amendment_sha256':original.sha(root/'execution-amendment.json'),
                                    'recovery_runner_sha256':original.sha(Path(__file__)),
                                    'actual_scheduler_budget':4096,'actual_gpu_memory_utilization':.80},execution_receipt)
            results[name] = result
            original.report(results)
        original.study.state(config,'round1_complete',results=str(root/'RESULTS.md'),
                             next_phase='Freeze finalists or a bounded next development round; objective remains active')


if __name__=='__main__':
    main()
