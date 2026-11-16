"""Independent float64 formula probes and full shard hashes for round-1 exports.

This file does not import the candidate operators or their calibration routines.
It validates the stored calibration coefficients rather than reoptimizing them.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.checkpoint_merge_14b.audit_export_values import Model, require, sha


def reference(values, c, historical):
    a = values[-1]
    x = values-a
    cap = 2*np.abs(np.asarray(historical)@x)
    rollback = np.zeros_like(a)
    positive = np.zeros_like(a)
    mass = 0.
    for j, delta in enumerate(np.diff(values, axis=0)):
        confidence = c['negative_interval_confidence'][j]
        if confidence > 0 and mass > 0:
            b = positive/mass
            overlap = 2*np.minimum(np.abs(delta),np.abs(b))/np.maximum(np.abs(delta)+np.abs(b),1e-30)
            rollback += confidence*overlap*(delta*b<0)*delta
        if c['quality_differences'][j] > 0:
            positive += c['quality_differences'][j]*delta
            mass += c['quality_differences'][j]
    qis = np.clip(-rollback,-cap,cap)
    slope = np.asarray(c['quality_residual'])@x/c['quality_residual_norm2']
    total = ((np.asarray(c['residual_projection'])@x)**2).sum(0)
    reliability = np.clip(slope**2*c['quality_residual_norm2']/np.maximum(total,1e-30),0,1)
    drr = reliability*np.clip(slope*c['quality_target_gain'],-cap,cap)
    u = np.asarray(c['portfolio_weights'])@x
    variance = np.maximum(np.einsum('ik,ij,jk->k',x,np.asarray(c['portfolio_covariance']),x),0)
    results = {'bcm_linear':a+u}
    for strength, suffix in [(.5,'05'),(1.,'10')]:
        results['qis_'+suffix] = a+strength*qis
        results['drr_'+suffix] = a+strength*drr
        results['bcm_'+suffix] = a+u**3/np.maximum(u**2+strength*variance,1e-30)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get(
        'BCM_RUN_ROOT', Path(__file__).resolve().parents[2].parent/'runs/temporal-search14b-r1')))
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = json.loads((root/'inputs.json').read_text())
    recipes = json.loads((root/'recipes.json').read_text())
    output = root/'independent-export-audit.json'
    require(not output.exists(), 'Refusing to overwrite an existing audit')
    for file, expected in frozen['code_sha256'].items():
        require(sha(Path(file)) == expected, f'Frozen code changed: {file}')
    require(sha(root/'recipes.json') == frozen['recipes_sha256'], 'Frozen recipes changed')
    models = [Model(Path(frozen['config']['pool'])/f'checkpoint-{s}') for s in recipes['steps']]
    arms = {name:Model(root/'models'/name) for name in recipes['exports']}
    shape = {k:v['shape'] for k,v in models[-1].specs.items()}
    require(len(shape)==443, 'Expected 443 Qwen3 tensors')
    integrity = {}
    for name, model in arms.items():
        require({k:v['shape'] for k,v in model.specs.items()}==shape, 'Wrong export tensor coordinates')
        require(all(v['dtype']=='F32' for v in model.specs.values()), 'Export is not FP32')
        manifest = json.loads((model.path/'EXPORT_READY.json').read_text())['weights']
        require({x['file'] for x in manifest} == set(model.index['weight_map'].values()), 'Wrong shard inventory')
        for entry in manifest:
            p = model.path/entry['file']
            require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],f'Shard hash mismatch: {p}')
        integrity[name] = {'shards':len(manifest), 'tensor_count':len(shape), 'all_full_shard_hashes_verified':True}
        print('HASHES',name,'verified',flush=True)
    rng = np.random.default_rng(20261110)
    probes = {n:{'coordinates':0,'max_fraction_of_tolerance':0.} for n in arms}
    for key in sorted(shape):
        count = models[-1].specs[key]['count']
        ix = np.unique(np.r_[0,count//2,count-1,rng.integers(count,size=29)])
        values = np.stack([m.samples(key,ix) for m in models])
        expected = reference(values,recipes['calibration'],recipes['historical_weights'])
        for name, model in arms.items():
            actual = model.samples(key,ix)
            fraction = abs(actual-expected[name])/(5e-7+2e-6*abs(expected[name]))
            require(np.isfinite(actual).all() and np.all(fraction<=1),f'Formula mismatch {name}:{key}, {fraction.max()}')
            probes[name]['coordinates'] += len(ix)
            probes[name]['max_fraction_of_tolerance'] = max(probes[name]['max_fraction_of_tolerance'],float(fraction.max()))
    result = {'checked_at':datetime.now(timezone.utc).isoformat(), 'inputs_sha256':sha(root/'inputs.json'),
              'recipes_sha256':sha(root/'recipes.json'), 'audit_code_sha256':sha(Path(__file__)),
              'integrity':integrity,'probes':probes,
              'scope':'Every full output shard hash; float64 formula probes across every tensor. Not a full independent formula/finite scan or source-weight content hash.'}
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print('All seven independent formula audits passed',flush=True)


if __name__=='__main__':
    main()
