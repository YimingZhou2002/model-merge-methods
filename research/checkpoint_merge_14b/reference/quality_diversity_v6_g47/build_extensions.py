"""CPU-only P2 recovery and preregistered coordinate trajectory gates."""
from contextlib import ExitStack
import json
import math
from pathlib import Path

import torch

from common import WORK, V6, source_set, norm2, export_models, wm, disk_guard, now, write_json, original_build

STEPS=(240,250,260,270,280,300)
TSG_NAMES=('TSG_TV','TSG_REV','TSG_QA')


def trajectory_gates(checkpoints,quality,update):
    """Returns gates in [0,1]; checkpoint index is chronological, never unordered."""
    first=checkpoints[0]
    travel=torch.zeros_like(first)
    for left,right in zip(checkpoints,checkpoints[1:]): travel.add_((right-left).abs())
    tv=torch.where(travel>0,1-(checkpoints[-1]-first).abs()/travel.clamp_min(1e-30),0.).clamp_(0,1)
    early=checkpoints[-2]-first;late=checkpoints[-1]-checkpoints[-2]
    opposed=((early>0)&(late<0))|((early<0)&(late>0))
    rollback=((update>0)&(late<0))|((update<0)&(late>0))
    reverse=2*torch.minimum(early.abs(),late.abs())/(early.abs()+late.abs()).clamp_min(1e-30)
    reverse=reverse.clamp_(0,1)*opposed*rollback
    q=torch.as_tensor(quality,dtype=torch.float64);q=q-q.mean()
    # Center relative to ck240 before accumulating: avoid cancellation of large base weights.
    mean=torch.zeros_like(first,dtype=torch.float64)
    for model in checkpoints: mean.add_((model-first).double()/len(checkpoints))
    cov=torch.zeros_like(mean);var=torch.zeros_like(mean)
    for model,weight in zip(checkpoints,q):
        centered=(model-first).double()-mean
        cov.add_(centered*weight);var.add_(centered.square())
    denominator=(var*q.square().sum()).sqrt()
    correlation=torch.where(denominator>0,cov/denominator.clamp_min(1e-30),0.).clamp_(-1,1).float()
    qa=tv*(1+update.sign()*correlation)/2
    return (tv,reverse,qa.clamp_(0,1))


def build_p2(p):
    disk_guard(115)
    members=sorted(p['bank'],key=int);weights=[p['bank'][s] for s in members]
    paths=[p['reused_models']['P1_linear']['path'],p['ties_model']['path'],p['source_models']['300']['path'],
           *[p['source_models'][s]['path'] for s in members]]
    sources=source_set(paths);stats={'d2':0.,'gated2':0.,'cross':0.,'endpoint2':0.,'layers':{}}
    with ExitStack() as stack:
        for s in sources:s.attach(stack)
        for key in sorted(sources[0].specs):
            tensors=[s.read(key,None) for s in sources]
            if any(not torch.isfinite(t).all() or t.dtype!=torch.float32 for t in tensors): raise ValueError('Invalid source tensor')
            g,d=original_build.gate_terms(*tensors[:3],tensors[3:],weights)
            row={'d2':norm2(d),'gated2':norm2(g*d),'cross':float((g.double()*d.double().square()).sum()),
                 'endpoint2':norm2((1-g)*d),'linear_anchor2':norm2(tensors[0]-tensors[2]),
                 'ties_anchor2':norm2(tensors[1]-tensors[2])}
            stats['layers'][key]=row
            for field in ('d2','gated2','cross','endpoint2'):stats[field]+=row[field]
    eta=math.sqrt(stats['gated2']/stats['d2']) if stats['d2'] else 0.
    contrast=max(0.,stats['gated2']+eta*eta*stats['d2']-2*eta*stats['cross'])
    stats.update(eta=eta,contrast2=contrast)
    if not stats['d2'] or min(contrast,stats['gated2'],stats['endpoint2'])<=1e-8*stats['d2']:
        stats.update(state='skipped_duplicate',jobs=[],reason='Frozen V6 endpoint/global near-duplicate gate')
    else:
        stats.update(gate_anchor2=0.,global_anchor2=0.)
        def transform(l,t,a,*donors):
            g,d=original_build.gate_terms(l,t,a,donors,weights)
            output=[l+g*d,l+eta*d]
            stats['gate_anchor2']+=norm2(output[0]-a);stats['global_anchor2']+=norm2(output[1]-a)
            return output
        export_models(['P2_gate','P2_global'],paths,transform,
                      {'proposal':'P2','norm_matched_pair':True,'norm_reference':'P1_linear','eta':eta},WORK/'models')
        stats.update(state='complete',jobs=['P2_gate','P2_global'])
    write_json(WORK/'P2_diagnostics.json',stats)


def build_tsg(p):
    disk_guard(300)
    paths=[p['source_models']['300']['path'],p['reused_models']['P1_linear']['path'],
           *[p['source_models'][str(s)]['path'] for s in STEPS]]
    quality=[p['TSG']['quality'][str(s)] for s in STEPS]
    sources=source_set(paths);raw=0.;gram=torch.zeros((4,4),dtype=torch.float64);layers={}
    gate_counts={name:{'coordinates':0,'gate_below_01':0,'gate_above_09':0} for name in TSG_NAMES}
    with ExitStack() as stack:
        for s in sources:s.attach(stack)
        for key in sorted(sources[0].specs):
            tensors=[s.read(key,None) for s in sources]
            if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in tensors): raise ValueError('Invalid source tensor')
            a,l,*checkpoints=tensors;u=l-a;gates=trajectory_gates(checkpoints,quality,u)
            updates=[u,*[g*u for g in gates]]
            row={'raw2':norm2(u),'gated2':[norm2(v) for v in updates[1:]]};layers[key]=row;raw+=row['raw2']
            for i in range(4):
                for j in range(i,4):
                    dot=float(torch.dot(updates[i].reshape(-1).double(),updates[j].reshape(-1).double()))
                    gram[i,j]+=dot
                    if i!=j:gram[j,i]+=dot
            for name,g in zip(TSG_NAMES,gates):
                gate_counts[name]['coordinates']+=g.numel()
                gate_counts[name]['gate_below_01']+=int((g<.1).sum())
                gate_counts[name]['gate_above_09']+=int((g>.9).sum())
    if raw<=0: raise ValueError('Same-bank linear equals anchor')
    coefficients={}
    for i,name in enumerate(TSG_NAMES,1):
        coefficients[name]=torch.eye(4,dtype=torch.float64)[i]
        coefficients[name+'_norm']=torch.tensor([math.sqrt(max(0.,float(gram[i,i]))/raw),0.,0.,0.],dtype=torch.float64)
    scale=math.sqrt(float(gram[3,3]/gram[1,1])) if gram[1,1]>0 else 0.
    coefficients['TSG_QA_tvnorm']=torch.tensor([0.,scale,0.,0.],dtype=torch.float64)
    # Deduplicate before exports, using only parameter distances, never reward outcomes.
    bases={'P1_anchor300':torch.zeros(4,dtype=torch.float64),'P1_linear':torch.tensor([1.,0.,0.,0.],dtype=torch.float64)}
    aliases={};unique=[]
    for name,c in coefficients.items():
        equivalent=None
        for existing,other in bases.items():
            difference=c-other
            if max(0.,float(difference@gram@difference))<=1e-8*raw:
                equivalent=existing;break
        if equivalent is not None:aliases[name]=equivalent
        else:unique.append(name);bases[name]=c
    metadata={'proposal':'TSG','steps':list(STEPS),'quality':quality,
              'coefficients':{n:c.tolist() for n,c in coefficients.items()},'aliases':aliases,
              'norm_reference':'checkpoint300','quality_is_causal':False}
    if unique:
        def transform(a,l,*checkpoints):
            u=l-a;gates=trajectory_gates(checkpoints,quality,u);updates=[u,*[g*u for g in gates]]
            outputs=[]
            for name in unique:
                delta=torch.zeros_like(a)
                for weight,update in zip(coefficients[name],updates):
                    if weight:delta.add_(update,alpha=float(weight))
                outputs.append(a+delta)
            return outputs
        export_models(unique,paths,transform,metadata,WORK/'models')
    actual={'P1_anchor300':0.,'P1_linear':math.sqrt(raw)}
    for name in unique:actual[name]=json.loads((WORK/'models'/name/'merge_audit.json').read_text())['actual_update_norm']
    for name,target in aliases.items():actual[name]=actual[target]
    pairs=[(n,n+'_norm') for n in TSG_NAMES]+[('TSG_QA','TSG_QA_tvnorm')]
    for a,b in pairs:
        if abs(actual[a]-actual[b])>.002*max(actual[a],actual[b])+1e-6:raise ValueError('Actual TSG norm matching failed')
    write_json(WORK/'TSG_diagnostics.json',{'state':'complete','jobs':unique,'aliases':aliases,
        'raw_update_norm2':raw,'gram':gram.tolist(),'coefficients':metadata['coefficients'],
        'actual_norms':actual,'gate_counts':gate_counts,'layers':layers,'QA_tv_scale':scale})


if __name__=='__main__':
    from queue_g47 import verify
    p=json.loads((WORK/'protocol.json').read_text());verify(p);torch.set_num_threads(8)
    for name,build in [('P2',build_p2),('TSG',build_tsg)]:
        path=WORK/(name+'_build_status.json')
        try:
            write_json(path,{'state':'building','updated_utc':now()});build(p)
            write_json(path,{'state':'complete','updated_utc':now()})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            write_json(path,{'state':'failed','reason':str(exc),'updated_utc':now()})
            write_json(WORK/(name+'_ALERT.json'),{'stage':name,'reason':str(exc),'updated_utc':now()})
            # An independent construction failure does not suppress available P1 or P3 arms.
