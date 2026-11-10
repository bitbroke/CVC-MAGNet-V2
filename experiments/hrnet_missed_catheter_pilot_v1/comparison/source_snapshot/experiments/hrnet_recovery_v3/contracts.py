"""Explicit supervision and offset contracts for future runs; no inferred attestations."""
import math
import numpy as np

def study_tip_supervision(instances, inventory_ids, confirmed_negative=False, allow_weak=False):
    """Require exact instance accounting; weak training needs an explicit separate arm."""
    ids=[str(r['catheter_instance_id']) for r in instances]
    expected={str(i) for i in inventory_ids}
    if confirmed_negative:
        if expected or instances: raise ValueError('Negative label conflicts with CVC inventory')
        return {'enabled':True,'source':'label_supported_negative','targets':[]}
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate instance IDs')
    if set(ids)!=expected or not expected:
        return {'enabled':False,'source':'incomplete_inventory','targets':[]}
    accepted={'provisional_engineering_review_v3','clinically_attested'}
    if allow_weak: accepted.add('weak_heuristic_unreviewed')
    targets=[]
    for r in instances:
        if r['tip_state']!='Assessable distal tip' or r['review_status'] not in accepted:
            return {'enabled':False,'source':'unknown_or_unreviewed','targets':[]}
        xy=(float(r['x']),float(r['y']))
        if not all(math.isfinite(v) for v in xy): raise ValueError('Nonfinite positive coordinate')
        targets.append(xy)
    return {'enabled':True,'source':'weak_enabled' if allow_weak else 'provisional_or_attested','targets':targets}

def exact_offset_target(hx,hy,width,height):
    """Use continuous GT coordinates, not a quadratic approximation to its Gaussian."""
    if not all(math.isfinite(x) for x in [hx,hy]): raise ValueError('Nonfinite coordinate')
    gx=int(math.floor(hx+0.5));gy=int(math.floor(hy+0.5))
    if not(0<=gx<width and 0<=gy<height): return None
    return gx,gy,float(hx-gx),float(hy-gy)

def encode_offsets(points,width,height):
    offsets=np.zeros((2,height,width),np.float32)
    weights=np.zeros((height,width),np.float32);counts={}
    for x,y in points:
        rec=exact_offset_target(x,y,width,height)
        if rec is None: continue
        gx,gy,dx,dy=rec;counts[(gx,gy)]=counts.get((gx,gy),0)+1
        offsets[:,gy,gx]=(dx,dy);weights[gy,gx]=1
    for (gx,gy),n in counts.items():
        if n>1: weights[gy,gx]=0;offsets[:,gy,gx]=0
    return offsets,weights
