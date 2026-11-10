import numpy as np
import torch
from torch import nn
from experiments.hrnet_recovery_v3.run_feature_pilot import path_stats,configure,SplitInference

def test_gap_metrics_complete_and_absent():
    poly=[[10.,30.],[90.,30.]];p=np.zeros((100,100),np.float32)
    a=path_stats(p,poly)
    assert a['coverage10']==0 and a['missed'] and a['longest_gap_native_px']==80
    p[30,10:91]=1
    b=path_stats(p,poly)
    assert b['coverage10']==1 and b['coverage_complete'] and b['longest_gap_native_px']==0

def test_tip_adapter_ignores_adapted_tip():
    class Fake(nn.Module):
        def __init__(self,value):super().__init__();self.value=value
        def forward(self,x):return dict(seg_logits=x+self.value,tip_logits=x*self.value,tip_probs=x*self.value)
    x=torch.ones(1,1,2,2);out=SplitInference(Fake(float('nan')),Fake(2))(x)
    assert torch.equal(out['tip_logits'],x*2)

def test_scope_and_bn_buffers():
    m=nn.Module();m.encoder=nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2));m.tip_head=nn.Linear(2,1);m.seg_head=nn.Linear(2,1)
    configure(m,False)
    assert all(p.requires_grad==n.startswith('seg_head.') for n,p in m.named_parameters())
    configure(m,True)
    assert all(p.requires_grad==(not n.startswith('tip_head.')) for n,p in m.named_parameters())
    assert not m.encoder[1].training
