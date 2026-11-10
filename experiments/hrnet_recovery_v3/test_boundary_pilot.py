import torch
from experiments.hrnet_recovery_v3.run_boundary_pilot import ring_mask,precision_loss,retention_loss

def test_ring_excludes_gt_tolerance_padding_and_far_background():
    y=torch.zeros(1,32,32);y[:,16,16]=1;v=torch.ones_like(y);v[:,:,:5]=0
    r=ring_mask(y,v)
    assert r[0,16,16]==0 and r[0,16,17]==0 and r[0,16,18]==1
    assert r[0,16,25]==0 and r[:,:,:5].sum()==0
    assert ring_mask(torch.zeros_like(y),v).sum()==0

def test_boundary_gradients_confined_to_ring():
    z=torch.zeros(1,1,32,32,requires_grad=True);r=torch.zeros_like(z);r[:,:,5:8,5:8]=1
    precision_loss(z,r).backward()
    assert (z.grad[r==1]>0).all() and (z.grad[r==0]==0).all()
    z.grad=None;precision_loss(z,torch.zeros_like(r)).backward();assert (z.grad==0).all()

def test_retention_protects_only_observed_teacher_foreground():
    z=torch.full((1,1,2,2),-2.,requires_grad=True);teacher=torch.full_like(z,2.,requires_grad=True)
    y=torch.zeros_like(z);y[:,:,0,0]=1;v=torch.ones_like(z)
    retention_loss(z,teacher,y,v).backward()
    assert z.grad[0,0,0,0]<0 and z.grad[0,0,1,1]==0 and teacher.grad is None

def test_fp32_boundary_reduction():
    z=torch.full((1,1,768,768),12.,dtype=torch.float16,requires_grad=True)
    loss=precision_loss(z,torch.ones_like(z));assert torch.isfinite(loss)
    loss.backward();assert torch.isfinite(z.grad).all() and z.grad.sum()>0
