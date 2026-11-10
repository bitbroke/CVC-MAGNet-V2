import unittest
import torch
from run_constrained_recall import constraint_losses

class ConstraintTests(unittest.TestCase):
    def test_reference_detached_and_padding_ignored(self):
        logits=torch.zeros(1,1,16,16,requires_grad=True)
        reference=torch.full_like(logits,4,requires_grad=True)
        target=torch.zeros_like(logits);target[:,:,8,8]=1
        valid=torch.ones_like(logits);valid[:,:,:2,:]=0
        fp,ret=constraint_losses(logits,reference,target,valid)
        (fp+ret).backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(logits.grad[:,:,:2,:].abs().sum().item(),0)
        self.assertLess(logits.grad[0,0,8,8].item(),0)
    def test_empty_foreground_has_no_retention(self):
        x=torch.zeros(1,1,16,16,requires_grad=True)
        fp,ret=constraint_losses(x,x.detach(),torch.zeros_like(x),torch.ones_like(x))
        self.assertEqual(ret.item(),0)
        self.assertAlmostEqual(fp.item(),.5)
        (fp+ret).backward();self.assertTrue(torch.isfinite(x.grad).all())
    def test_no_valid_pixels_zero_gradient(self):
        x=torch.zeros(1,1,16,16,requires_grad=True)
        fp,ret=constraint_losses(x,x.detach(),torch.zeros_like(x),torch.zeros_like(x))
        (fp+ret).backward();self.assertEqual(x.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
