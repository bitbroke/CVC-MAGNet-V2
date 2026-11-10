import unittest
import torch
from run_recall_pilot import recall_loss

class RecallLossTests(unittest.TestCase):
    def test_instances_have_equal_weight(self):
        logits=torch.zeros(1,1,4,4,requires_grad=True)
        loss=recall_loss(logits,[torch.tensor([0]),torch.tensor([1,2,3,4])])
        loss.backward()
        self.assertAlmostEqual(loss.item(),.5)
        self.assertAlmostEqual(logits.grad.flatten()[0].item(),4*logits.grad.flatten()[1].item())
        self.assertEqual(logits.grad.flatten()[5:].abs().sum().item(),0)
    def test_negative_has_no_extra_recall_gradient(self):
        logits=torch.zeros(1,1,4,4,requires_grad=True)
        recall_loss(logits,[]).backward()
        self.assertEqual(logits.grad.abs().sum().item(),0)
    def test_missed_instance_has_higher_loss(self):
        indices=[torch.tensor([0,1]),torch.tensor([2,3])]
        good=torch.tensor([8.,8.,8.,8.]);missed=torch.tensor([8.,8.,-8.,-8.])
        self.assertGreater(recall_loss(missed,indices).item(),recall_loss(good,indices).item())
if __name__=='__main__':unittest.main()
