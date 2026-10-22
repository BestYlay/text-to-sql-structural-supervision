import unittest
import torch
from text2sql.losses import loss_sums,joint_loss,StructureHeads
from text2sql.batching import collate


class LossTests(unittest.TestCase):
    def test_sql_and_forward_kl_direction(self):
        student=torch.tensor([[2.,0.,-1.]],requires_grad=True)
        teacher=torch.log_softmax(torch.tensor([[0.,2.,-1.]]),-1)
        ce,kl=loss_sums(student,teacher,torch.tensor([1]))
        expected=(teacher.exp()*(teacher-torch.log_softmax(student,-1))).sum()
        self.assertTrue(torch.allclose(kl,expected))
        (ce+.1*kl).backward();self.assertIsNotNone(student.grad)

    def test_joint_token_and_sample_normalization(self):
        result=joint_loss(*[torch.tensor(x) for x in [12.,4.,2.,6.,3.]],tokens=4,samples=2)
        self.assertAlmostEqual(result.item(),3+.1+.1*(4+3)/2,places=5)

    def test_padding_and_auxiliary_gradient(self):
        batch=collate([dict(input_ids=[1,2,3],labels=[-100,2,3]),dict(input_ids=[1,2],labels=[-100,2])],0)
        self.assertEqual(batch['labels'][1].tolist(),[-100,2,-100])
        heads=StructureHeads(8,width=4);context=torch.randn(8,requires_grad=True)
        outputs=heads(context,torch.randn(2,8),torch.randn(3,8),torch.tensor([[0,1]]))
        sum(x.sum() for x in outputs).backward();self.assertIsNotNone(context.grad)


if __name__=='__main__':unittest.main()
