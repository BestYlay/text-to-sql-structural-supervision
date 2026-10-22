import math
import torch
from torch import nn
import torch.nn.functional as F

def balanced_bce(logits,labels):
    if not len(logits):return logits.sum()*0
    losses=F.binary_cross_entropy_with_logits(logits.float(),labels.float(),reduction='none')
    terms=[losses[mask].mean() for mask in [labels>0,labels==0] if bool(mask.any())]
    return torch.stack(terms).mean()

class StructureHeads(nn.Module):
    def __init__(self,hidden,width=128):
        super().__init__();self.width=width
        self.query=nn.ModuleList([nn.Linear(hidden,width) for _ in range(3)])
        self.keys=nn.ModuleList([nn.Linear(hidden,width),nn.Linear(hidden,width),nn.Linear(2*hidden,width)])
        self.bias=nn.Parameter(torch.zeros(3))
    def forward(self,context,tables,columns,edges):
        if len(edges):
            a=columns[edges[:,0]];b=columns[edges[:,1]];features=torch.cat([a+b,torch.abs(a-b)],-1)
        else:features=columns.new_empty((0,2*columns.shape[-1]))
        return [(self.keys[i](x)*self.query[i](context)).sum(-1)/math.sqrt(self.width)+self.bias[i]
                for i,x in enumerate([tables,columns,features])]

def loss_sums(student_logits,teacher_logp,targets,chunk=16):
    """Both outputs are sums across the same shifted assistant positions."""
    ce=student_logits.new_zeros((),dtype=torch.float32);kl=ce.clone()
    for start in range(0,len(targets),chunk):
        logits=student_logits[start:start+chunk].float()
        logq=F.log_softmax(logits,dim=-1)
        logp=teacher_logp[start:start+chunk].detach()
        ce=ce+F.nll_loss(logq,targets[start:start+chunk],reduction='sum')
        kl=kl+F.kl_div(logq,logp,log_target=True,reduction='sum')
    return ce,kl

def joint_loss(ce, kl, table, column, edge, tokens, samples, kl_weight=.1, structure_weight=.1):
    if tokens <= 0 or samples <= 0:
        raise ValueError('Empty accumulation window')
    loss = ce / tokens
    if kl_weight:
        loss = loss + kl_weight * kl / tokens
    if structure_weight:
        loss = loss + structure_weight * ((table + column) / 2 + edge) / samples
    return loss
