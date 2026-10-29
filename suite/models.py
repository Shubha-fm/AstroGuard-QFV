"""Trainable controls. Quantum and classical residual maps each have 24 parameters.

All Transformer variants share the same embedding and head. Full networks are NOT
parameter matched exactly, even when the residual maps are equal in parameter count.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class TwoLayer24(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(4,3,bias=False),nn.Tanh(),nn.Linear(3,4,bias=False),nn.Tanh())
    def forward(self,x):return self.net(x)

class AffineGated24(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(4,4,bias=True);self.gains=nn.Parameter(torch.zeros(4))
    def forward(self,x):return torch.tanh(self.linear(x))*torch.exp(self.gains)

class Quantum24(nn.Module):
    def __init__(self):
        super().__init__()
        try:import pennylane as qml
        except ImportError as exc:raise RuntimeError('Quantum24 needs pip install -r requirements-quantum.txt') from exc
        dev=qml.device('default.qubit',wires=4)
        @qml.qnode(dev,interface='torch',diff_method='backprop')
        def circuit(inputs,weights):
            qml.AngleEmbedding(inputs,wires=range(4),rotation='Y')
            qml.StronglyEntanglingLayers(weights,wires=range(4))
            return [qml.expval(qml.PauliZ(w)) for w in range(4)]
        self.qlayer=qml.qnn.TorchLayer(circuit,{'weights':(2,4,3)})
    def forward(self,x):return self.qlayer(x).float()

class Positional(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        position=torch.arange(128,dtype=torch.float32).unsqueeze(1)
        factor=torch.exp(torch.arange(0,hidden,2,dtype=torch.float32)*(-math.log(10000.0)/hidden))
        pe=torch.zeros(1,128,hidden);pe[:,:,0::2]=torch.sin(position*factor)
        if hidden%2:pe[:,:,1::2]=torch.cos(position*factor[:-1])
        else:pe[:,:,1::2]=torch.cos(position*factor)
        self.register_buffer('pe',pe,persistent=False)
    def forward(self,x):return x+self.pe[:,:x.size(1)]

class Classifier(nn.Module):
    def __init__(self,variant,hidden=64,layers=2,heads=4):
        super().__init__();self.variant=variant
        self.input=nn.Linear(6,hidden)
        if variant=='bigru':
            self.encoder=nn.GRU(hidden,hidden//2,num_layers=max(1,layers),batch_first=True,bidirectional=True)
        elif variant=='cnn':
            self.encoder=nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=2),nn.GELU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.GELU())
        else:
            encoder=nn.TransformerEncoderLayer(hidden,heads,hidden*2,batch_first=True,activation='gelu',dropout=0.1,norm_first=True)
            self.encoder=nn.TransformerEncoder(encoder,layers,norm=nn.LayerNorm(hidden));self.position=Positional(hidden)
        self.quantum_input=None;self.residual=None
        if variant in ('classical24a','classical24b','quantum24'):
            self.quantum_input=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,4),nn.Tanh())
            self.residual={'classical24a':TwoLayer24,'classical24b':AffineGated24,'quantum24':Quantum24}[variant]()
            assert sum(p.numel() for p in self.residual.parameters() if p.requires_grad)==24
        fused=hidden+(4 if self.residual is not None else 0)
        self.head=nn.Sequential(nn.LayerNorm(fused),nn.Linear(fused,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,3))
        self.reconstruction=nn.Linear(hidden,6)
        self.projection=nn.Sequential(nn.Linear(hidden,64),nn.GELU(),nn.Linear(64,32))
    def forward(self,x,length,auxiliary=False):
        x=self.input(x);length=length.long().clamp(1,x.shape[1]);idx=torch.arange(x.shape[1],device=x.device)[None,:];valid=idx<length[:,None]
        if self.variant=='bigru':
            from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
            p=pack_padded_sequence(x,length.detach().cpu(),batch_first=True,enforce_sorted=False)
            h,_=self.encoder(p);h,_=pad_packed_sequence(h,batch_first=True,total_length=x.size(1))
        elif self.variant=='cnn':h=self.encoder(x.transpose(1,2)).transpose(1,2)
        else:h=self.encoder(self.position(x),src_key_padding_mask=~valid)
        pooled=(h*valid.unsqueeze(-1)).sum(1)/length.unsqueeze(1).clamp(min=1)
        base_pooled=pooled
        if self.residual is not None:
            z=self.quantum_input(pooled)*math.pi
            pooled=torch.cat([pooled,self.residual(z)],dim=-1)
        logits=self.head(pooled)
        if auxiliary:return logits,self.reconstruction(h),F.normalize(self.projection(base_pooled),dim=-1)
        return logits

def parameter_count(model):return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
