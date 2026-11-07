"""Small numeric encoders, optional recurrent memory, and supervised prediction heads."""
import torch
from torch import nn
import torch.nn.functional as F

INPUT_SIZE=41
ACTION_SIZE=8

class Agent(nn.Module):
    def __init__(self, kind="gru"):
        super().__init__()
        if kind not in ("gru","mlp"):
            raise ValueError(kind)
        self.kind=kind
        self.encoder=nn.Sequential(nn.Linear(INPUT_SIZE,32),nn.ReLU())
        self.memory=nn.GRU(32,48,batch_first=True) if kind=="gru" else nn.Sequential(nn.Linear(32,48),nn.Tanh())
        self.policy=nn.Linear(48,ACTION_SIZE)
        self.belief=nn.Linear(48,5)
        self.predictor=nn.Sequential(nn.Linear(48+ACTION_SIZE,64),nn.ReLU(),nn.Linear(64,43))

    def forward(self, observations, actions=None, hidden=None):
        encoded=self.encoder(observations)
        if self.kind=="gru":
            state,hidden=self.memory(encoded,hidden)
        else:
            state=self.memory(encoded)
            hidden=None
        belief=self.belief(state)
        belief=torch.cat((belief[...,:1].sigmoid(),belief[...,1:]),dim=-1)
        prediction=None
        if actions is not None:
            prediction=self.predictor(torch.cat((state,F.one_hot(actions,8).float()),dim=-1))
        return self.policy(state),belief,prediction,hidden

    def session(self):
        return Session(self)

class Session:
    def __init__(self,agent):
        self.agent=agent
        self.hidden=None

    @torch.no_grad()
    def choose(self,values):
        logits,belief,_,self.hidden=self.agent(torch.tensor(values,dtype=torch.float32)[None,None],hidden=self.hidden)
        return int(logits[0,0].argmax()),belief[0,0].tolist()

def save(agent,path):
    torch.save({"kind":agent.kind,"state":agent.state_dict()},path)

def load(path):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    agent=Agent(payload["kind"])
    agent.load_state_dict(payload["state"])
    return agent.eval()
