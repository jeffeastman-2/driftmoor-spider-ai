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
        self.prediction_mode="absolute"
        self.encoder=nn.Sequential(nn.Linear(INPUT_SIZE,32),nn.ReLU())
        self.memory=nn.GRU(32,48,batch_first=True) if kind=="gru" else nn.Sequential(nn.Linear(32,48),nn.Tanh())
        self.policy=nn.Linear(48,ACTION_SIZE)
        self.belief=nn.Linear(48,5)
        self.predictor=nn.Sequential(nn.Linear(48+ACTION_SIZE,64),nn.ReLU(),nn.Linear(64,43))

    def representation(self, observations, hidden=None):
        """Perception and memory, updated once per real observation."""
        encoded=self.encoder(observations)
        if self.kind=="gru":
            state,hidden=self.memory(encoded,hidden)
        else:
            state=self.memory(encoded)
            hidden=None
        return state,hidden

    def predict_futures(self,state,actions,observations):
        prediction=self.predictor(torch.cat((state,F.one_hot(actions,8).float()),dim=-1))
        if self.prediction_mode=="hp_delta":
            prediction=torch.cat((prediction[...,:16],
                observations[...,16:17]+prediction[...,16:17],prediction[...,17:]),dim=-1)
        return prediction

    def forward(self, observations, actions=None, hidden=None):
        state,hidden=self.representation(observations,hidden)
        belief=self.belief(state)
        belief=torch.cat((belief[...,:1].sigmoid(),belief[...,1:]),dim=-1)
        prediction=None
        if actions is not None:
            prediction=self.predict_futures(state,actions,observations)
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

def save(agent,path,observation_schema="withdrawal-cornered-v6"):
    payload={"kind":agent.kind,"state":agent.state_dict(),"prediction_mode":agent.prediction_mode}
    if observation_schema is not None:
        payload["observation_schema"]=observation_schema
    torch.save(payload,path)

def load(path,*,expected_schema="withdrawal-cornered-v6"):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if expected_schema is not None and payload.get("observation_schema")!=expected_schema:
        raise ValueError("Checkpoint observation schema does not match this experiment")
    agent=Agent(payload["kind"])
    agent.prediction_mode=payload.get("prediction_mode","absolute")
    if agent.prediction_mode not in ("absolute","hp_delta"):
        raise ValueError("Unknown prediction mode")
    agent.load_state_dict(payload["state"])
    return agent.eval()
