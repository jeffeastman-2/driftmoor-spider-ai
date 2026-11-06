"""Small action-conditioned neural predictor; no V-JEPA weights are used."""
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, inputs=44, observations=17):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs,96), nn.ReLU(),
            nn.Linear(96,96), nn.ReLU(), nn.Linear(96,observations+1),
        )

    def forward(self, x):
        return self.layers(x)


class Predictor:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, rows):
        with torch.no_grad():
            output = self.model(torch.tensor(rows,dtype=torch.float32))
            output[:,-1] = output[:,-1].sigmoid()
            return output.tolist()


def load_predictor(path):
    torch.set_num_threads(1)
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    model = Network(checkpoint["inputs"],checkpoint["observations"])
    model.load_state_dict(checkpoint["state_dict"])
    return Predictor(model)
