"""Direct two-step outcome model: next observation plus two admission heads."""
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(44,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU(),nn.Linear(96,19),
        )

    def forward(self,x):
        return self.layers(x)


class Predictor:
    def __init__(self,model):
        self.model = model.eval()

    def __call__(self,rows):
        with torch.no_grad():
            out = self.model(torch.tensor(rows,dtype=torch.float32))
            out[:,-2:] = out[:,-2:].sigmoid()
            return out.tolist()


def load_predictor(path):
    torch.set_num_threads(1)
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["version"] != 2:
        raise ValueError("Expected v2 checkpoint")
    model = Network()
    model.load_state_dict(checkpoint["state_dict"])
    return Predictor(model)
