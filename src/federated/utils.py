import torch.nn as nn

class IDSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x).squeeze()

def get_parameters(model):
    return [param.detach().cpu().numpy() for param in model.parameters()]

def set_parameters(model, parameters):
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), parameters):
            param.copy_(torch.tensor(new_param, dtype=torch.float32))