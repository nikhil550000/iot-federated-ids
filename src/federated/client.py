import torch
import flwr as fl
from src.federated.utils import IDSModel, get_parameters, set_parameters
from src.data.dataloader import get_client_loader
from src.config import Config

cfg = Config()

class PyTorchClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = IDSModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        self.loader = get_client_loader(client_id, cfg.batch_size)

    def fit(self, global_params, config):
        # Personalization: Blend global and local weights
        set_parameters(self.model, global_params)
        original_params = get_parameters(self.model)  # Backup

        # Local training
        self.model.train()
        for X, y in self.loader:
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()

        # Blend parameters
        updated_params = get_parameters(self.model)
        personalized_params = [
            cfg.alpha * g + (1 - cfg.alpha) * l
            for g, l in zip(global_params, updated_params)
        ]

        # Add DP noise
        if cfg.noise_scale > 0:
            personalized_params = [
                p + torch.randn(p.shape) * cfg.noise_scale
                for p in personalized_params
            ]

        return personalized_params, len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in self.loader:
                preds = self.model(X)
                loss += self.criterion(preds, y).item()
                correct += ((preds > 0.5) == y).sum().item()
        accuracy = correct / len(self.loader.dataset)
        return loss, len(self.loader.dataset), {"accuracy": accuracy}