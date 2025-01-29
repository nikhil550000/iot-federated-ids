import flwr as fl
import torch
from src.federated.utils import IDSModel, get_parameters, set_parameters
from src.data.dataloader import get_test_loader
from src.config import Config

cfg = Config()

def get_evaluate_fn():
    # Load test data
    test_loader = get_test_loader(cfg.batch_size)
    model = IDSModel()

    def evaluate_fn(server_round, parameters, config):
        set_parameters(model, parameters)
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in test_loader:
                preds = model(X)
                loss += torch.nn.functional.binary_cross_entropy(preds, y).item()
                correct += ((preds > 0.5) == y).sum().item()
        accuracy = correct / len(test_loader.dataset)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate_fn

def start_server():
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=cfg.num_clients,
        evaluate_fn=get_evaluate_fn(),
    )
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )