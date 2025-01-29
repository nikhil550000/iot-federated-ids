import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class IoTDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        self.X = torch.tensor(df["feature"].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_client_loader(client_id, batch_size=32):
    client_path = f"/content/drive/MyDrive/iot_federated_ids/data/clients/client_{client_id}.csv"
    dataset = IoTDataset(client_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size=32):
    test_path = "/content/drive/MyDrive/iot_federated_ids/data/processed/test.csv"
    dataset = IoTDataset(test_path)
    return DataLoader(dataset, batch_size=batch_size)