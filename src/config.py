class Config:
    num_clients = 10
    num_rounds = 20
    batch_size = 32
    alpha = 0.7  # Personalization factor
    noise_scale = 0.1  # DP noise
    model_save_path = "/content/drive/MyDrive/iot_federated_ids/models/global_model.pt"
    test_data_path = "/content/drive/MyDrive/iot_federated_ids/data/processed/test.csv"