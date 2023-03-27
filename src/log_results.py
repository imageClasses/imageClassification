import torch
import json
import os

class Logger():
    "Logs and saves metadata and model"
    
    def __init__(self):
        self.result_dict = {
            "model_name": None,
            "model_str": None,
            "optimizer": None,
            "loss_function": None,
            "epoch_time": [],
            "train_loss": [], #Logged per epoch
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_loss": None,
            "test_accuracy": None,
            "early_stop": False,
            "best_epoch": None
        }
        
    def log(self, key, val):
        self.result_dict[key] = val
        
    def append(self, key, val):
        self.result_dict[key].append(val)
        
    def save_model(self, model):
        path = f'saved_models/{self.result_dict["model_name"]}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model.state_dict(), f'{path}/model_state.pt')
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(self.result_dict, f, indent=2)