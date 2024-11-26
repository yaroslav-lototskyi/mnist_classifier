import torch
from models.base_model import BaseModel
from training.utils import load_data
from training.evaluate import evaluate_model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = load_data()
    model = BaseModel()
    model.load_state_dict(torch.load("model.pth"))
    evaluate_model(model, test_loader, device)
