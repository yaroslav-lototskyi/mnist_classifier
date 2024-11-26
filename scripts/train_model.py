import torch
from models.base_model import BaseModel
from training.utils import load_data
from training.train import train_model
from visualization.plot_training_results import plot_training_results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = load_data()
    model = BaseModel()
    train_losses = train_model(model, train_loader, device)
    plot_training_results(train_losses)
    torch.save(model.state_dict(), "model.pth")
