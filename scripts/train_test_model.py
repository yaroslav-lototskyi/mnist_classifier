import torch
import time
from models.base_model import BaseModel
from training.utils import load_data
from training.train_evaluate import train_evaluate_model
from visualization.plot_training_test_results import plot_training_test_results

if __name__ == "__main__":
    start_time = time.time() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()
    model = BaseModel()
    train_losses, test_accuracies = train_evaluate_model(model, train_loader, test_loader, device)
    plot_training_test_results(train_losses, test_accuracies)
    torch.save(model.state_dict(), "model.pth")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.2f} seconds.")
