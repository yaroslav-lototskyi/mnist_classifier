import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(BaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
