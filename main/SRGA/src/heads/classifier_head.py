import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(x)