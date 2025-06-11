import torch
import torch.nn as nn
import torch.nn.functional as F


class StrokeGRUDiffusion(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        output, _ = self.gru(x)
        out = self.fc(output)
        return out