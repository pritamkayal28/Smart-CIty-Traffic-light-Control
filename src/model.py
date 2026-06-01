import torch.nn as nn
from src.config import PHASES

class DQN(nn.Module):
    """
    Deep Q-Network for Traffic Light Control.
    Input size: 4 (queue lengths in N, S, E, W directions)
    Hidden layers: 64, 64 nodes with ReLU activation
    Output size: 6 (number of valid traffic light signal phases)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(PHASES))
        )

    def forward(self, x):
        return self.net(x)
