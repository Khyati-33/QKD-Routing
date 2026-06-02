import torch, torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU()
        )
        self.value    = nn.Sequential(nn.Linear(hidden//2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage= nn.Sequential(nn.Linear(hidden//2, 32), nn.ReLU(), nn.Linear(32, n_actions))

    def forward(self, x):
        h = self.shared(x)
        V = self.value(h)
        A = self.advantage(h)
        return V + (A - A.mean(dim=1, keepdim=True))   # Q = V + (A - mean(A))
