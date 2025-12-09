import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_tasks, emb_dim=2048, drop_ratio=0.5):
        super(MLP, self).__init__()
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.drop_ratio = drop_ratio

        self.fc1 = nn.Linear(self.emb_dim, 1200)
        self.bn1 = nn.BatchNorm1d(1200)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(self.drop_ratio)
        self.fc2 = nn.Linear(1200, self.num_tasks)
        self.drop2 = nn.Dropout(self.drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
