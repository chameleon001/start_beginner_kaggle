
# https://wikidocs.net/60036

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
torch.manual_seed(1)

# %%
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

#%%
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)