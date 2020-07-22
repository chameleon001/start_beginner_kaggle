
# https://wikidocs.net/60848
#%%

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

if device == 'cuda':