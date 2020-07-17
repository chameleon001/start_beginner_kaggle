
# https://wikidocs.net/60572

#%% 

import torch
import torch.nn.functional as F

torch.manual_seed(1)

# %%

z = torch.FloatTensor([1,2,3])

hypothesis = F.softmax(z, dim=0)
print(hypothesis)


# %%
z = torch.rand(3, 5, requires_grad=True)
print(z)

# %%
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

#%%
y = torch.randint(5, (3,)).long()
print(y)
# %%
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# %%
# print(y_unsqueeze(1))
# %%
print(y_one_hot)

# %%
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# %%

# low level


# %%

# %%
# Low level
torch.log(F.softmax(z, dim=1))

(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# %%
# high level1
F.log_softmax(z,dim=1)
(y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()

#%%
# high level2
F.log_softmax(z,dim=1)
F.nll_loss(F.log_softmax(z, dim=1), y)

#%%

# high level2
F.log_softmax(z,dim=1)
F.cross_entropy(z, y)