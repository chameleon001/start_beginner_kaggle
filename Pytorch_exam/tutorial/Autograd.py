
#%%

# https://wikidocs.net/60754

# AutoGrad

#%%

import torch

#%%

w = torch.tensor(2.0, requires_grad=True)

y = w**2 
z = 2*y + 5 # 2*(w**2)

#%%
z.backward()

print('수식을 w로 미분한 값 : {}'.format(w.grad))


# %%
