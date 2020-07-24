# https://wikidocs.net/61046

#%%
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

digits = load_digits()

print(digits.images[0])

# %%
print(digits.target[0])

# %%
print('전체 샘플의 수 : {}'.format(len(digits.images)))

# %%
images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:5]):
    plt.subplot(2,5, index+1)
    plt.axis('off')
    plt.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    plt.title('sample : %i' % label)

# %%

for i in range(5):
    print(i, '번 인덱스 샘플의 레이블 : ', digits.target[i])


# %%
print(digits.data[0])

# %%

X = digits.data
Y = digits.target

# %%

import torch
import torch.nn as nn
from torch import optim

# %%
model = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10)
)

# %%

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)


# %%
loss_fn = nn.CrossEntropyLoss()

# %%
optimizer = optim.Adam(model.parameters())
# %%
losses = []


# %%
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X) # forwar 연산
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))

    losses.append(loss.item())
# %%
plt.plot(losses)

# %%
