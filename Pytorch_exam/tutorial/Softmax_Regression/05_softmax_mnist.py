
# https://wikidocs.net/60324
#

#%%

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random


# %%
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

print(device)

#%%
# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# %%
# hyperparameters
training_epochs = 15
batch_size = 100

# %%
# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# %%

# dataset = dataload�� ���
# batch_size = ��ġ ũ��
# shuffle = �� ����ũ���� �̴� ��ġ ������ ������
# drop_last = ������ ��ġ�� ����������
data_loader = DataLoader(dataset = mnist_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)
                        

# %%

linear = nn.Linear(784, 10, bias=True).to(device)

#%%

# mnist data image of shape 28*28 784
linear = nn.Linear(784, 10, bias=True).to(device)
#%%
criterion = nn.CrossEntropyLoss().to(device) # ���������� ����Ʈ�ƽ� �Լ��� �����ϰ� ����.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

# %%
for epoch in range(training_epochs): # �ռ� training_epochs�� ���� 15�� ������.
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # ��ġ ũ�Ⱑ 100�̹Ƿ� �Ʒ��� ���꿡�� X�� (100, 784)�� �ټ��� �ȴ�.
        X = X.view(-1, 28 * 28).to(device)
        # ���̺��� ��-�� ���ڵ��� �� ���°� �ƴ϶� 0 ~ 9�� ����.
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
# %%


# %%
