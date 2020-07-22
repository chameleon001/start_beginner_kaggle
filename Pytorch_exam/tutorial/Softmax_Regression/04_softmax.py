
# https://wikidocs.net/60575

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# %%
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# %%
print(x_train.shape)
print(y_train.shape)

# %%

y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape)

# %%
W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# %%
optimizer = optim.SGD([W,b], lr=0.1)

# %%

nb_epochs = 1000
for epoch in range(nb_epochs +1):
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

        # cost�? H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마?�� 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))


# %%

W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.1)

nb_epochs = 1000


# %%

for epoch in range(nb_epochs + 1):

    #cost ���
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    #cost�� h(x) ����

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 ==  0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
# �� ����� ���
# ���� ���� �� �ʱ�ȭ. 4���� Ư���� ������ 3���� Ŭ������ �з�. input_dim=4, output_dim=3.
model = nn.Linear(4, 3)

# %%

# optimizer ����
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) ���
    prediction = model(x_train)

    # cost ���
    cost = F.cross_entropy(prediction, y_train)

    # cost�� H(x) ����
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20������ �α� ���
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%

# Ŭ���� �̿�

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output�� 3!

    def forward(self, x):
        return self.linear(x)

# %%
model = SoftmaxClassifierModel()

# %%
# optimizer ����
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) ���
    prediction = model(x_train)

    # cost ���
    cost = F.cross_entropy(prediction, y_train)

    # cost�� H(x) ����
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20������ �α� ���
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
