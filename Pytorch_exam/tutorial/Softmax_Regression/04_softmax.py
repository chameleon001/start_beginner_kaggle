
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

        # cost濡? H(x) 媛쒖꽑
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100踰덈쭏?떎 濡쒓렇 異쒕젰
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

    #cost 계산
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    #cost로 h(x) 개선

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 ==  0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
# 모델 사용한 방식
# 모델을 선언 및 초기화. 4개의 특성을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3.
model = nn.Linear(4, 3)

# %%

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%

# 클래스 이용

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3!

    def forward(self, x):
        return self.linear(x)

# %%
model = SoftmaxClassifierModel()

# %%
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
