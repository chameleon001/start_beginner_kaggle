
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


#%%
model = LinearRegressionModel()

#optimizer 설정, 경사하강법 SGD 사용
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000

#%%
for epoch in range(nb_epochs+1):

    #H(x) 계산
    prediction = model(x_train)

    #cost
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()

    cost.backward()

    #w와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))



# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#%%
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#%%
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
        # 다중 선형 회귀이므로 input= 3 ouput=1
    
    def forward(self, x):
        return self.linear(x)

#%%
model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000

for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일

    #cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()

    cost.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

# %%
