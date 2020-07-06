# https://wikidocs.net/53560


#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#%%

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

print(x_train)
print(y_train)

#%%
# 가중치 W를 0으로 초기화, 
W = torch.zeros(1, requires_grad=True)
print(W)


#%%
b = torch.zeros(1, requires_grad=True)
print(b)
# %%

optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    #제곱 오차 계산식
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

# %%
# x_train 1,2,3 이고 y_train 2,4,6이므로
# W가 2에 가까워지고 b가 0이 되면 h(x) = 2x가 되므로 정답이 되어진다.

#%%

#optimizer zero_grad가 필요한 이유는
# pytorch에서는 미분을 통해 얻은 기울기를 이전에 게산된
# 기울기 값에 누적시키는 특징이 있다.
w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

  z = 2*w

  z.backward()
  print('수식을 w로 미분한 값 : {}'.format(w.grad))


# %%
#Random Seed를 할 경우 다음과 같이 고정이 되는걸 볼수있다.
torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1,3):
  print(torch.rand(1))

torch.manual_seed(5)
print('랜덤 시드가 5일 때')
for i in range(1,3):
  print(torch.rand(1))

torch.manual_seed(3)
print('랜덤 시드가 다시 3일 때')
for i in range(1,3):
  print(torch.rand(1))

