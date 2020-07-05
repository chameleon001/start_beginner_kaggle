# https://wikidocs.net/55409

#%%
import torch.nn as nn
import torch
import torch.nn.functional as F

torch.manual_seed(1)

# %%
x_train = torch.FloatTensor(([1],[2],[3]))
y_train = torch.FloatTensor(([2],[4],[6]))

#%%
model = nn.Linear(1,1)

# %%
print(list(model.parameters()))

# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 


# %%
nb_epochs = 2000

for epoch in range(nb_epochs + 1):

    prediction = model(x_train)

    #평균 제곱오차
    cost = F.mse_loss(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print("훈련 후 입력이 4 일때의 예측값 : ",pred_y)

# %%
print(list(model.parameters()))

# %%
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# %%
model = nn.Linear(3,1)

# %%
print(list(model.parameters()))

# %%
