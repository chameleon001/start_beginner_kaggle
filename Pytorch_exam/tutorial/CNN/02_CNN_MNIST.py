
# https://wikidocs.net/63565
#%%

import torch
import torch.nn as nn

inputs = torch.Tensor(1,1,28,28)
print('텐서의 크기 : {}'.format(inputs.shape))

# %%

conv1 = nn.Conv2d(1,32,3,padding=1)
print(conv1)

# %%
conv2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
print(conv2)

# %%
pool = nn.MaxPool2d(2)
print(pool)

# %%
out = conv1(inputs)
print(out.shape)
# %%
out = conv2(out)
print(out.shape)

# %%
out = pool(out)
print(out.shape)
# %%
out = out.view(out.size(0), -1) 
print(out.shape)

# %%
fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)

# %%

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init


# %%
#device ='cpu'
device ='cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

if device =='cuda':
    torch.cuda.manual_seed(777)
# %%
learning_rate = 0.001
training_epochs = 15
batch_size = 100


# %%
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

# %%
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# %%
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # ImgIn shape = (?, 28,28, 1)
        # Conv       -> (?, 28, 28, 32)
        # Conv       -> (?, 14, 14, 32)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ImgIn shape = (?, 14, 14, 32)
        # Conv       -> (?, 14, 14, 64)
        # Pool       -> (?, 7, 7, 64)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 7*7*64 inputs -> 10 output
        self.fc = torch.nn.Linear(7*7*64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        

# %%
model = CNN().to(device)

# %%
criterion = torch.nn.CrossEntropyLoss().to(device)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

# %%

for epoch in range(training_epochs):
    avg_cost = 0

    #미니 배치 X는 미니배치, Y는 레이블
    for X, Y in data_loader:
        X=X.to(device)
        Y=Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# %%
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

# %%
