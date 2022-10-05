from json import load
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(15), tr.ToTensor()])

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0



#1. 데이터
path = './_data/torch_data/'

# train_dataset = MNIST(path, train=True, download=True, transform=transf)
# test_dataset = MNIST(path, train=False, download=True, transform=transf)
# print(train_dataset[0][0].shape) # torch.Size([1, 15, 15])

train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

print(x_train.shape, x_test.size())
print(y_train.shape, y_test.size())
# torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
# torch.Size([60000]) torch.Size([10000])

# print(np.min(x_train.numpy()), np.max(x_train.numpy()))
# 0.0 1.0


# tensor 와 torch 가 다른점
# 60000, 28, 28, 1 -> torch shape :: 60000, 1, 28, 28

# view = reshape 
# x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)
x_train, x_test = x_train.unsqueeze(1), x_test.unsqueeze(1)
print(x_train.shape, x_test.size()) # torch.Size([60000, 1, 28, 28]) torch.Size([10000, 1, 28, 28])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True) 
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

#2. 모델
class CNN(nn.Module) : 
    def __init__(self, num_features) :
        super(CNN, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5)            
        )
            
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5)            
        ) # 32, 5, 5
 
        self.hidden_layer3 = nn.Linear(32*5*5, 32)
        
        self.output_layer = nn.Linear(in_features=32, out_features=10)

    def forward(self, x) :
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = x.view(x.shape[0], -1) # flatten
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x    

model = CNN(1).to(DEVICE)

print(model)

'''
CNN(
  (hidden_layer1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.5, inplace=False)
  )
  (hidden_layer2): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.5, inplace=False)
  )
  (hidden_layer3): Linear(in_features=800, out_features=32, bias=True)
  (output_layer): Linear(in_features=32, out_features=10, bias=True)
)
'''




#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4) # 0.0001

def train(model, criterion, optimizer, loader) : 
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader :
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
        y_pred = torch.argmax(hypothesis, 1)
        acc = (y_pred == y_batch).float().mean() # acc
        epoch_acc += acc.item()
        
    return epoch_loss / len(loader), epoch_acc / len(loader)

# ㄴ hist = model.fit(x_train, y_train)
# hist 에는 loss와 acc가 들어간다
# 엄밀하게 말하면 hist라기보다는 loss와 acc를 반환해주는개념 


def evaluate(model, criterion, loader) : 
    model.eval() # 평가모드이기 때문에 레이어단계에서 한정짓는 것에 대해서 영향을받지 않는다.(Dropout, BatchNormalization 등)!!!
    
    epoch_loss  = 0
    epoch_acc = 0

    with torch.no_grad() : 
        for x_batch, y_batch in loader : 
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)             

            epoch_loss += loss.item()
    
            y_pred = torch.argmax(hypothesis, 1)
            acc = (y_pred == y_batch).float().mean() # acc
            epoch_acc += acc.item()
        
        return epoch_loss / len(loader), epoch_acc / len(loader)

# ㄴ loss, acc = model.evaluate(x_test, y_test)

epochs = 1

for epoch in range(1, epochs+1) : 
    
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch : {}, loss : {:.4f}, acc : {:.3f}, val_loss : {:.4f}, val_acc : {:.3f}'.format(
        epoch, loss, acc, val_loss, val_acc))
        
        
print("="*50)
print(model)

'''

print(model) :: 연산량이 나오지않음, 모델의 구성만 나옴.

==================================================
CNN(
  (hidden_layer1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.5, inplace=False)
  )
  (hidden_layer2): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.5, inplace=False)
  )
  (hidden_layer3): Linear(in_features=800, out_features=32, bias=True)
  (output_layer): Linear(in_features=32, out_features=10, bias=True)
)
'''

# pip install torchsummary

from torchsummary import summary
summary(model, (1, 28, 28))

'''

:: torch model summary :: 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 26, 26]             640
              ReLU-2           [-1, 64, 26, 26]               0
         MaxPool2d-3           [-1, 64, 13, 13]               0
           Dropout-4           [-1, 64, 13, 13]               0
            Conv2d-5           [-1, 32, 11, 11]          18,464
              ReLU-6           [-1, 32, 11, 11]               0
         MaxPool2d-7             [-1, 32, 5, 5]               0
           Dropout-8             [-1, 32, 5, 5]               0
            Linear-9                   [-1, 32]          25,632
           Linear-10                   [-1, 10]             330
================================================================
Total params: 45,066
Trainable params: 45,066
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.90
Params size (MB): 0.17
Estimated Total Size (MB): 1.07
----------------------------------------------------------------
'''