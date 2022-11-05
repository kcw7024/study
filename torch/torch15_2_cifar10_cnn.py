from json import load
from torchvision.datasets import MNIST, CIFAR10
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

train_dataset = CIFAR10(path, train=True, download=True)
test_dataset = CIFAR10(path, train=False, download=True)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

print(x_train.shape, x_test.size())
print(y_train.shape, y_test.size())
# torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])
# torch.Size([50000]) torch.Size([10000])

# print(np.min(x_train.numpy()), np.max(x_train.numpy()))
# 0.0 1.0


# tensor 와 torch 가 다른점
# 60000, 28, 28, 1 -> torch shape :: 60000, 1, 28, 28

# view = reshape 
x_train, x_test = x_train.reshape(-1, 3, 32, 32), x_test.reshape(-1, 3, 32, 32)
# x_train, x_test = x_train.unsqueeze(1), x_test.unsqueeze(1) 
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
            nn.Conv2d(num_features, 32, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5)            
        )
            
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5)            
        ) # 32, 5, 5
 
        self.hidden_layer3 = nn.Linear(64*5*5, 32)
        
        self.output_layer = nn.Linear(in_features=32, out_features=10)

    def forward(self, x) :
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = x.view(x.shape[0], -1) # flatten
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x    

model = CNN(3).to(DEVICE)

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

epochs = 10

for epoch in range(1, epochs+1) : 
    
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch : {}, loss : {:.4f}, acc : {:.3f}, val_loss : {:.4f}, val_acc : {:.3f}'.format(
        epoch, loss, acc, val_loss, val_acc))
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    