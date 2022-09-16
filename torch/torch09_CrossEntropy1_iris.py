
from calendar import EPOCH
from sklearn.datasets import load_breast_cancer, load_iris
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0


#1. 데이터 
datasets = load_iris()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.FloatTensor(y_train).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)

x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)
# y_test = torch.FloatTensor(y_test).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())
print(x_train.shape)

# torch.Size([105, 4])
# torch.Size([105, 4])

#2. 모델

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 48),
    nn.ReLU(),
    nn.Linear(48, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    # nn.Softmax() < 생략해도됨
).to(DEVICE)


#3. 컴파일, 훈련

# criterion = nn.BCELoss() # 이진분류
criterion = nn.CrossEntropyLoss() # sparse corssentorpy 
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.trian()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 1500

for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss :{}'.format(epoch, loss))    

def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()
    
loss = evaluate(model, criterion, x_test, y_test)
print('최종 LOSS : ', loss)

# y_pred = model(x_test)
# print(y_pred[:10])

# y_pred = (model(x_test) >= 0.5).float()
# print(y_pred[:10])

y_pred = torch.argmax(model(x_test), axis=1)
# print(y_pred[:10])

score = (y_pred == y_test).float().mean()
print('ACC1 : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_pred)  # ERROR
# print('ACC : ', score)
score = accuracy_score(y_test.cpu(), y_pred.cpu())
print('ACC2 : ', score)


'''

최종 LOSS :  0.5865715742111206
tensor([2, 2, 1, 1, 2, 0, 0, 2, 0, 0], device='cuda:0')
ACC1 : 0.9333
ACC2 :  0.9333333333333333

'''