# logistic regression :: 논리회귀 , 이진분류에만 사용!!!! regression + sigmoid

from calendar import EPOCH
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, load_boston
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0


# 1. 데이터
path = 'C:/study/_data/bike_sharing/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)


x = torch.Tensor(x.values)
y = torch.Tensor(y.values)

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
    )

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)


print(x_train.size())
print(x_train.shape)


# torch.Size([1257, 64])
# torch.Size([1257, 64])

#2. 모델

# model = nn.Sequential(
#     nn.Linear(13, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 48),
#     nn.ReLU(),
#     nn.Linear(48, 16),
#     nn.ReLU(),
#     nn.Linear(16, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
#     # nn.Sigmoid(),
# ).to(DEVICE)


class Model(nn.Module) :
    def __init__(self, input_dim, output_dim) :
        # super().__init__() # super 사용시 불러온 모듈의 함수와 변수 모두 사용하겠다
        super(Model, self).__init__() # 위와같음
        
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input_size): 
        x = self.linear1(input_size)                
        x = self.relu(x)                
        x = self.linear2(x)
        x = self.relu(x)                                
        x = self.linear3(x)
        x = self.relu(x)                                
        x = self.linear4(x)
        # x = self.sigmoid(x)
        # x = self.softmax(x)
        return x 
    
    
model = Model(12, 1).to(DEVICE)


#3. 컴파일, 훈련

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(model, criterion, optimizer, x_train, y_train):
    # model.trian()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 2500

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

y_pred = model(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
print('R2 : ', score)


'''
최종 LOSS :  32022.61328125
R2 :  -0.003084159360155825

'''