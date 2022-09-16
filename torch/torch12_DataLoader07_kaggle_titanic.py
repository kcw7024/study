# logistic regression :: 논리회귀 , 이진분류에만 사용!!!! regression + sigmoid

from calendar import EPOCH
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0


#1. 데이터 

path = './_data/kaggle_titanic/'

train_set = pd.read_csv(path + 'train.csv', index_col=0)
#print(train_set.shape) #(891, 12)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
#print(test_set.shape) #(418, 11)


#1. 데이터

#print(train_set.Pclass.value_counts())
# 3    491
# 1    216
# 2    184
# Name: Pclass, dtype: int64

#각 등급별의 생존 비율을 확인
Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize=True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
#print(f"Percentage of Pclass 1 who survived: {Pclass1}")
#print(f"Percentage of Pclass 2 who survived: {Pclass2}")
#print(f"Percentage of Pclass 3 who survived: {Pclass3}")
# 결과값
# Percentage of Pclass 1 who survived: 62.96296296296296
# Percentage of Pclass 2 who survived: 47.28260869565217
# Percentage of Pclass 3 who survived: 24.236252545824847

#여자와 남자의 생존 비율을 확인
female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
#print(f"Percentage of females who survived: {female}")
#print(f"Percentage of males who survived: {male}")
# Percentage of females who survived: 74.20382165605095
# Percentage of males who survived: 18.890814558058924

# 결측치 처리
# 각 컬럼의 결측치를 처리해준다
train_set = train_set.fillna({"Embarked": "S"}) # S 값을 넣어준다
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean()) # 평균값으로 넣어준다

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True) # pandas의 원핫인코딩사용
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean()) # 평균값으로 넣어준다
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape) #(891, 10) (418, 8)


# x와 y 변수 지정

x = train_set.drop(['Survived'], axis=1)  
#print(x)
#print(x.columns)
#print(x.shape) # (891, 8)
y = train_set['Survived'] 
#print(y)
#print(y.shape) # (891,)

    
    
# x = torch.FloatTensor(x)
# y = torch.FloatTensor(y)

x = torch.Tensor(x.values)
y = torch.Tensor(y.values)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# print(x_train.size())
# print(x_train.shape)

# DateLoader 시작
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train) # x, y 를 합쳐준다.
test_set = TensorDataset(x_test, y_test) 

# print(train_set) # <torch.utils.data.dataset.TensorDataset object at 0x000002A121CDBF70>
# print("="*30, "train_set[0]") 
# print(train_set[0])
# print("="*30, "train_set[0][0]")
# print(train_set[0][0])
# print("="*30, "train_set[0][1]")
# print(len(train_set[0][1]))
# print("="*30, "train_set 총 갯수")
# print(len(train_set)) # 398

# x, y  배치 결합
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=True)

# print(train_loader) # <torch.utils.data.dataloader.DataLoader object at 0x000002B437F0E910>
# print("="*30, "train_loader[0]") 
# print(train_loader[0]) # ERROR
# print("="*30, "train_loader[0][0]")
# print(train_loader[0][0])
# print("="*30, "train_loader[0][1]")
# print(len(train_loader[0][1]))
# print("="*30, "train_loader 총 갯수")
# print(len(train_loader)) # 398


# model class 화 
# class ()안에는 상위클래스만 넣을수있음
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

    def forward(self, input_size): 
        x = self.linear1(input_size)                
        x = self.relu(x)                
        x = self.linear2(x)
        x = self.relu(x)                                
        x = self.linear3(x)
        x = self.relu(x)                                
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x 
    
    
model = Model(8, 1).to(DEVICE)

#3. 컴파일, 훈련

criterion = nn.BCELoss() # 이진분류
optimizer = optim.Adam(model.parameters(), lr=0.01)



def train(model, criterion, optimizer, loader):
    # model.train()

    total_loss = 0
    
    for x_batch, y_batch in loader :
        optimizer.zero_grad()
        hypothesis = model(x_batch) # batch 단위대로 들어감
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # 회당 나오는 loss를 더해줌(누적)
        
    return total_loss / len(loader)

EPOCHS = 100

for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch %  10 == 0 : 
        print('epoch : {}, loss :{}'.format(epoch, loss)) 
    
       
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0

    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
        
    return total_loss
    
loss = evaluate(model, criterion, test_loader)
print('최종 LOSS : ', loss)

y_pred = (model(x_test) >= 0.5).float()
score = (y_pred == y_test).float().mean()
print('ACC1 : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu(), y_pred.cpu())
print('ACC2 : ', score)


'''
최종 LOSS :  9.719212532043457
ACC1 : 0.8134
ACC2 :  0.8134328358208955

'''



