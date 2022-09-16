# logistic regression :: 논리회귀 , 이진분류에만 사용!!!! regression + sigmoid

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

#2. 모델

# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
#     nn.Sigmoid(),
# ).to(DEVICE)
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
    
    
model = Model(4, 3).to(DEVICE)

#3. 컴파일, 훈련

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(model, criterion, optimizer, x_train, y_train):
    # model.trian()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 100

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

y_pred = torch.argmax(model(x_test), axis=1)

score = (y_pred == y_test).float().mean()
print('ACC1 : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu(), y_pred.cpu())
print('ACC2 : ', score)

'''

최종 LOSS :  0.6133090257644653
ACC1 : 0.9333
ACC2 :  0.9333333333333333

'''