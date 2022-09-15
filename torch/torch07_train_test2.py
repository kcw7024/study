from unittest import result
import numpy as np
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# cuda가 있으면 cuda를 사용하고, 없음 cpu 사용
print('torch ::',torch.__version__,'사용 DEVICE ::',DEVICE)
# torch :: 1.12.1 사용 DEVICE :: cuda

#1.데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_predict = np.array([11,12,13])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, 
    train_size=0.7, # 원하는 비율로 나눈다. test size와 train size중 하나의값만 지정해도 정상작동
    #shuffle=False, # 데이터셋을 섞는다. 
    random_state=66 # 랜던값 고정
    
    #train_test_split 함수의 기본은 shuffle=True
)

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

x_predict = torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# torch.Size([7, 1]) torch.Size([7, 1]) torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델 구성
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 3),
    nn.ReLU(),
    nn.Linear(3, 2),
    nn.ReLU(),
    nn.Linear(2, 6),
    nn.Linear(6, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Linear(2, 1),    
).to(DEVICE)

#3. 컴파일, 훈련

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 2200

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss :{}'.format(epoch, loss))    

def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        x_predict = model(x_test)
        results = criterion(x_predict, y_test)
        return results.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 LOSS ::', loss2)

results = model(x_predict)
results = results.cpu().detach().numpy()
print('예측결과 : ', results)


# 최종 LOSS :: 0.13646523654460907
# 예측결과 :  [[11.      ]
#  [12.000001]
#  [13.000001]]



