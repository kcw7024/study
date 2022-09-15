from calendar import EPOCH
from inspect import Parameter
import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# cuda가 있으면 cuda를 사용하고, 없음 cpu 사용
print('torch ::',torch.__version__,'사용 DEVICE ::',DEVICE)
# torch :: 1.12.1 사용 DEVICE :: cuda

#1. 데이터

x = np.array([1,2,3]) # (3, )
y = np.array([1,2,3]) 
x_test = np.array([4])
# torch는 numpy가 아니라 torchdata로 사용.(Tensor형 데이터로 변환) !필수!

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) 
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE) 
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE) 


print(torch.mean(x).item(), torch.std(x).item())

# 스케일링
# x_test = (x_test - torch.mean(x)) / torch.std(x)
x_test = (x_test - torch.min(x)) / (torch.max(x) - torch.min(x))
x_test = x_test * (torch.max(x_test) - torch.min(x_test)) + torch.min(x_test)
# x = (x - torch.mean(x)) / torch.max(x) # torch의 분산으로 x에서 평균값을 빼준걸 나눠준다 
# torch.Size([3, 1]) torch.Size([3, 1]) 
x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
x = x * (torch.max(x) - torch.min(x)) + torch.min(x)


print(x, y)
print(x.shape, y.shape) # torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델 구성

# model = Sequential() : 텐서형태
model = nn.Linear(1, 1).to(DEVICE) # input (x), output(y)


#3. 컴파일, 훈련

# loss에 대한 반환값으로 변수명을 criterion을 많이씀~
# model.compile(loss='mse', optimizer='SGD')
criterion = nn.MSELoss()
optimizer  = optim.SGD(model.parameters(), lr=0.01)
# optim.Adam(model.Parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y) : 
    # model.train() 훈련모드!!!! == 디폴트값임.
    optimizer.zero_grad() # 손실함수의 기울기를 초기화 한다. 미분값 누적을 방지하기위해 0값으로 초기화
    
    hypothesis = model(x)
    
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis, y) # class이기 때문에 문법 변경ㄴ
    loss = F.mse_loss(hypothesis, y)
    
    
    loss.backward() # 역전파
    optimizer.step() # 역전파시킨것에대한 가중치 갱신   
    # zero_grad, backward, step, 무조건 들어감. 외워버려!
    # 정방향 진행후 역전파로 훈련하여 가중치갱신, = 1 epochs
    return loss.item()

epochs = 1000

for epoch in range(1, epochs+1) : 
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss :{}'.format(epoch, loss))    


#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)

def evaluate(model, criterion, x, y):
    model.eval() # 평가모드로 시작.(평가에서 무조건)
    
    with torch.no_grad() :  # 그라디언트를 사용하지 않겠다! :: 순전파만쓰겠음. 역전파는사용하지않겠다!
        y_predict = model(x)
        results = criterion(y_predict, y)
    return results.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :: ', loss2)

# y_pred = model.predict([4])
# results = model(torch.Tensor([[4]]).to(DEVICE))
results = model(x_test)
print('예측값 :: ', results.item())

