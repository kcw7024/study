from cProfile import run
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

x = np.array([range(10)]) #범위 함수
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]]
             )

x_test = np.array([9])

x = torch.FloatTensor(np.transpose(x)).to(DEVICE) 
y = torch.FloatTensor(np.transpose(y)).to(DEVICE) 
x_test = torch.FloatTensor(np.transpose(x_test)).to(DEVICE)

# x = torch.transpose(x, 0, 1)
# x_test = torch.transpose(x_test, 0, 1)


# y = torch.transpose(y).to(DEVICE)

print(torch.mean(x).item(), torch.std(x).item())

# 스케일링
x_test = (x_test - torch.mean(x)) / torch.std(x)
x = (x - torch.mean(x)) / torch.std(x) # torch의 분산으로 x에서 평균값을 빼준걸 나눠준다 

print(x, y)
print(x.shape, y.shape, x_test.shape) 
# torch.Size([10, 1]) torch.Size([10, 3]) torch.Size([1])

#2. 모델 구성


# model = Sequential() : 텐서형태
model = nn.Sequential(
    nn.Linear(1, 4),
    nn.Linear(4, 5),    
    nn.Linear(5, 3),
    nn.ReLU(),    
    nn.Linear(3, 2),    
    nn.Linear(2, 3),    
).to(DEVICE)

#3. 컴파일, 훈련

# loss에 대한 반환값으로 변수명을 criterion을 많이씀~
# model.compile(loss='mse', optimizer='SGD')
criterion = nn.MSELoss()
optimizer  = optim.SGD(model.parameters(), lr=0.001)
# optim.Adam(model.Parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y) : 
    # model.train() 훈련모드!!!! == 디폴트값임.
    optimizer.zero_grad() # 손실함수의 기울기를 초기화 한다. 미분값 누적을 방지하기위해 0값으로 초기화
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis, y) # class이기 때문에 문법 변경ㄴ
    # loss = F.mse_loss(hypothesis, y)
        
    loss.backward() # 역전파
    optimizer.step() # 역전파시킨것에대한 가중치 갱신   
    # zero_grad, backward, step, 무조건 들어감. 외워버려!
    # 정방향 진행후 역전파로 훈련하여 가중치갱신, = 1 epochs
    return loss.item()

epochs = 2800


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
results = model(x_test.to(DEVICE))
# print(results.item())
# results = results.cpu().detach().numpy()
results = results.tolist()

print('예측값 :: ', results)
# print(results.item())



'''
최종 loss ::  0.2959948778152466
예측값 ::  [10.3093,  1.8624,  0.3259]
'''