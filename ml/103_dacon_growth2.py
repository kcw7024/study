import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import 

# import zipfile
# filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
# os.chdir("D:\study_data\_data\dacon_growth/test_target")
# with zipfile.ZipFile("submission_2.zip", 'w') as my_zip:
#     for i in filelist:
#         my_zip.write(i)
#     my_zip.close()

#1. 데이터

path = 'D:/study_data/_data/dacon_growth/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        #print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끝')

train_data, train_label_data = aaa(train_input_list, train_target_list) #, False)
val_data, val_label_data = aaa(val_input_list, val_target_list) #, False)

print(train_data[0])
print(len(train_data), len(train_label_data)) # 1607 1607
print(len(train_data[0])) # 1440

print(train_label_data) # 1440
print(train_data.shape, train_label_data.shape)   # (1607, 1440, 37) (1607,)
print(val_data.shape, val_label_data.shape)   # (206, 1440, 37) (206,)

#2. 모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(1440, 37)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 훈련
# model.fit(train_data, val_data)

#4. 평가, 예측

model.compile(loss='rmse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

import time
start_time = time.time()

model.fit(train_data, val_data, epochs=1, batch_size=128, #validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                  
                 )

end_time = time.time() - start_time

print("걸린시간 : ", end_time-start_time)

#4. 평가, 예측
print(('#'*70) + '1.기본출력')

loss = model.evaluate(train_label_data, val_label_data)
print('loss : ', loss)

y_predict = model.predict(train_label_data)
from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(val_label_data, y_predict)
print('r2 score : ' , r2)
