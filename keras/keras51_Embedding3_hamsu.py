from colorsys import yiq_to_rgb
from keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터

docs = [
        '너무 재밌어요','참 최고예요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고 싶네요', '글쎄요', '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '민수가 못 생기긴 했어요', '안결 혼해요'
        ]

# 긍정1 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) #(14, )

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밌어요': 3, '최고예요': 4, '잘': 5, '만든': 6, '영화예요': 7, 
# '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, 
# '싶네요': 15, '글쎄요': 16, '별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
# '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '민수가': 25, '못': 26, 
# '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], 
# [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]
# 길이가 맞지 않기때문에 가장 큰 요소에 맞춰 적은 곳에 0을 채워넣어준다
# 너무 클때는 잘라서 버린다.(ex 30000개~ 너무 긺)

from keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=5, truncating='post') #maxlen = 최대글자수 지정   truncating= 
print(pad_x)
print(pad_x.shape) # (14, 5)

word_size = len(token.word_index)
print("word_size : ", word_size) #word_size : 단어사전의 갯수 30

print(np.unique(pad_x, return_counts=True))

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
# array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
#       dtype=int64))


# 통상 앞부분에 0을 채워준다. 연산을 고려함.
# 제일 큰 수기준으로 0을 채워준다
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25 26 27 28]
#  [ 0  0  0 29 30]]


#2. 모델
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

model = Sequential()

#현재 input 원핫하지 않은 상태의 : (14, 5) 
#output_dim : output 노드의 갯수
#임베딩은 input_dim을 제일 앞에 넣어준다.
#input_dim : 단어사전의 갯수, 
#model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) #원핫하지 않고 임베딩하여 바로 인풋.
#model.add(Embedding(input_dim=31, output_dim=10)) #input_length를 모를때는 명시하지 않아도 자동으로 잡아준다.
#model.add(Embedding(31, 10))
#model.add(Embedding(31, 10, 5)) #error
#model.add(Embedding(31, 10, input_length=5))
#model.add(LSTM(32)) #요게 첫번째로가려면 원핫해줘야함.
#model.add(Dense(1, activation='sigmoid'))

input = Input(shape=(5, ))
emd = Embedding(31, 10)(input)
lstm = LSTM(32)(emd)
output = Dense(1, activation='sigmoid')(lstm)

model = Model(inputs = input, outputs = output)



#model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 10)             310          # input_dim(전체단어사전의갯수) * output_dim #10개짜리가 5개씩 묶여있는걸 던져줌
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5504
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,847
Trainable params: 5,847
Non-trainable params: 0
_________________________________________________________________

'''


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=30, batch_size=16)


#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)

### 실습 ###
x_predict = ['나는 형권이가 정말 재미없다 너무 정말']

token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)
print(x_predict)
y_predict = model.predict(x_predict)
print(y_predict)


if y_predict >= 0.5 :
    print("긍정")
else :
    print("부정")



