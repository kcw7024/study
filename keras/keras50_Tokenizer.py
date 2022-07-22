from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'


token = Tokenizer()
token.fit_on_texts([text]) 

print(token.word_index)

x = token.texts_to_sequences([text])
print(x)

#[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# x = to_categorical(x) 
# print(x)
# print(x.shape) #(1, 11, 9)

ohe = OneHotEncoder()
x = np.array(x)
x = x.reshape(-1, 1)
x = ohe.fit_transform(x).toarray()
print(x)

# 0부터 시작하니까 추가댐
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]

# 원핫으로수정해볼것.
# ohe = OneHotEncoder()
# x = ohe.fit_transform(x)
# print(x)

