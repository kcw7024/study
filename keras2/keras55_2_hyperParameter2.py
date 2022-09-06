# 노드추가 learning_rate 추가 

from pickletools import optimize
from random import vonmisesvariate
from tabnanny import verbose
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Input, Dropout
import keras
import tensorflow as tf
print(tf.__version__)

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. 모델

def build_model(drop=0.5,optimizer='adam',activation='relu'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)    
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], 
                  loss='sparse_categorical_crossentropy')
    
    return model


def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']
    return {"batch_size" : batchs, "optimizer" : optimizers,
            "drop" : dropout, "activation" : activation}
    

hyperparameters = create_hyperparameter()
#print(hyperparameters)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


keras_model = KerasClassifier(build_fn=build_model, verbose=1)

    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# model = GridSearchCV(keras_model, hyperparameters, cv=3)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=2, verbose=1)


import time
start = time.time()
model.fit(x_train, y_train, epochs=3, validation_split=0.4)
end = time.time()

print("걸린시간 : ", end-start)
print("model.best_params_ :", model.best_params_)
print("model.best_estimator_ :", model.best_estimator_)
print("model.best_score_ :", model.best_score_)
print("model.score :", model.score)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
print("acc : ", accuracy_score(y_test, y_pred))

'''
걸린시간 :  296.9476501941681
model.best_params_ : {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 500, 'activation': 'relu'}        
model.best_estimator_ : <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000269B69CEF10>
model.best_score_ : 0.9742333292961121
acc :  0.9624

'''