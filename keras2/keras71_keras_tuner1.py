import tensorflow as tf
from tensorflow.keras.datasets import mnist
import keras_tuner as kt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255., x_test/255.

def get_model(hp) : 
    hp_unit1 = hp.Int('units1', min_value=16, max_value=512, step=16)
    hp_unit2 = hp.Int('units2', min_value=16, max_value=512, step=16)
    hp_unit3 = hp.Int('units3', min_value=16, max_value=512, step=16)
    hp_unit4 = hp.Int('units4', min_value=16, max_value=512, step=16)
    
    hp_drop = hp.Choice('dropout', values=[0.0, 0.2, 0.3, 0.4, 0.5])
    
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    