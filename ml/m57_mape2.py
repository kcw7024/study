from gettext import npgettext
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import tensorflow as tf
import keras


#1.
# y_true = np.array([100., 2])
# y_pred = np.array([200., 102])
# mae :  100.0
# mape :  25.5


#2.
y_true = np.array([100., 200])
y_pred = np.array([200., 300])
# mae :  100.0
# mape :  0.75

mae = mean_absolute_error(
    y_true, y_pred
)
print('mae : ', mae)

mape = mean_absolute_percentage_error(
    y_true, y_pred
)
print('mape : ', mape)

mape_tf = keras.metrics.mean_absolute_percentage_error(
    y_true, y_pred
)
print('mape_tf :: ', mape_tf.numpy()) # Tensor 형태로 나오기때문에 numpy로 바꿔서 출력해줌 걍 보기 편하니까