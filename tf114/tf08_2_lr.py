# 실습
# lr 수정해서 epoch를 100번 이하로 줄인다.
# step = 100 이하, w = 1.99, b = 0.99

# y = wx + b
from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(123)  # 항상 같은 랜덤값이 지정 됨

# 1. 데이터

x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
# shape :: input shape
W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # random_normal = 갯수
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 2. 모델 구성
hypothesis = x_train * W + b  # y = wx + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y_train))  # mse
# square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)  # 로스값이 최적화된 minimize위치를 찾아낸다
# model.compile(loss = 'mse', optimizer = 'sgd')

# 3-2. 훈련
# session은 항상 열고난뒤 닫아줘야 한다/ 마지막에 close
with tf.compat.v1.Session() as sess:
    # sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())  # 초기화 시킨다.

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run([train, loss, W, b],  # 그래프화
                                             feed_dict=(
                                                 {x_train: x_train_data, y_train: y_train_data})
                                             )
        if step % 20 == 0:  # %20 의 나머지가 0이 아닐때 프린트함. 즉, 20번에 한번씩 실행시킨다.
            print(step, loss_val, W_val, b_val)

    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    y_predict = x_test * W_val + b_val  # y_predict = model.predict(x_test)
    sess = tf.compat.v1.Session()
    print('[6,7,8] 예측 : ', sess.run(
        y_predict, feed_dict={x_test: x_test_data}))


########################################################################################

# x_data = [6, 7, 8]
# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
# y_predict = x_test * W_val + b_val  # y_predict = model.predict(x_test)
# sess = tf.compat.v1.Session()
# print('[6,7,8] 예측 : ', sess.run(y_predict, feed_dict={x_test:x_data}))
# [6,7,8] 예측 :  [6.000005 7.000006 8.000008]

#######################################################################################
