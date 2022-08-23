from pickletools import optimize
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.set_random_seed(123)

# 실습
# 08_2를 카피해서 아래를 만든다.


###########################################################################################

print("1. Session // sess.run")

# 1. Session // sess.run

# y = wx + b
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
sess = tf.compat.v1.Session()
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


###########################################################################################

print("2. Session // 변수.eval")

# 2. Session // 변수.eval

x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
# shape :: input shape
W2 = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # random_normal = 갯수
b2 = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# sess = tf.compat.v1.InteractiveSession()
# sess.run(tf.compat.v1.global_variables_initializer())
# W2 = W2.eval(session=sess)
# b2 = b2.eval(session=sess)

# 2. 모델 구성
hypothesis = x_train * W2 + b2  # y = wx + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y_train))  # mse
# square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)  # 로스값이 최적화된 minimize위치를 찾아낸다
# model.compile(loss = 'mse', optimizer = 'sgd')


loss = {x_train: x_train_data, y_train: y_train_data}


# 3-2. 훈련
# session은 항상 열고난뒤 닫아줘야 한다/ 마지막에 close
with tf.compat.v1.Session() as sess:
    # sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())  # 초기화 시킨다.

    epochs = 101
    for step in range(epochs):
        # sess.run(train)

        _, loss_val, W2_val, b2_val = sess.run([train, loss, W2_val, b2_val],  # 그래프화
                                               feed_dict=(
                                               {x_train: x_train_data, y_train: y_train_data})
                                               )
        if step % 20 == 0:  # %20 의 나머지가 0이 아닐때 프린트함. 즉, 20번에 한번씩 실행시킨다.
            print(step, loss_val, W2_val, b2_val)

    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    y_predict = x_test * W2_val + b2_val  # y_predict = model.predict(x_test)
    sess = tf.compat.v1.Session()
    print('[6,7,8] 예측 : ', sess.run(
        y_predict, feed_dict={x_test: x_test_data}))


###########################################################################################

print("3. InteractiveSession() // 변수.eval")

# 3. InteractiveSession() // 변수.eval

# 1. 데이터

x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
# shape :: input shape
W3 = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # random_normal = 갯수
b3 = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 2. 모델 구성
hypothesis = x_train * W3 + b3  # y = wx + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y_train))  # mse
# square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)  # 로스값이 최적화된 minimize위치를 찾아낸다
# model.compile(loss = 'mse', optimizer = 'sgd')

# 3-2. 훈련

# sess.run(tf.compat.v1.global_variables_initializer())
# W3 = W3.eval(session=sess)
# b3 = b3.eval(session=sess)

# session은 항상 열고난뒤 닫아줘야 한다/ 마지막에 close
with tf.compat.v1.Session() as sess:
    # sess = tf.compat.v1.Session()
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.global_variables_initializer())  # 초기화 시킨다.

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        W3_val = W3.eval(),
        b3_val = b3.eval()
        _, loss_val, W3_val, b3_val = sess.run([train, loss, W3, b3],  # 그래프화
                                               feed_dict=(
            {x_train: x_train_data, y_train: y_train_data})
        )
        if step % 20 == 0:  # %20 의 나머지가 0이 아닐때 프린트함. 즉, 20번에 한번씩 실행시킨다.
            print(step, loss_val, W3_val, b3_val)

    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    y_predict = x_test * W3_val + b3_val  # y_predict = model.predict(x_test)
    sess = tf.compat.v1.Session()
    print('[6,7,8] 예측 : ', sess.run(
        y_predict, feed_dict={x_test: x_test_data}))
