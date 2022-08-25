from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(1234)


# 1. 데이터

x1_data = [73., 93., 89., 96., 73., ]  # 국어
x2_data = [80., 88., 91., 98., 66., ]  # 영어
x3_data = [75., 93., 90., 100., 70., ]  # 수학
y_data = [152., 185., 180., 196., 142., ]  # 환산점수

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)


w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

# 2. 모델

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

loss = tf.reduce_mean(tf.square(hypothesis-y_data))  # mse
# square = 제곱, reduce_mean = 평균
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 3-1. 컴파일

# with tf.compat.v1.Session() as sess:
#     # sess = tf.compat.v1.Session()
#     sess.run(tf.global_variables_initializer())  # 초기화 시킨다.

#     epochs = 1001
#     for step in range(epochs):
#         # sess.run(train)
#         _, loss_val, W1_val, W2_val, W3_val, b_val = sess.run([train, loss, w1, w2, w3, b],  # 그래프화
#                                                               feed_dict=(
#             {x1: x1_data, x2: x2_data, x3: x3_data})
#         )
#         if step % 20 == 0:  # %20 의 나머지가 0이 아닐때 프린트함. 즉, 20번에 한번씩 실행시킨다.
#             print(step, loss_val, W1_val, W2_val, W3_val, b_val)

# y_predict = x1_data * W1_val
# print(y_predict)
# r2 = r2_score(y_data, y_predict)
# print("R2 :: ", r2)
# mae = mean_absolute_error(y_data, y_predict)
# print("mae :: ", mae)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)

sess.close()

# 4. 평가, 예측

# y_predict = x_data * hy_val
# print(y_predict)

r2 = r2_score(y_data, hy_val)
print("R2 :: ", r2)
mae = mean_absolute_error(y_data, hy_val)
print("mae :: ", mae)

# R2 ::  0.9725176614062717
# mae ::  2.9330657958984374

# R2 ::  0.9992461476667329
# mae ::  0.4439666748046875
