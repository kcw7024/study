import tensorflow as tf
tf.compat.v1.set_random_seed(123)

variable = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')

# 1. 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(variable)
print('aaa : ', aaa)
# aaa :  [-1.5080816]
sess.close()

# 2. 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = variable.eval(session=sess) # 변수를 받기위한 필수과정
print('bbb : ', bbb)
# bbb :  [-1.5080816]
sess.close()

# 3. 초기화 세번째
sess = tf.compat.v1.InteractiveSession()  # 별개로 eval의 파라미터를 지정하지 않아도 된다.
sess.run(tf.compat.v1.global_variables_initializer())
ccc = variable.eval() # 변수를 받기위한 필수과정
print('ccc : ', ccc)
# ccc :  [-1.5080816]
sess.close()
