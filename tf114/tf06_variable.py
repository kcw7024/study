import tensorflow as tf
sess = tf.compat.v1.Session()


x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)

# 초기값을 넣을 수 있는 상태로 만들어준다
init = tf.compat.v1.global_variables_initializer()
sess.run(init)  # 실행을 시켜줘야 먹힘

print(sess.run(x+y))
