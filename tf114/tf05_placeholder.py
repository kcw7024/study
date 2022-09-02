import tensorflow as tf
import numpy as np
print(tf.__version__)
print(tf.executing_eagerly())

# 즉시실행모드
tf.compat.v1.disable_eager_execution()  # OFF

print(tf.executing_eagerly())

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

###placeholder####

# 1. placeholder를 정의
a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

# 2. 수식
add_node = a + b

# 3. placeholder로 정의한 공간, 그값에(feed_dict) 숫자를 넣어준다.
print(sess.run(add_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(add_node, feed_dict={a: [1, 3], b: [2, 4]}))

# 2-2. 수식
add_and_triple = add_node * 3

print(add_and_triple)  # Tensor("mul:0", dtype=float32)
print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))
