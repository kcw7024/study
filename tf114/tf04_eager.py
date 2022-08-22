import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())  # False

# tf.disable_eager_execution()
# WARNING:tensorflow:From c:/study/tf114/tf04_eager.py:5: The name tf.disable_eager_execution is deprecated. Please use tf.compat.v1.disable_eager_execution instead.

# 즉시 실행 모드!
# 버전 1점대에서만 사용
# 즉시실행 모드 :: 텐서 2점대 버전을 즉시실행한다
tf.compat.v1.disable_eager_execution()

# print(tf.executing_eagerly())

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))
