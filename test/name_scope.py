# https://github.com/tensorflow/tensorflow/issues/14361
# https://www.tinymind.cn/articles/3844
import tensorflow as tf

with tf.name_scope("test") as scope:
    a = tf.keras.layers.Input(shape=(5,),name='a')
    print(a.name)
    b = tf.keras.layers.Dense(5,name='test/b')
    c = b(a)
    # print(a.get_weights())
    print(b.name)
    print(c.name)