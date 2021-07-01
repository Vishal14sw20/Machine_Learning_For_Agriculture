import tensorflow as tf
import numpy as np

x0 = tf.Variable(1.2)
x1 = tf.Variable(3.1)
w0=tf.Variable(1.5)
w1=tf.Variable(-1.5)
w2=tf.Variable(2.0)

#x0=1.2
#x1=3.1
#w0=1.5
#w1=-1.5

#w2=2.0

def backward():
    with tf.GradientTape() as t:
        f = forward()
    return t.gradient(f,[w0,w1,w2,x0,x1])


def forward():
    return tf.square(tf.tanh(1/(w0*x0+w1*x1+w2))-1)

forward_value =forward()
backward_w1 = backward()
print(forward_value)
print(backward_w1)


"""init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(forward_value.eval())"""
    #print(backward_w1.eval())
