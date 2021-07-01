import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n = 100
true_w = tf.expand_dims(tf.constant([3, 4, 6],dtype=tf.dtypes.float32),axis=0)

x_true=tf.random.uniform((2,n))
# adding bais
x_true = tf.concat([x_true,tf.expand_dims(tf.ones(n),axis=0)],axis=0)

z_m = tf.matmul(true_w,x_true)
z_true = z_m+tf.random.normal(z_m.shape)

def loss(y_true,y_pred):
    return tf.math.reduce_mean(tf.math.square(y_true-y_pred))

w = tf.Variable(tf.random.uniform((1,3)))
lr = 0.5
epochs = 20
for i in range(epochs):
    with tf.GradientTape() as t:
        z = tf.matmul(w, x_true)

        epoch_loss = loss(z_true, z)
        print(epoch_loss.numpy())
    dw = t.gradient(epoch_loss, w)
    w.assign_sub(lr*dw)


print(w)


