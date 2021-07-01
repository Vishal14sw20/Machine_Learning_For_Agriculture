import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_random_x(num_x):
    return np.random.rand(num_x) * 2 - 1


def f_org(x):
    return 1.0 / (np.exp(-10 * x ** 2) + 1) - 0.5


def f_noisy(x, s=0.1):
    return f_org(x) + s * np.random.randn(len(x))

x = get_random_x(1000)
x_test = np.array([1,2,3,4,5,6,7,8,9,0])
y = f_noisy(x)
k = 4
lr = 0.06
epochs = 50000
losses = []

class Model(object):
    def __init__(self):
        self.w1 = tf.Variable(tf.random.uniform([k,1],dtype=tf.dtypes.float64))
        self.w2 = tf.Variable(tf.random.uniform([1,k],dtype=tf.dtypes.float64))
        self.b1 = tf.Variable(tf.random.uniform([k,1],dtype=tf.dtypes.float64))
        self.b2 = tf.Variable(tf.random.uniform([1],dtype=tf.dtypes.float64))


    def __call__(self, x):
        hidden_layer=tf.math.sigmoid((tf.matmul(self.w1,x.reshape(1,-1)))+self.b1)
        output_layer = tf.matmul(self.w2,hidden_layer)+self.b2
        return output_layer

def loss(y,y_hat):
    return tf.reduce_mean(tf.square(y-y_hat))

def update_param(model):
    with tf.GradientTape() as t:
        c_loss=loss(y,model(x))
        losses.append(c_loss)
    dw1,dw2,db1,db2 =t.gradient(c_loss,[model.w1,model.w2,model.b1,model.b2])
    model.w1.assign_sub(lr*dw1)
    model.w2.assign_sub(lr*dw2)
    model.b1.assign_sub(lr*db1)
    model.b2.assign_sub(lr*db2)

model = Model()
for i in range(epochs):
    update_param(model)

y_predicted = model(x)
fig, ax1 = plt.subplots()
ax1.plot(losses,range(epochs))
fig, ax = plt.subplots()
ax.scatter(x,y)
ax.scatter(x,np.squeeze(y_predicted))
plt.show()


