import tensorflow as tf
from tensorflow import keras

input = keras.Input(shape=(128,128,3))

conv1 = keras.layers.Conv2D(64,kernel_size=3,padding="same",activation='relu')(input)
conv1 = keras.layers.Conv2D(64,kernel_size=3,padding="same",activation='relu')(conv1)

pool1 = keras.layers.MaxPooling2D(pool_size=(2,2),padding="same")(conv1)

conv2 = keras.layers.Conv2D(128,kernel_size=3,padding="same",activation='relu')(pool1)
conv2 = keras.layers.Conv2D(128,kernel_size=3,padding="same",activation='relu')(conv2)

pool2 = keras.layers.MaxPooling2D(pool_size=(2,2),padding="same")(conv2)

conv3 = keras.layers.Conv2D(256,kernel_size=3,padding="same",activation='relu')(pool2)
conv3 = keras.layers.Conv2D(256,kernel_size=3,padding="same",activation='relu')(conv3)

up_conv1 =keras.layers.Conv2DTranspose(filters=256,kernel_size=(3,3),strides=2,padding="same")(conv3)
merge1 = keras.layers.concatenate([conv2,up_conv1])

conv4 = keras.layers.Conv2D(128,kernel_size=3,padding="same",activation='relu')(merge1)
conv4 = keras.layers.Conv2D(128,kernel_size=3,padding="same",activation='relu')(conv4)

up_conv2 =keras.layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2,padding="same")(conv4)
merge2 = keras.layers.concatenate([conv1,up_conv2])

conv5 = keras.layers.Conv2D(64,kernel_size=3,padding="same",activation='relu')(merge2)
conv5 = keras.layers.Conv2D(64,kernel_size=3,padding="same",activation='relu')(conv5)

output= keras.layers.Conv2D(3,kernel_size=1,padding="same")(conv5)

model = keras.Model(inputs=input, outputs=output)

tf.keras.utils.plot_model(model,show_shapes=True)
