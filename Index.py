import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# !unzip '/content/Train_UQcUa52 (1).zip'
df = pd.read_csv("/content/train.csv")
df.head()
X = df.filename
Y = df.label
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
X_data = []
for name in X:
  path = "/content/Images/train/"+name
  i = mpimg.imread(path)
  X_data.append(i)
np.array(X_data[112]).shape

x_train, x_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.20)
x_train = np.array(x_train)
x_test = np.array(x_test)
#x_train.reshape(-1,1)
x_train.shape
input_shape = (28, 28, 4)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
# Importing the required Keras modules containing model and layers
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=20)