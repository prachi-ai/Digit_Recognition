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
#plt.imshow(np.round(x_train[115]))
# Reshaping the array to 4-dims so that it can work with the Keras API

#x_train = x_train.reshape(x_train.shape[0], 56, 56, 1)
#x_test = x_test.reshape(x_test.shape[0], 56, 56, 1)
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
