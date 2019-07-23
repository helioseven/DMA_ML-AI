import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# read in data sets
train_frame = pd.read_csv("fashion-mnist_train.csv")
valid_frame = pd.read_csv("fashion-mnist_test.csv")

# convert pandas dataframes into numpy arrays
training = np.array(train_frame, dtype="float32")
validate = np.array(valid_frame, dtype="float32")

# list of strings associated with each label
labels = ["Shirt/Top",
		  "Trouser",
		  "Pullover",
		  "Dress",
		  "Coat",
		  "Sandal",
		  "Shirt",
		  "Sneaker",
		  "Bag",
		  "Ankle Boot"]

# splitting training and validation data sets into features and labels
X_train = training[:,1:]/255
Y_train = training[:,0]
X_valid = validate[:,1:]/255
Y_valid = validate[:,0]

# reshaping data to fit into a linear regression model
X_train = X_train.reshape((X_train.shape[0], 784))
X_valid = X_valid.reshape((X_valid.shape[0], 784))
Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_valid = Y_valid.reshape((Y_valid.shape[0], 1))

# build model and fit it
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

# pick a random item out of the validation data set
rand = random.randint(1, 10000)
example_item = validate[rand,1:]

# generate predictions for our randomly selected item
prediction = lr_model.predict(X_valid)[0][0]
prediction = int(round(prediction))
# print our prediction, then print correct label
print(labels[prediction])
print(labels[int(validate[rand, 0])])

# display the actual image on a pyplot
display_item = example_item.reshape((28, 28))
plt.imshow(display_item)
plt.show()