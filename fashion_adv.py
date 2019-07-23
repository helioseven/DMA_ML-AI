import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

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
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_valid = X_valid.reshape((X_valid.shape[0], 28, 28, 1))
Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_valid = Y_valid.reshape((Y_valid.shape[0], 1))

# build our model, compile, and summarize it
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=1,
	             input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D((2, 2), 1))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
			  optimizer=Adam(lr=0.001), metrics=["accuracy"])
model.summary()

# fit the model to our training data
model.fit(X_train, Y_train, batch_size=512, epochs=10, verbose=1,
	      validation_data=(X_valid, Y_valid))

# evaluate model on validation data and print results
results = model.evaluate(X_valid, Y_valid)
print("Validation loss:")
print(results[0])
print("Validation accuracy:")
print(results[1])