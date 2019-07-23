import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load a new data set
dataset = pd.read_csv("./housing.csv")

# data pre-processing
dataset.dropna(inplace=True)

# initializations
X = dataset[["ocean_proximity"]]
Y = dataset[["median_house_value"]]

prelist = X.values.tolist()
strlist = []
numlist = []

# convert list of list of strings to list of strings
for item in prelist:
	strlist.append(item[0])

# encode list of strings as numbers
for item in strlist:
	if item == 'INLAND':
		numlist.append(0)
	elif item == 'NEAR BAY':
		numlist.append(1)
	elif item == 'NEAR OCEAN':
		numlist.append(2)
	elif item == '<1H OCEAN':
		numlist.append(3)
	elif item == 'ISLAND':
		numlist.append(4)

# reconstructing our DataFrame from processed list of features
d = {"ocean_proximity" : numlist}
X = pd.DataFrame(data=d)

# split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# create new model, then fit it
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

# make predictions for our test data set and graph comparison
predictions = lr_model.predict(X_test)
fig = plt.scatter(Y_test, predictions).get_figure()
fig.savefig(fname="output.png")