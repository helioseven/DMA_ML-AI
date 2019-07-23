import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# uncomment the next line if you are working in Jupyter to see results in window
# %matplotlib inline

# load a new data set
dataset = pd.read_csv("./housing.csv")

# data pre-processing
dataset = dataset.drop(columns=["ocean_proximity"])
dataset.dropna(inplace=True)

# grab relevent columns of dataset for inputs
X = dataset[["longitude", "latitude", "total_rooms", "population", "households", "median_income"]]
# and then again for outcomes
Y = dataset[["median_house_value"]]

# split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# create new model, then fit it
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

# make predictions for our test data set and graph comparison
predictions = lr_model.predict(X_test)
fig = plt.scatter(Y_test, predictions, color="black").get_figure()

# generate an idenitity (X = Y) line and plot it
actuals = Y_test.values.tolist()
identity_line = np.linspace(max(min(actuals), min(predictions)),
                            min(max(actuals), max(predictions)))
plt.plot(identity_line, identity_line, color="red",
	     linestyle="dashed", linewidth=2.5)

# add some pretty printing to our graph
plt.title("Scatterplot of Housing Price Prediction Accuracy")
plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")

# save the figure and then show it
fig.savefig(fname="output.png")
plt.show()