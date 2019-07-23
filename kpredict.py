import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# generate randomized data set
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.2, random_state=101)

# build model and fit it
model = KMeans(n_clusters = 4)
model.fit(data[0])

# get input from the user
in_x = input("Supply an X value:")
in_y = input("Supply a Y value:")

# convert input strings to float values
in_x = float(in_x)
in_y = float(in_y)

# construct a data set from our input values
data_point = [[in_x, in_y]]

# generate a prediction set for our constructed data set
prediction = model.predict(data_point)

# print out the first (in this case only) prediction
print(prediction[0])