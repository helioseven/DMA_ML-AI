import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# generate randomized data set
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.2, random_state=101)

# build model and fit it
model = KMeans(n_clusters = 4)
model.fit(data[0])

'''
# doesn't work as intended!
# kmeans algorithm generates different labels than original dataset
numlist = []
for index in range(len(data[1])):
	if data[1][index] == model.labels_[index]:
		numlist.append(1)
	else:
		numlist.append(0)
'''

# plt.scatter(data[0][:,0], data[0][:,1], c = model.labels_)

# build new pyplot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

# build first subplot
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c = model.labels_)

# build second subplot
ax2.set_title("Original Data")
ax2.scatter(data[0][:,0],data[0][:,1],c = data[1])

# save figure to disk
fig.savefig("kmeans.png")

# show plot
plt.show()