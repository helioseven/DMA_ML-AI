import matplotlib.pyplot as plt
import numpy as np

def makeGraph(inList):
	# generate figure
	fig = plt.figure()
	# plot the list
	plt.plot(inList, color="black")

	# add some pretty printing to our graph
	plt.title("Scores per Episode")
	plt.xlabel("Episode")
	plt.ylabel("Score")

	# save the figure
	fig.savefig(fname="output.png")

# generates a random sampling of the list passed in
def sampleMemory(buffered_list, batch_size):
	buffer_size = len(buffered_list)
	# generates a list of random indices to pick
	index = np.random.choice(np.arange(buffer_size),
							 size=batch_size,
							 replace=False)
	return [buffered_list[i] for i in index]