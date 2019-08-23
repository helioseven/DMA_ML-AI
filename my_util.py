import numpy as np
import matplotlib.pyplot as plt

def argmax_4d(matrix):
	index = (0, 0, 0, 0)
    high_score = 0.0
    for i in range(len(matrix)):
    	for j in range(len(matrix[i])):
    		for k in range(len(matrix[i][j])):
    			for l in range(len(matrix[i][j][k])):
        			if matrix[i][j][k][l] > high_score:
        				index = (i, j, k, l)
        				high_score = matrix[i][j][k][l]
    return index

def make_graph(inList):
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

def sample_memory(buffered_list, batch_size):
    buffer_size = len(buffered_list)

    index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
    return [buffered_list[i] for i in index]