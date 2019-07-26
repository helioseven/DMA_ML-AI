import matplotlib.pyplot as plt

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