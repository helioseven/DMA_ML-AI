import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio as imgio

def makeAnimation(inFrames):
	# convert all RGB array to PIL Images first
	images = []
	for frame in inFrames:
		images.append(Image.fromarray(frame, 'RGB'))

	# then simply pass the list of Images to imageio
	imgio.mimsave("sequence.gif", images, format='GIF', duration=0.005)

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
	index = np.random.choice(buffer_size, size=batch_size, replace=False)
	return [buffered_list[i] for i in index]