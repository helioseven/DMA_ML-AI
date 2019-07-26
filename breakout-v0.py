import gym
import pandas as pd
import numpy as np
from collections import deque
import random
import os.path
import csv

import skimage
from skimage import transform
from skimage.color import rgb2gray

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, Activation, Input

from my_util import makeGraph

#constants
stack_size= 6
batch_size = 64
gamma = 0.618
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.00001
decay_step = 0
learning_rate = 0.00025 

# process a raw image frame by cropping, scaling, and normalizing(2)
def processFrame(frame):

	gray = rgb2gray(frame)
	normalized = gray / 255.0
	return transform.resize(normalized, [100,80])

#start using frames in episodes and stack them
def stackFrames(stacked_frames, new_frame):
	return_stack = stacked_frames
	return_state = None
	#process a new frame
	frame = processFrame(new_frame)
	return_stack.append(frame)

	return_state = np.stack(return_stack, axis=2)
	return return_state, return_stack

def predictAction(model, decay_step):
	tradeoff = np.random.random()

	epsilon = max_epsilon \
				   + (min_epsilon - max_epsilon) \
				   * np.exp(-decay_rate * decay_step)

	if epsilon > tradeoff:
		choice = random.randint(1, len(action_codes)) - 1

	else:
		feats = np.array(frame_stack).reshape(1, *state_space) # * unpacks tuple or list
		choice = np.argmax(model.predict(feats))

	return action_codes[choice]

def sampleMemory(buffered_list, batch_size):
	buffer_size = len(buffered_list)

	index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
	return [buffered_list[i] for i in index]

def getModel():
	model = Sequential()
	model.add(Conv2D( 200, (60, 60), input_shape=state_space))
	model.add(Conv2D(100, (20, 20)))
	model.add(Flatten())
	model.add(Dense(100, activation="relu"))
	model.add(Dense(50, activation="relu"))
	model.add(Dense(10, activation="relu"))
	model.add(Dense(action_space, activation="softmax"))
	opt = keras.optimizers.Adam(lr=learning_rate,
								beta_1=min_epsilon,
								beta_2=max_epsilon,
								decay=decay_rate)
	model.compile(optimizer=opt, loss="categorical_crossentropy")
	return model

#create gym environment.    - basic structure(1)
env = gym.make("Breakout-v0")
state_space = (100, 80, stack_size)
action_space = env.action_space.n
action_codes = np.identity(action_space, dtype=np.int).tolist()

#check what k value it works w  - uses 4
	#print(action_space)

# generating a frame stack filled with empty (zeros) images
blank_imgs = [np.zeros((100, 80), dtype=np.int) \
					   for i in range(stack_size)]
frame_stack = deque(blank_imgs, maxlen = stack_size)


scores_list = []


#build model and memory
model = getModel()
memory = deque(maxlen=1000)
if os.path.isfile("memories.csv"):
	with open("memories.csv", "r") as f:
		reader = csv.reader(f)
		memory = list(reader)
success = False

for episode in range(100):
	state = env.reset()
	score = 0
	state, frame_stack = stackFrames(frame_stack, state)

	for step in range(250):
		#env.render()


		decay_step += 1
		#generate random action
		action = np.argmax(predictAction(model, decay_step))

		#apply the action to step env
		obs, reward, done, _ = env.step(action) #basic structure

		#add reward to score
		score =+ reward

		if done ==True:
			if score > 10:
				success = True
			obs = np.zeros((210, 160))
			obs, frame_stack = stackFrames(frame_stack, obs)

			scores_list.append(score)
			
			break

		else:
			obs, frame_stack = stackFrames(frame_stack, obs)

		# either way, compile memory and add it to collection
		memory.append((state, action, reward))
		# set state for next iteration
		state = obs

	if success:
		break

	# after each episode, do training if more than 100 memories
	if len(memory) > 100:
		# first, separate memory into component data items
		batch = sampleMemory(memory, batch_size)
		actions = [item[1] for item in batch]
		states = np.array([item[0] for item in batch], ndmin=3)
		rewards = [item[2] for item in batch]
		next_states = np.array([item[0] for item in batch], ndmin=3)

		# generate expected outcomes for predictions
		predicts = model.predict(next_states)
		targets = [gamma * np.max(item) for item in predicts]
		targets = [targets[i] + rewards[i] for i in range(len(targets))]
		target_fit = [item for item in model.predict(states)]

		# populate labels with expected outcomes
		for i in range(len(target_fit)):
			target_fit[i][actions[i]] = targets[i]

		# format features and labels for training
		feats = np.array(states).reshape(-1, *state_space)
		lbls = np.array(target_fit).reshape(-1, action_space)
		# train the model on selected batch
		model.train_on_batch(x=feats, y=lbls)

	print(score)

if success:
	print("Success!")
else:
	print("Failure.")

print(scores_list)
makeGraph(scores_list)



#loop that goes throught al; the epidosdes
#inside that , loop through steps, for loops construction (2), inside step function pick random actions very step and apply to environmetn
#env.render smwhere
#reward state thingie 