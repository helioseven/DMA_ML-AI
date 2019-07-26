import gym
import pandas as pd
import numpy as np
from collections import deque
import random

import skimage
from skimage import transform
from skimage.color import rgb2gray

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, Activation, Input

def processFrame(frame):
	gray = rgb2gray(frame)
	cropped = gray[8:-12, 4:-12]
	normalized = cropped/255.0
	return transform.resize(normalized, [180, 120])


def getModel():
	model = Sequential()
	model.add(Flatten())
	model.add(Dense(50, activation="relu"))
	model.add(Dense(25, activation="relu"))
	model.add(Dense(10, activation="relu"))
	model.add(Dense(action_space, activation="softmax"))
	opt = keras.optimizers.Adam(lr=learning_rate,
								beta_1=min_epsilon,
								beta_2=max_epsilon,
								decay=decay_rate)
	model.compile(optimizer=opt,
				  loss="categorical_crossentropy")
	return model
###a bunch of functions
#news frames in episdoes
def stackFrames(stacked_frames, frame):
	return_stack = stacked_frames
	return_state = None

	frame = processFrame(frame)
	return_stack.append(frame)
	# if first frame of the episode, fill the stack with frame
	
	# build our return state, and return it with the stack
	return_state = np.stack(return_stack, axis=2)
	return return_state, return_stack

def predictAction(model, decay_step):
	tradeoff = np.random.random()
	# update epsilon based on decay_step
	epsilon = max_epsilon \
				   + (min_epsilon - max_epsilon) \
				   * np.exp(-decay_rate * decay_step)

	# if random number is less than tradeoff, generate random action
	if epsilon > tradeoff:
		choice = random.randint(1, len(action_codes)) - 1
	else:
		# reshape frame_stack to model's desired input shape
		feats = np.array(frame_stack).reshape(1, *state_space)
		predicts = model.predict(feats)
		choice = np.argmax(predicts)

	# return the action code associated with choice made
	return action_codes[choice]

#the memory stuff and you're playing the game for that number of episode
# lots of constants
env = gym.make("Tennis-v0")

stack_size = 4
total_episodes = 10000
max_steps = 250
batch_size = 64
learning_rate = 0.00025
gamma = 0.618
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.00001
decay_step = 0
action_space = env.action_space.n

# generating a frame stack filled with empty (zeros) images
blank_frames = [np.zeros((180, 120), dtype=np.int) \
					   for i in range(stack_size)]
frame_stack = deque(blank_frames, maxlen = stack_size)
state_space = (180, 120, stack_size)

model = getModel()
memory = deque(maxlen=1000)
# main code
# start by building gym environment
# ideally state_space is not hard-coded, but it'll do for now

action_codes = np.identity(action_space, dtype=np.int).tolist()




for episode in range(total_episodes):
	print("Episode number: {}".format(episode))

	# reset environment, initialize variables
	state = env.reset()
	score = 0
	state, frame_stack = stackFrames(frame_stack, state)

	# iterate through steps in the episode
	for step in range(max_steps):
		# render, so we can watch
		env.render()
		# increment decay_step to update epsilon
		decay_step += 1

		# generate an action
		action = np.argmax(predictAction(model, decay_step))

		# apply action to step the environment
		obs, reward, done, _ = env.step(action)

		# add received reward to episode score
		score += reward

		# if the last frame of the episode,
		# append empty frame to frame_stack
		# and append score to scores_list
		if done == True:
			obs = np.zeros((8, ))
			obs, frame_stack = stackFrames(frame_stack, obs)
		# otherwise, simply add current frame to frame_stack
		else:
			obs, frame_stack = stackFrames(frame_stack, obs)

		# either way, compile memory and add it to collection
		memory.append((state, action, reward, obs, done))
		# set state for next iteration
		state = obs

###end of memory stuff



model = getModel()
memory = deque(maxlen=1000)






print("All done!")
