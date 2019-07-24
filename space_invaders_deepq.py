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
	normalized = cropped / 255.0
	return transform.resize(normalized, [110, 84])

def stackFrames(stacked_frames, new_frame, new_episode):
	return_stack = stacked_frames
	return_state = None
	frame = processFrame(new_frame)

	if new_episode:
		for _ in range(stack_size):
			return_stack.append(frame)
	else:
		return_stack.append(frame)

	return_state = np.stack(return_stack, axis=2)
	return return_state, return_stack

def predictAction(model, decay_step, state):
	tradeoff = np.random.random()
	epsilon = max_epsilon \
				   + (min_epsilon - max_epsilon) \
				   * np.exp(-decay_rate * decay_step)

	if epsilon > tradeoff:
		choice = random.randint(1, len(action_codes)) - 1
	else:
		feats = np.array(frame_stack).reshape(1, *state_space)
		choice = np.argmax(model.predict(feats))

	return action_codes[choice]

def sampleMemory(buffered_list, batch_size):
	buffer_size = len(buffered_list)
	index = np.random.choice(np.arange(buffer_size),
							 size=batch_size,
							 replace=False)
	return [buffered_list[i] for i in index]

def getModel():
	model = Sequential()
	model.add(Input(shape=state_space))
	model.add(Conv2D(200, (60, 60)))
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

env = gym.make("SpaceInvaders-v0")
# state_space = env.observation_space
state_space = (4, 110, 84)
action_space = env.action_space.n
action_codes = np.identity(action_space, dtype=np.int).tolist()

stack_size = 4
blank_imgs = [np.zeros((110, 84), dtype=np.int) \
					   for i in range(stack_size)]
frame_stack = deque(blank_imgs, maxlen = stack_size)
learning_rate = 0.00025
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.00001

'''
# testing
'''