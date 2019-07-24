import gym
import pandas as pd
import numpy as np
from collections import deque
import random

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, Activation, Input

# if first frame of episode, fills stack with new_frame
# if not first frame, adds new_frame to existing stack
def stackFrames(stacked_frames, frame, new_episode):
	return_stack = stacked_frames
	return_state = None

	# if first frame of the episode, fill the stack with frame
	if new_episode:
		for _ in range(stack_size):
			return_stack.append(frame)
	# otherwise, add frame to stack once
	else:
		return_stack.append(frame)

	# build our return state, and return it with the stack
	return_state = np.stack(return_stack, axis=1)
	return return_state, return_stack

# generate a new action based on either random or prediction
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

# generates a random sampling of the list passed in
def sampleMemory(buffered_list, batch_size):
	buffer_size = len(buffered_list)
	# generates a list of random indices to pick
	index = np.random.choice(np.arange(buffer_size),
							 size=batch_size,
							 replace=False)
	return [buffered_list[i] for i in index]

# defines our keras model for Deep-Q training
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

# main code
# start by building gym environment
env = gym.make("LunarLander-v2")
# ideally state_space is not hard-coded, but it'll do for now
state_space = (8, 4)
action_space = env.action_space.n
action_codes = np.identity(action_space, dtype=np.int).tolist()

# lots of constants
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
# generating a frame stack filled with empty (zeros) images
blank_frames = [np.zeros((110, 84), dtype=np.int) \
					   for i in range(stack_size)]
frame_stack = deque(blank_frames, maxlen = stack_size)
# a few other initializations
terminal_Qs_batch = []
rewards_list = []

# build model, and create memory collection
model = getModel()
memory = deque(maxlen=1000)
for episode in range(total_episodes):
	print("Episode number: {}".format(episode))

	# reset environment, initialize variables
	state = env.reset()
	score = 0
	state, frame_stack = stackFrames(frame_stack, state, True)

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
			obs, frame_stack = stackFrames(frame_stack, obs, False)
			rewards_list.append(score)
		# otherwise, simply add current frame to frame_stack
		else:
			obs, frame_stack = stackFrames(frame_stack, obs, False)

		# either way, compile memory and add it to collection
		memory.append((state, action, reward, obs, done))
		# set state for next iteration
		state = obs

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

	print("Score for episode {}: {}".format(episode, score))

print("All done!")