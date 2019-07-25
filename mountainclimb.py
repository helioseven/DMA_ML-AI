import gym
import random
import numpy as np
import tflearn
import pandas as pd
from collections import deque

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, Activation, Input

def stackFrames(stacked_frames, frame, new_episode):
	return_stack = stacked_frames
	return_state = None

	if new_episode:
		for _ in range(stack_size):
			return_stack.append(frame)
	else:
		return_stack.append(frame)

	return_state = np.stack(return_stack, axis=1)
	return return_state, return_stack

def predictAction(model, decay_step):
	tradeoff = np.random.random()
	epsilon = max_epsilon + (min_epsilon - max_epsilon) *np.exp(-decay_rate * decay_step)

	if epsilon > tradeoff:
		choice = random.randint(1, len(action_codes)) - 1
	else:
		feats = np.array(frame_stack).reshape(1, *state_space)
		predicts = model.predict(feats)
		choice = np.argmax(predicts)

	return action_codes[choice]

	def sampleMemory(buffered_list, batch_size):
		buffer_size = len(buffered_list)
		# generates a list of random indices to pick
		index = np.random.choice(np.arange(buffer_size),
								 size=batch_size,
								 replace=False)
		return [buffered_list[i] for i in index]

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
env = gym.make("MountainCarContinuous-v0")
# ideally state_space is not hard-coded, but it'll do for now
state_space = (2, 4)

action_space = 9
action_codes = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
# action_codes = np.identity(action_space, dtype=np.int).tolist()

# lots of constants
success = False
stack_size = 4
total_episodes = 10000
max_steps = 250
batch_size = 64
learning_rate = 0.00025
gamma = 0.618
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001
decay_step = 0
# generating a frame stack filled with empty (zeros) images
blank_frames = [np.zeros((2, ), dtype=np.int) \
					   for i in range(stack_size)]
frame_stack = deque(blank_frames, maxlen = stack_size)

# build model, and create memory collection
model = getModel()
memory = deque(maxlen=1000)
for episode in range(total_episodes):
	# reset environment, initialize variables
	state = env.reset()
	score = 0
	state, frame_stack = stackFrames(frame_stack, state, True)

	# iterate through steps in the episode
	for step in range(max_steps):
		rewards_list = []
		# render, so we can watch
		env.render()
		# increment decay_step to update epsilon
		decay_step += 1

		# generate an action
		action = np.array([predictAction(model, decay_step)])

		# apply action to step the environment
		obs, reward, done, _ = env.step(action)

		# NEW CODE
		reward = np.sin(3 * obs[0])*.45+.55
		# END NEW CODE

		# add received reward to episode score
		score += reward

		# if the last frame of the episode,
		# append empty frame to frame_stack
		# and append score to scores_list
		if done == True:
			if obs[0] >= 0.45:
				success = True
				break
			obs = np.zeros((2, ))
			obs, frame_stack = stackFrames(frame_stack, obs, False)
		# otherwise, simply add current frame to frame_stack
		else:
			obs, frame_stack = stackFrames(frame_stack, obs, False)

		choice = np.where(action_codes == action)
		# either way, compile memory and add it to collection
		memory.append((state, choice, reward))
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
			target_fit[i][actions[i][0]] = targets[i]

		# format features and labels for training
		feats = np.array(states).reshape(-1, *state_space)
		lbls = np.array(target_fit).reshape(-1, action_space)
		# train the model on selected batch
		model.train_on_batch(x=feats, y=lbls)

	#print("Score for episode {}: {}".format(episode, score))
	print(str(episode) + ", " + str(score))

if success:
	print("success")
else:
	print("failure")