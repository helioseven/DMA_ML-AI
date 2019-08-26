import gym
import pandas as pd
import numpy as np
from collections import deque
import random
import copy

import keras
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers.core import Dense, Dropout, Flatten

from my_util import makeGraph, sampleMemory

# constants
success = False
stack_size = 16
total_episodes = 100000
max_steps = 250
max_memory = int(2.5 * total_episodes)
batch_size = 256
learning_rate = 0.0025
gamma = 0.618
max_epsilon = 1.0
min_epsilon = 0.001
decay_rate = 10 / total_episodes
decay_step = 0

# environment setup
env = gym.make("LunarLander-v2")

# figure out size of state and action spaces
state_space = (stack_size, 8)
action_space = env.action_space.n
# generate an array of all possible action codes (1-hot encoding)
action_codes = np.identity(action_space, dtype=np.int).tolist()

### functions ###

# takes a possibly None stack of frames and a new frame and stacks
def stackFrames(stacked_frames, frame):
	# if first frame of the episode, fill the stack with frame
	if stacked_frames is None:
		prev_stack = []
		for _ in range(stack_size):
			prev_stack.append(frame)
	# otherwise, shorten existing stack by one and add frame once
	else:
		prev_stack = stacked_frames.tolist()[1:]
		prev_stack.append(frame)

	# stackify and return
	stack = np.stack(prev_stack)
	return stack

# defines our keras model for Deep-Q training
def getModel():
	model = Sequential()
	model.add(SimpleRNN(units=64, input_shape=state_space))
	model.add(Dense(48, activation="relu"))
	model.add(Dense(24, activation="relu"))
	model.add(Dense(8, activation="relu"))
	model.add(Dense(action_space, activation="softmax"))
	opt = keras.optimizers.Adam(lr=learning_rate,
								beta_1=min_epsilon,
								beta_2=max_epsilon,
								decay=decay_rate)
	model.compile(optimizer=opt,
				  loss="categorical_crossentropy")
	return model

# performs model training
def trainModel():
	# first, separate memory into component data items
	batch = sampleMemory(memory, batch_size)
	states = np.array([item[0] for item in batch])
	states = states.reshape(batch_size, *state_space)
	actions = [item[1] for item in batch]
	rewards = [item[2] for item in batch]

	# generate expected rewards for selected states
	predicts = model.predict(states)
	targets = [gamma * np.max(item) for item in predicts]
	targets = [targets[i] + rewards[i] for i in range(len(targets))]
	target_fit = copy.deepcopy(predicts)

	# populate labels with expected outcomes
	for i in range(len(target_fit)):
		target_fit[i][actions[i]] = targets[i]

	# format features and labels for training
	feats = np.array(states).reshape(batch_size, *state_space)
	lbls = np.array(target_fit).reshape(batch_size, action_space)
	# train the model on selected batch
	model.train_on_batch(x=feats, y=lbls)

# generate a new action based on either random or prediction
def predictAction(frame_stack):
	# random number to compare to epsilon
	tradeoff = np.random.random()
	# update epsilon based on decay_step
	epsilon = max_epsilon \
			  + (min_epsilon - max_epsilon) \
			  * np.exp(-decay_rate * decay_step)

	if epsilon > tradeoff:
		# in early training, generate mostly random moves
		choice = random.randint(1, len(action_codes)) - 1
	else:
		# as epsilon decays, more moves are based on predicted rewards
		# first, reshape frame_stack to model's desired input shape
		feats = frame_stack.reshape(1, *state_space)
		# generate predictions based on frame_stack features
		predicts = model.predict(feats)
		# generate a choice (index) based on predicted rewards
		choice = np.argmax(predicts)

	# return the action code associated with choice made
	return action_codes[choice]

### main code ###

# build model, and create memory collection
model = getModel()
memory = deque(maxlen=max_memory)
scores_list = []

for episode in range(total_episodes):
	# reset environment, initialize variables
	obs = env.reset()
	score = 0.0
	state = stackFrames(None, obs)

	# iterate through steps in the episode
	for step in range(max_steps):
		if episode % 100 == 0:
			# render every 100th episode to window, so we can watch
			env.render()
		# increment decay_step to update epsilon
		decay_step += 1

		# generate an action
		action = np.argmax(predictAction(state))

		# apply action to step the environment
		obs, reward, done, _ = env.step(action)

		# add received reward to episode score
		score += reward
		success = score >= 200.0

		# if the last frame of the episode, flag success if so,
		# and append empty frame to state
		if done == True or success:
			obs = np.zeros((8,))
			state = stackFrames(state, obs)
			break
		# otherwise, simply add current frame to state
		else:
			state = stackFrames(state, obs)

		# either way, compile memory and add it to collection
		memory.append((state, action, reward))

	scores_list.append(score)

	if success:
		break

	# after each episode, do training if more than 500 memories
	if len(memory) > 500:
		trainModel()

	print("Score for episode {}: {}".format(episode, score))

makeGraph(scores_list)

if success:
	print("We win!")
else:
	print("We lose...")