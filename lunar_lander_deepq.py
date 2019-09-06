import random
from os.path import isfile
from shutil import copyfile
from collections import deque
import numpy as np
import gym

from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense
from keras.optimizers import Adam

from my_util import makeAnimation, makeGraph, sampleMemory

# constants
success = False
stack_size = 32
batch_size = 512
total_episodes = 100000
max_steps = 500
max_memories = batch_size * max_steps
gamma = 0.9
max_epsilon = 1.0
min_epsilon = 0.001
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
	if isfile("lunar_lander_dqn.h5"):
		copyfile("lunar_lander_dqn.h5", "lunar_lander_dqn_old.h5")
		model = load_model("lunar_lander_dqn.h5")
	else:
		model = Sequential()
		model.add(LSTM(units=32, return_sequences=True, input_shape=state_space))
		model.add(LSTM(units=64))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(16, activation='relu'))
		model.add(Dense(action_space, activation='softmax'))
		model.compile(optimizer=Adam(lr=0.00025), loss='categorical_crossentropy')
	return model

# performs model training
def trainModel():
	# first, separate memory into component data items
	batch = sampleMemory(memory, batch_size)
	states = np.array([item[0] for item in batch]).reshape(batch_size, *state_space)
	new_states = np.array([item[1] for item in batch]).reshape(batch_size, *state_space)
	actions = [item[2] for item in batch]
	rewards = [item[3] for item in batch]
	is_done = [item[4] for item in batch]

	# generate expected rewards for selected states
	target_fit = model.predict(states)
	predicts = model.predict(new_states)
	expects = [gamma * np.max(item) for item in predicts]
	targets = [(0.0 if is_done[i] else expects[i]) + rewards[i] for i in range(batch_size)]

	# populate labels with expected outcomes
	for i in range(batch_size):
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
	# update epsilon based on decay_step and constants
	epsilon = max_epsilon + \
			  (min_epsilon - max_epsilon) * \
			  np.exp(-5 * decay_step / total_episodes)

	# in early training, generate mostly random moves
	if epsilon < tradeoff:
		choice = random.randint(1, action_space) - 1
	# as epsilon decays towards 1, more moves are based on predicted rewards
	else:
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
memory = deque(maxlen=max_memories)
scores_list = []

for episode in range(total_episodes):
	# increment decay_step to update epsilon
	decay_step += 1
	score = 0.0
	actions_taken = []

	# generate new random seed and store it for possible future use
	seed = env.seed()[0]
	# reset environment, stack start state
	obs = env.reset()
	state = stackFrames(None, obs)

	# iterate through steps in the episode
	for step in range(max_steps):
		if episode % 100 == 0:
			# render every 100th episode to window, so we can watch
			env.render()

		# generate an action
		code = predictAction(state)
		action = np.argmax(code)
		# track actions each episode for terminal output
		actions_taken.append(action)

		# apply action to step the environment
		obs, reward, done, _ = env.step(action)

		# add received reward to episode score
		score += reward
		# finish episode if score is below arbitrary threshold
		if score < -200.0:
			done = True
		# finish episode if score is above winning threshold
		if score >= 200.0:
			success = True
			done = True

		# compile memory and add it to collection
		new_state = stackFrames(state, obs)
		memory.append((state, new_state, action, reward, done))

		# after adding memory, break if done
		if done:
			break

		# update state
		state = new_state

	scores_list.append(score)
	print("Score for episode {}: {}".format(episode, score))

	# added some bits to make a .gif of the successful episode
	if success:
		renders = []
		# use stored seed to guarantee determinism
		env.seed(seed)
		# reset env and store first frame
		env.reset()
		renders.append(env.render(mode='rgb_array'))
		for action in actions_taken:
			# then just step through the loop
			env.step(action)
			renders.append(env.render(mode='rgb_array'))
		# pass the array to a utility function to generate sequence.gif
		makeAnimation(renders)
		break

	# after each episode, train model if sufficient memories exist
	if len(memory) > batch_size * 4:
		trainModel()

# save model model only when finished (could add checkpoints)
model.save("lunar_lander_dqn.h5")
# uses utility function to create graph image and save as output.png
makeGraph(scores_list)

if success:
	print("We win!")
else:
	print("We lose...")