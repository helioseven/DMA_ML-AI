import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten

def main():
	# generate data
	f_data, l_data = generate_data(10, 10, 0)

	# build model and train it
	model = getModel()
	model.fit(f_data, l_data, epochs=2, verbose=1)

	# play games with random and model agents, print results
	print("playing game with random agent")
	play_game(10, 50)
	print("playing game with model agent")
	play_game(10, 50, model)

def getStateSize():
	state = env.reset()
	action = env.action_space.sample()
	obs, _, _, _ = env.step(action)
	return len(obs)

def getModel():
	model = Sequential()
	model.add(Dense(100, input_shape=(210, 160, 3)))
	model.add(Dropout(0.2))
	model.add(Dense(50, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(25, activation="relu"))
	model.add(Flatten())
	model.add(Dropout(0.8))
	model.add(Dense(6, activation="softmax"))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	return model

def generate_data(number_of_games, max_turns, threshold_score):
	# variable initializations for data generation
	feat_data = []
	lbl_data = []
	l_1hot = [0 for i in range(n_actions)]

	# loop through number of games specified
	for i in range(number_of_games):
		# reset gym environment
		env.reset()
		# variable initiailizations for each playthrough
		game_memory = []
		prev_obs = []
		score = 0

		# loop through gym steps until max_turns is reached
		for turn in range(max_turns):
			# pick an action at random
			action = env.action_space.sample()
			# step environment with chosen action
			obs, reward, done, info = env.step(action)
			# add reward to current score for this playthrough
			score += int(reward)

			# if not the first turn, append game state / action
			# pair to game_memory list
			if turn > 0:
				game_memory.append([prev_obs, int(action)])
			prev_obs = obs

			# if simulation completes successfully, break
			if done == True:
				break

		# for each playthrough, record obs/action pairs for
		# steps on which score was above the threshold
		if score >= threshold_score:
			for item in game_memory:
				label = np.array(list(l_1hot))
				label[item[1]] = 1

				# append each feature and label to respective lists
				feat_data.append(item[0])
				lbl_data.append(label)

	# print number of games played through and return data
	print("{} examples were made.".format(len(feat_data)))
	return np.array(feat_data), np.array(lbl_data)

def play_game(number_of_games, number_of_moves, model=None):
	for i in range(number_of_games):
		print(i)
		state = env.reset()
		score = 0
		for step in range(number_of_moves):
			# start with a non-action
			action = None
			if model == None:
				# if no model was passed, choose a random action
				action = env.action_space.sample()
			else:
				# if a model was passed, use it to choose action
				state = state.reshape(1, 210, 160, 3)
				action = np.argmax(model.predict(state))

			# record results of action
			obs, reward, done, info = env.step(action)
			score += int(reward)
			state = obs

			# if any playthrough completes, print score and return
			if done == True:
				print("Score: {}".format(score))
				break

# setting up our gym environment
env = gym.make("SpaceInvaders-v0")
env.reset()
# variable initializations
n_actions = env.action_space.n
n_states = getStateSize()

main()