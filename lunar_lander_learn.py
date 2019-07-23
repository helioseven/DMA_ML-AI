import random
import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten

def main():
	model = getModel()
	high_score = 0
	features = []
	labels = []

	i = 1
	while high_score < 200:
		print("No success yet, generating more data.")
		f_data, l_data, score = generate_data(10, 500, model)

		if len(f_data) < 1:
			continue
		if score >= 200:
			break

		for item in f_data:
			features.append(item)
		for item in l_data:
			labels.append(item)

		if score > high_score:
			high_score = score
		i += 1
		learn_rate = 1 / i

		model.fit(np.array(features), np.array(labels), epochs=1, verbose=1)

	print("Success!")

def getStateShape():
	state = env.reset()
	action = env.action_space.sample()
	obs, reward, _, _ = env.step(action)
	init_reward = reward
	return np.array(obs).shape

def getModel():
	model = Sequential()
	model.add(Dense(100, input_shape=nd_states))
	model.add(Dropout(0.2))
	model.add(Dense(50, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(25, activation="relu"))
	model.add(Dropout(0.8))
	model.add(Dense(4, activation="softmax"))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	return model

def generate_data(number_of_games, max_turns, model):
	# variable initializations for data generation
	feat_data = []
	lbl_data = []
	l_1hot = [0 for i in range(n_actions)]
	best_score = 0

	# loop through number of games specified
	for i in range(number_of_games):
		# reset gym environment
		state = env.reset()
		# variable initiailizations for each playthrough
		game_memory = []
		score = 0

		# loop through gym steps until max_turns is reached
		for turn in range(max_turns):
			if random.random() >= learn_rate:
				# if a model was passed, use it to choose action
				state = np.array(state).reshape(1, nd_states[0])
				action = np.argmax(model.predict(state))
			else:
				# pick an action at random
				action = env.action_space.sample()
			# step environment with chosen action
			obs, reward, done, _ = env.step(action)
			env.render()
			# add reward to current score for this playthrough
			score += int(reward)

			# if not the first turn, append game state / action
			# pair to game_memory list
			if turn > 0:
				game_memory.append([state, int(action), reward])
			state = obs

			# if simulation completes successfully, break
			if done == True:
				break

		# for each playthrough, record obs/action pairs for
		# steps on which score was above the threshold
		if score > best_score:
			best_score = score

#		prev_reward = init_reward
		for item in game_memory:
			if item[2] >= 25.0:
#				print(item[2])
				label = np.array(list(l_1hot))
				label[item[1]] = 1

				# append each feature and label to respective lists
				feat_data.append(item[0])
				lbl_data.append(label)

#			prev_reward = item[2]

	# print number of games played through and return data
	print("{} examples were made.".format(len(feat_data)))
	return feat_data, lbl_data, best_score

# setting up our gym environment
env = gym.make("LunarLander-v2")
env.reset()
# variable initializations
# init_reward = 0.0
n_actions = env.action_space.n
nd_states = getStateShape()
learn_rate = 1.0

main()