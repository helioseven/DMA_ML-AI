import numpy as np
import gym
import random

# start by creating the gym environment
env = gym.make("Taxi-v2")
# tons of variable initializations
n_actions = env.action_space.n
n_states = env.observation_space.n
q_table = np.zeros((n_states, n_actions))
success = False
threshold = 15.0
total_episodes = 50000
total_test_episodes = 100
max_steps = 99
learning_rate = 0.7
gamma = 0.618
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# iterate through episodes
for episode in range(total_episodes):
	state = env.reset()
	step = 0
	score = 0
	done = False

	# iterate through steps for a given episode
	for step in range(max_steps):
		# generate a tradeoff variable, randomly
		tradeoff = random.uniform(0, 1)

		# compare tradeoff against epsilon
		if tradeoff > epsilon:
			# return action from the QTable
			action = np.argmax(q_table[state,:])
		else:
			# return random action
			action = env.action_space.sample()

		# step the simulation, store results
		new_state, reward, done, _ = env.step(action)
		# render simulation
		env.render()

		# add reward from last action to episode score
		score += reward

		# update the QTable according to Q-algorithm
		q_table[state, action] = q_table[state, action] \
								 + learning_rate \
								 * (reward + gamma \
								 * np.max(q_table[new_state,:]) \
								 - q_table[state, action])

		# store new_state into previous state for next iteration
		state = new_state
		# if score threshold has been reached, break
		if score >= threshold:
			success = True
			break
		# if simulation is over, break
		if done == True:
			break

	# after episode is finished, print episode number
	print(episode)
	# if success, break
	if success == True:
		break

	# update epsilon for next episode
	epsilon = min_epsilon \
	+ (max_epsilon - min_epsilon) \
	* np.exp(-decay_rate * episode)

# if success when we're done, print success message
if success == True:
	print("Success!")