import random
import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten



#create gym environment.    10-25 basic structure
env = gym.make("Breakout-v0")
state_space = (210, 160, 3)
action_space = env.action_space.n

print(action_space)

for episode in range(1000):
	env.reset()

	for step in range(500):
		env.render()
		#generate random action
		action = env.action_space.sample()
		#apply the action to step env
		obs, _, _, _ = env.step(action)






#loop that goes throught al; the epidosdes
#inside that , loop through steps, for loops contsution (2), inside step function pick random actions very step and apply to environmetn
#env.render smwhere
#reward state thingie 