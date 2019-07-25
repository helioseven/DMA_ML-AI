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

#constants
stack_size=6

# process a raw image frame by cropping, scaling, and normalizing(2)
def processFrame(frame):
	gray = rgb2gray(frame)
	cropped = gray[8:-12, 4:-12]
	normalized = cropped / 255.0
	return transform.resize(normalized, [210, 160])

#start using frames in episodes and stack them
def stackFrames(new_frame, stacked_frames, new episode):
	return_stack = stacked_frames
	#process a new frame
	frame = processFrame(new_frame)

	#if its the first frame of the episode, fill the stack with the first frame
	if new_episode:
		for _ in range(stack_size):
			return_stack.append(frame)
	#else just add the current frame to the stack
	else:
		return_stack.append(frame)

	return return_stack

#create gym environment.    - basic structure(1)
env = gym.make("Breakout-v0")
state_space = (210, 160, 3)
action_space = env.action_space.n

#check what k value it works w  - uses 4
	#print(action_space)

#build model and memory
model = getModel()
memory = deque(maxlen=1000)

for episode in range(1000):
	env.reset()
	score = 0

	for step in range(500):
		env.render()
		#generate random action
		action = env.action_space.sample()
		#apply the action to step env
		obs, reward, done, _ = env.step(action) #basic structure

		#add reward to score
		score =+ reward

		if done ==True:
			obs = np.zeros((210, 160))
			obs, frame_stack = stackFrames(frame_stack, obs, False)
		else:
			obs, frame_stack = stackFrames((frame_stack, obs, False))








#loop that goes throught al; the epidosdes
#inside that , loop through steps, for loops construction (2), inside step function pick random actions very step and apply to environmetn
#env.render smwhere
#reward state thingie 