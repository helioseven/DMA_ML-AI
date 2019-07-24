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

env = gym.make("SpaceInvaders-v0")
state_space = env.observation_space
action_space = env.action_space.n
action_code = np.identity(action_space, dtype=int).tolist()

stack_size = 4
blank_imgs = [np.zeros((110, 84), dtype=np.int) \
					   for i in range(stack_size)]
frame_stack = deque(blank_imgs, maxlen = stack_size)

'''
# testing
env.reset()
obs, _, _, _ = env.step(env.action_space.sample())
frame = processFrame(obs)
result1, result2 = stackFrames(frame_stack, frame, True)
print(result1)
print(type(result1))
print(result2)
print(type(result2))
obs, _, _, _ = env.step(env.action_space.sample())
frame = processFrame(obs)
result1, result2 = stackFrames(frame_stack, frame, False)
print(result1)
print(type(result1))
print(result2)
print(type(result2))
'''