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

env = gym.make("SpaceInvaders-v0")
state_space = env.observation_space
action_space = env.action_space.n
action_code = np.identity(action_space, dtype=int).tolist()

'''
env.reset()
obs, _, _, _ = env.step(env.action_space.sample())
result = processFrame(obs)
print(result)
print(type(result))
'''