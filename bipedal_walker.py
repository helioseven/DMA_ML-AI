import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import copy
from collections import deque
from my_util import argmax_4d, make_graph, sample_memory

# constants
episodes = 2#10000
max_steps = 1000
stack_size = 15

#learning_rate = 0.00025
learning_rate = 0.0001
decay_rate = 0.00001
min_epsilon = 0.01
max_epsilon = 1

batch_size = 64
gamma = 0.618

#step_size = 0.0625
step_size = 0.5

show_render = False
new_model = True

####################

scores = []
decay_step = 0

env = gym.make('BipedalWalker-v2')
obs = env.reset()

####################

# state_space is 24
state_space = obs.shape[0]

# possible_actions is a vector of all possible increments between -1 and 1 based on step_size
possible_actions = []
for i in range(int(2/step_size)+1):
    possible_actions.append(step_size*i-1)
nb_choices = len(possible_actions)

# action_space is 4 * (nb_choices) matrix of zeroes
action_space = np.array([[0]*nb_choices]*env.action_space.shape[0])

# action_codes is the (nb_choices) * (nb_choices) identity matrix
action_codes = np.identity(nb_choices, dtype=np.int)

# previous_prediction is (nb_choices) * (nb_choices) * (nb_choices) * (nb_choices) matrix of zeroes
prediction_space = np.array([[[[0.0]*nb_choices]*nb_choices]*nb_choices]*nb_choices)

####################

# Create model
def create_model(state_space, stack_size):
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(100, (1,8), input_shape=(state_space, stack_size, 1)),
                                 tf.keras.layers.Conv2D(75,(1,8)),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(1200, activation="relu"),
                                 tf.keras.layers.Dense(800, activation="relu"),
                                 tf.keras.layers.Dense(nb_choices**4, activation="softmax")
                                 ])
    opt = tf.keras.optimizers.Adam(lr=learning_rate,
                                beta_1=min_epsilon,
                                beta_2=max_epsilon,
                                decay=decay_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy")
    return model

def load_model():
    model = tf.keras.models.load_model("trained_model/bipedal_walker_model.h5")
    return model

# Stack the previous (stack_size) observations to give the neural network
# Stack: oldest --- newest
def stack_observations(obs, previous_stack):
    if previous_stack == None:
        new_stack = [np.zeros(obs.shape, dtype=np.int)]*stack_size
        new_stack[len(new_stack)-1] = obs
    else:
        new_stack = previous_stack
        del new_stack[0]
        new_stack.append(obs)

    obs_stack = np.stack(new_stack, axis=1)
    return obs_stack

# returns an action either randomly or based on model prediction
def predict_actions(model, obs_stack):
    nb_random = np.random.random()
    epsilon = max_epsilon + (min_epsilon - max_epsilon) * np.exp(-decay_rate * decay_step)

    if epsilon > nb_random:
        # generate random predictions
        predictions = random.randint(0, nb_choices-1, prediction_space.shape)
    else:
        # use model to predict
        feats = obs_stack.reshape(1, 24, stack_size, 1)
        predictions = model.predict(feats)

        # for each 16 numbers, find the argmax
        predictions = predictions.reshape(nb_choices, nb_choices, nb_choices, nb_choices)

    (i, j, k, l) = argmax_4d(predictions)
    return [action_codes[i], action_codes[j], action_codes[k], action_codes[l]]

####################

model = create_model(state_space, stack_size)

memory = deque(maxlen=1000)

for i in range(episodes):
    obs = env.reset()
    score = 0
    obs_stack = stack_observations(obs, None)

    for j in range(max_steps):
        if show_render:
            env.render()

        decay_step += 1
        action = predict_actions(model, obs_stack)

        obs, reward, done, _ = env.step(action)
        score += reward

        if done == True:
            obs_stack = stack_observations(np.zeros(state_space,), obs_stack)
            break
        else:
            obs_stack = stack_observations(obs, obs_stack)

        memory.append((obs_stack, action, reward))

        state = obs

        if len(memory) > 500:
            batch = sample_memory(memory, batch_size)
            states = np.array([item[0] for item in batch])

            states_np = states.reshape(batch_size, *(24, stack_size))
            next_states = copy.deepcopy(states_np)
            actions = [item[1] for item in batch]
            rewards = [item[2] for item in batch]

            #print(next_states.reshape(1, next_states.shape[0], next_states.shape[2], next_states.shape[1]).shape)
            next_states = next_states.reshape(*next_states.shape, 1)
            predicts = model.predict(next_states)

            targets = [gamma * np.max(item) for item in predicts]
            targets = [targets[i] + rewards[i] for i in range(len(targets))]
            states = states.reshape(*states.shape, 1)
            target_fit = [item for item in np.array(model.predict(states)).reshape(-1, 4, nb_choices)]

            for i in range(batch_size):
                code = argmax_4d(actions[i])
                for k in range(len(code)):
                    target_fit[i][k] = targets[i]

            feats = np.array(states).reshape(-1, 24, stack_size, 1)
            labels = np.array(target_fit).reshape(-1, 4*nb_choices)
            #print("Features: "+str(feats.shape))
            #print("Labels: "+str(labels.shape))
            #print(labels)
            model.train_on_batch(x=feats, y=labels)

    scores.append(score)
    print("Episode number: {}".format(i+1))
    print("Score: {}".format(score))

#TODO: figure out how to save model
#model.save("trained_model/bipedal_walker_model.h5")
fig = make_graph(scores)
# fig.savefig(fname="output.png")
plt.show()