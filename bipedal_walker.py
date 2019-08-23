import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import copy
from collections import deque
from my_util import argmax_4d, make_graph, sample_memory

# constants
episodes = 100000
max_steps = 1000
stack_size = 15

learning_rate = 0.00025
decay_rate = 0.00001
min_epsilon = 0.01
max_epsilon = 1

batch_size = 64
gamma = 0.618

step_size = 0.5#0.0625

show_render = True
save_figure = True
show_figure = True
new_model = True

####################

env = gym.make('BipedalWalker-v2')
obs = env.reset()

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

# prediction_space is (nb_choices) * (nb_choices) * (nb_choices) * (nb_choices) matrix of zeroes
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

def train_model():
    batch = sample_memory(memory, batch_size)
    states = np.array([item[0] for item in batch]).reshape(batch_size, state_space, stack_size, 1)
    actions = [item[1] for item in batch]
    rewards = [item[2] for item in batch]
    predicts = model.predict(states).reshape(batch_size, *(prediction_space.shape))

    # targets are a composite of the expected and actual rewards for the actual action taken
    # for each batch item, we first get the max predicted reward from all currently possible actions and multiply by gamma
    targets = [gamma * np.max(item) for item in predicts]
    # we then add to that the actual reward gotten from the actual action taken
    targets = [targets[i] + rewards[i] for i in range(batch_size)]

    # lastly, we copy the predictions and update the reward for the actual action taken with the target
    target_fit = copy.deepcopy(predicts)
    for i in range(batch_size):
        a, b, c, d = argmax_4d(target_fit[i])
        target_fit[i][a][b][c][d] = targets[i]
        # the idea here is that target_fit is mostly just model predictions,
        # with one value in the prediction matrix being updated to target for each batch item.
        # when fed to the model for training, the model predictions already fit the model (by definition),
        # so the only data point in target_fit that affects the model is the target

    feats = np.array(states).reshape(batch_size, state_space, stack_size, 1)
    labels = np.array(target_fit).reshape(batch_size, nb_choices**4)
    result = model.train_on_batch(x=feats, y=labels)

####################

# Stack the previous (stack_size) observations to give the neural network
# Stack: oldest --- newest
def stack_observations(obs, previous_stack):
    if previous_stack is None:
        new_stack = [np.zeros(obs.shape, dtype=np.int)]*stack_size
        new_stack[len(new_stack)-1] = obs
    else:
        prev_stack = np.stack(previous_stack, axis=-1)
        new_stack = prev_stack.tolist()[1:]
        new_stack.append(obs)

    obs_stack = np.stack(new_stack, axis=1)
    return obs_stack

# returns an action either randomly or based on model prediction
def predict_action():
    nb_random = np.random.random()
    epsilon = max_epsilon + (min_epsilon - max_epsilon) * np.exp(-decay_rate * decay_step)

    if epsilon > nb_random:
        # generate random predictions
        predictions = np.random.randint(0, nb_choices-1, prediction_space.shape)
    else:
        # use model to predict
        feats = obs_stack.reshape(1, state_space, stack_size, 1)
        predictions = model.predict(feats)
        predictions = predictions.reshape(*(prediction_space.shape))

    return argmax_4d(predictions)

####################

scores = []
decay_step = 0

model = create_model(state_space, stack_size)

memory = deque(maxlen=10000)

for i in range(episodes):
    obs = env.reset()
    score = 0
    obs_stack = stack_observations(obs, None)

    for j in range(max_steps):
        is_tenth_ep = i % 10 == 0
        if show_render and is_tenth_ep:
            env.render()

        decay_step += 1
        a, b, c, d = predict_action()
        action_indices = (a, b, c, d)
        action = [possible_actions[a], possible_actions[b], possible_actions[c], possible_actions[d]]

        obs, reward, done, _ = env.step(action)
        score += reward

        if done == True:
            obs_stack = stack_observations(np.zeros(state_space,), obs_stack)
            break
        else:
            obs_stack = stack_observations(obs, obs_stack)

        memory.append((obs_stack, action_indices, reward))

        state = obs

    scores.append(score)
    print("Episode number: {}".format(i+1))
    print("Episode score: {}".format(score))

    if len(memory) > 500:
        train_model()

#TODO: figure out how to save model
#model.save("trained_model/bipedal_walker_model.h5")
fig = make_graph(scores)
if save_figure:
    fig.savefig(fname="output.png")
if show_figure:
    plt.show()