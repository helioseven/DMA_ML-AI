import numpy as np
import random

import gym
import tensorflow as tf

import copy
from collections import deque

# Stack the previous (stack_size) observations to give the neural network
# Stack: oldest --- newest
def stack_observations(obs, previous_stack, stack_size, new_episode):
    if previous_stack == None and not new_episode:
        raise Exception('"None" input was unexpected: please enter a previous stack or set new_episode to True')
    if new_episode:
        new_stack = [np.zeros(obs.shape, dtype=np.int) for i in range(stack_size)]
        new_stack[len(new_stack)-1] = obs
    else:
        new_stack = previous_stack
        del new_stack[0]
        new_stack.append(obs)

    state = np.stack(new_stack, axis=1)
    return new_stack, state

# Create model
def create_model(nb_action, state_space, stack_size, learning_rate, min_epsilon, max_epsilon, decay_rate):
    model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(100, activation="relu", input_shape=(1, state_space*stack_size)),
                                 tf.keras.layers.Dense(50, activation="relu"),
                                 tf.keras.layers.Dense(10, activation="relu"),
                                 tf.keras.layers.Dense(len(np.array(nb_action).flatten()), activation="softmax")])
    opt = tf.keras.optimizers.Adam(lr=learning_rate,
                                beta_1=min_epsilon,
                                beta_2=max_epsilon,
                                decay=decay_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy")
    return model

def predict_action(model, env, obs_stack, nb_action, action_codes, max_epsilon, min_epsilon, decay_rate, decay_step, state_space, stack_size):
    nb_random = np.random.random()
    epsilon = max_epsilon + (min_epsilon - max_epsilon) * np.exp(-decay_rate * decay_step)

    if epsilon > nb_random:
        # make input random

        prediction = np.array(nb_action)

        for i in range(len(nb_action)):
            for j in range(len(nb_action[i])):
                prediction[i][j] = random.randint(1, len(nb_action[i]))-1
    else:
        # use model to predict
        feats = np.array(obs_stack).reshape(1, state_space*stack_size)
        '''
        rewards = model.predict(feats)
        choice1 = np.argmax(rewards[0])
        ...
        choice4 = np.argmax(rewards[3])
        choice = [choice1, ..., choice4]
        '''

        ## TODO: Break actions down to increments, and then predict rewards based on that
        # for each 16 numbers, find the argmax
        prediction_flat = model.predict(feats)[0]
        prediction = nb_action
        l = 0
        for i in range(len(nb_action)):
            for j in range(len(nb_action[i])):
                prediction[i][j] = prediction_flat[l]
                l+=1
    choice = []
    for i in range(len(nb_action)):
        choice.append(np.argmax(prediction[i]))
    output = []
    for i in range(len(choice)):
        output.append(action_codes[i][int(choice[i])])
    return output


####################
# constants
episodes = 10000
max_steps = 1000
stack_size = 4

learning_rate = 0.00025
decay_rate = 0.00001
min_epsilon = 0.01
max_epsilon = 1

batch_size = 64
gamma = 0.618

step_size = 0.125

possible_actions = []
for i in range(int(2/step_size)+1):
    possible_actions.append(step_size*i-1)
print(possible_actions)
####################
scores = []
decay_step = 0

env = gym.make('BipedalWalker-v2')
obs = env.reset()

state_space = obs.shape[0]

nb_action = [[0]*int(2/step_size)]*env.action_space.shape[0]
action_codes = copy.deepcopy(nb_action)
for i in range(len(action_codes)):
    action_codes[i] = np.identity(len(action_codes[i]), dtype=np.int)

model = create_model(nb_action, state_space, stack_size, learning_rate, min_epsilon, max_epsilon, decay_rate)
memory = deque(maxlen=1000)

for i in range(episodes):
    print("Episode number: "+str(i+1))
    state = env.reset()
    score = 0
    obs_stack, state = stack_observations(state, None, stack_size, True)

    for j in range(max_steps):
        env.render()
        decay_step += 1

        # FIXME: fix predict_action function
        action = predict_action(model, env, obs_stack, nb_action, action_codes, max_epsilon, min_epsilon, decay_rate, decay_step, state_space, stack_size)
        final_action = []
        for i in range(len(action)):
            final_action.append(possible_actions[np.argmax(action[i])])


        #print(action)

        obs, reward, done, _ = env.step(final_action)
        score += reward

        if done == True:
            obs, frame_stack = stack_observations(np.zeros(state_space,), obs_stack, stack_size, False)
            break
        else:
            obs, frame_stack = stack_observations(obs, obs_stack, stack_size, False)

            if len(memory) > 100:
                batch - sampleMemory(memory, batch_size)
                actions = [item[1] for item in batch]
                states = np.array([item[0] for item in batch], ndmin=3)
                rewards = [item[2] for item in batch]
                next_states = np.array([item[0] for item in batch], ndmin=3)

                predicts = model.predict(next_states)
                targets = [gamma * np.max(item) for item in predicts]
                targets = [targets[i] + rewards[i] for i in range(len(targets))]
                target_fit = [item for item in model.predict(states)]

                for i in range(len(target_fit)):
                    target_fit[i][actions[i]] = target[i]

                    feats = np.array(states).reshape(-1, *state_space)
                    lables = np.array(target_fit).reshape(-1, action_space)

                    model.train_on_batch(x=feats, y=labels)

    scores.append(score)
    print(score)

print("DONE")
