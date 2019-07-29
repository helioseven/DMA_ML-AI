import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import copy
from collections import deque

path = "/Users/student/DMA_ML-AI/"

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
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(100, (1,8), input_shape=(state_space, stack_size, 1)),
                                 tf.keras.layers.Conv2D(75,(1,8)),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(50, activation="relu"),
                                 tf.keras.layers.Dense(25, activation="relu"),
                                 tf.keras.layers.Dense(10, activation="relu"),
                                 tf.keras.layers.Dense(len(np.array(nb_action).flatten()), activation="softmax")
                                 ])
    opt = tf.keras.optimizers.Adam(lr=learning_rate,
                                beta_1=min_epsilon,
                                beta_2=max_epsilon,
                                decay=decay_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy")
    return model

def load_model():
    model = tf.keras.models.load_model(path+"trained_model/bipedal_walker_model.h5")
    return model

def predict_action(model, env, obs_stack, nb_action, action_codes, max_epsilon, min_epsilon, decay_rate, decay_step, state_space, stack_size, step_size):
    nb_random = np.random.random()
    epsilon = max_epsilon + (min_epsilon - max_epsilon) * np.exp(-decay_rate * decay_step)

    if epsilon > 1:
    #if epsilon > nb_random:
        # make input random

        prediction = np.array(nb_action)

        for i in range(len(nb_action)):
            for j in range(len(nb_action[i])):
                prediction[i][j] = random.randint(1, len(nb_action[i]))-1
    else:
        # use model to predict
        feats = np.array(obs_stack).reshape(1, 24, stack_size, 1)
        '''
        rewards = model.predict(feats)
        choice1 = np.argmax(rewards[0])
        ...
        choice4 = np.argmax(rewards[3])
        choice = [choice1, ..., choice4]
        '''

        # for each 16 numbers, find the argmax
        predictions = model.predict(feats)
        #print(predictions)

        predictions = predictions.reshape(int(2/step_size), 4)

    choice = []
    for i in range(len(nb_action)):
        choice.append(np.argmax(predictions[i]))
    output = []
    for i in range(len(choice)):
        output.append(action_codes[i][int(choice[i])])
    return output

def sample_memory(buffered_list, batch_size):
    buffer_size = len(buffered_list)

    index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
    return [buffered_list[i] for i in index]

def graph_results(scores):
    fig = plt.figure()
    plt.plot(scores, color="black")

    plt.title("Scores per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    return fig

def argmax_2d(matrix):
    final_action = []
    for i in range(len(matrix)):
        final_action.append(possible_actions[np.argmax(matrix[i])])
    return final_action


####################
# constants
episodes = 10000
max_steps = 1000
stack_size = 15

learning_rate = 0.00025
decay_rate = 0.00001
min_epsilon = 0.01
max_epsilon = 1

batch_size = 64
gamma = 0.618

step_size = 0.125

show_render = False

possible_actions = []
for i in range(int(2/step_size)+1):
    possible_actions.append(step_size*i-1)
####################
scores = []
decay_step = 0

env = gym.make('BipedalWalker-v2')
print(env.observation_space.shape)
obs = env.reset()

state_space = obs.shape[0]

nb_action = [[0]*int(2/step_size)]*env.action_space.shape[0]

action_codes = copy.deepcopy(nb_action)
for i in range(len(action_codes)):
    action_codes[i] = np.identity(len(action_codes[i]), dtype=np.int)

if not os.path.exists(path+"trained_model/bipedal_walker_model.h5"):
    print("No model found: creating new model")
    model = create_model(nb_action, state_space, stack_size, learning_rate, min_epsilon, max_epsilon, decay_rate)
    new_model = True
else:
    print("Model found: loading model")
    model = load_model()
    new_model = False

    episodes = 1
memory = deque(maxlen=1000)

for i in range(episodes):
    print("Episode number: "+str(i+1))
    state = env.reset()
    score = 0
    obs_stack, state = stack_observations(state, None, stack_size, True)

    for j in range(max_steps):
        #env.render()
        decay_step += 1

        action = predict_action(model, env, obs_stack, nb_action, action_codes, max_epsilon, min_epsilon, decay_rate, decay_step, state_space, stack_size, step_size)
        '''
        final_action = []
        for i in range(len(action)):
            final_action.append(possible_actions[np.argmax(action[i])])
        '''
        final_action = argmax_2d(action)

        #print(action)

        obs, reward, done, _ = env.step(final_action)
        score += reward

        if done == True:
            obs_stack, state = stack_observations(np.zeros(state_space,), obs_stack, stack_size, False)
            break
        else:
            obs_stack, state = stack_observations(obs, obs_stack, stack_size, False)

        memory.append((state, action, reward))

        state = obs

        if len(memory) > 500:
            #print("Training")
            batch = sample_memory(memory, batch_size)
            states = np.array([item[0] for item in batch])
            #print("Mark 2: {}".format(states.shape))
            #print(states.shape)

            #print(states)
            #print(states_np.shape)

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
            target_fit = [item for item in np.array(model.predict(states)).reshape(-1, 4, int(2/step_size))]

            for i in range(batch_size):
                code = argmax_2d(actions[i])
                for k in range(len(code)):
                    target_fit[i][k] = targets[i]

            feats = np.array(states).reshape(-1, 24, stack_size, 1)
            labels = np.array(target_fit).reshape(-1, 4*int(2/step_size))
            #print("Features: "+str(feats.shape))
            #print("Labels: "+str(labels.shape))
            model.train_on_batch(x=feats, y=labels)

    scores.append(score)
    #print(score)

model.save(path+"trained_model/bipedal_walker_model.h5")
fig = graph_results(scores)
fig.savefig(fname="output.png")
plt.show(fig)
