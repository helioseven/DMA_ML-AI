import gym
import random
import numpy as np
import tflearn
import statistics
from collections import Counter

# setting up our gym environment
env = gym.make("CartPole-v0")
env.reset()
    
# global variables
LEARNING_RATE = 1e-3
GOAL_STEPS = 500
SCORE_REQUIREMENT = 50
INITIAL_GAMES = 10000

# main script code
def main():
    # get training data and train a new model
    training_data = initial_population()
    model = train_model(training_data)

    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        # reset the gym environment
        env.reset()
        for _ in range(GOAL_STEPS):
            # render the gym environment
            env.render()

            # pick a new action depending on previous trials (or randomly)
            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            # add that action to list of choices
            choices.append(action)

            # apply the action, record data
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            # append game state / action pair to memory list
            game_memory.append([new_observation, action])
            # add reward for the action to our current score
            score+=reward
            # when we are done, break out of the for loop
            if done: break

        # append the score from this environment run-through to our list of scores
        scores.append(score)

    # display some output
    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(SCORE_REQUIREMENT)

# this just creates a random population as a base to start learning
def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(INITIAL_GAMES):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(GOAL_STEPS):
            # choose random action (0 or 1)
            action = random.randrange(0, 2)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            # if the ai has failed
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= SCORE_REQUIREMENT:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', statistics.mean(accepted_scores))
    print('Median score for accepted scores:', statistics.median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size):

    network = tflearn.layers.core.input_data(shape=[None, input_size, 1], name='input')

    # network = tflearn.layers.core.fully_connected(network, 128, activation='relu')
    # network = tflearn.layers.core.dropout(network, 0.8)

    # network = tflearn.layers.core.fully_connected(network, 256, activation='relu')
    # network = tflearn.layers.core.dropout(network, 0.8)

    network = tflearn.layers.core.fully_connected(network, 512, activation='relu')
    network = tflearn.layers.core.dropout(network, 0.8)

    # network = tflearn.layers.core.fully_connected(network, 256, activation='relu')
    # network = tflearn.layers.core.dropout(network, 0.8)

    # network = tflearn.layers.core.fully_connected(network, 128, activation='relu')
    # network = tflearn.layers.core.dropout(network, 0.8)

    network = tflearn.layers.core.fully_connected(network, 2, activation='softmax')
    network = tflearn.layers.estimator.regression(network, optimizer='adam', learning_rate=LEARNING_RATE, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    # build a feature set out of the training data
    X = np.array([i[0] for i in training_data]).reshape(-1,
        len(training_data[0][0]),
        1)
    # build a label set out of the training data
    y = [i[1] for i in training_data]

    # if passed model does not exist, create one
    if not model:
        model = neural_network_model(input_size = len(X[0]))

    # perform model fitting/training
    model.fit({'input': X}, {'targets': y},
        n_epoch=10,
        snapshot_step=500,
        show_metric=True,
        run_id='openai_learning')

    # return the model
    return model

main()