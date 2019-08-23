import numpy as np
import matplotlib.pyplot as plt

def argmax_4d(matrix):
    _, b, c, d = matrix.shape
    c *= d
    b *= c

    i = np.argmax(matrix)
    _a = i // b
    i %= b
    _b = i // c
    i %= c
    _c = i // d
    i %= d

    return (_a, _b, _c, i)

def make_graph(inList):
    # generate figure
    fig = plt.figure()
    # plot the list
    plt.plot(inList, color="black")

    # add some pretty printing to our graph
    plt.title("Scores per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    # return the figure
    return fig

def sample_memory(buffered_list, batch_size):
    buffer_size = len(buffered_list)

    index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
    return [buffered_list[i] for i in index]