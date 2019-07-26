import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import Input

model = Sequential()
model.add(LSTM(300, batch_input_shape=(1, None, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop', loss='crossentropy', metrics=['accuracy'])

from NLPhelperGEN.yelp_loader import *
mydata = yelp()
for epoch in range(10):
    #Now, I personally like knowing how far in training I am. So to see
    # that, I print what epoch I'm on for each epoch.
    print('Epoch {}/{}'.format(epoch+1, 10))
    #Now, we'll go through all the sentences in the dataset, one-by-one
    for i in range(len(mydata.data)):
        model.train_on_batch(x=np.array(mydata.data[i]).reshape(1, -1, 1),
                                          y=np.array(mydata.labels[i].reshape(1, -1, 2)))

accuracy=[]
for i in range(len(mydata.data)):
    acc, loss = model.evaluate(x=np.array(mydata.data[i]).reshape(1, -1, 1),
                                               y=np.array(mydata.labels[i].reshape(1, -1, 2),
                                               verbose=0))
    accuracy.append(acc)
print(sum(accuracy)/len(accuracy)*100)