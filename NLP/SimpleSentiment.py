import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from NLPhelperGEN.yelp_loader import *

mydata = yelp()
mydata.reviews
mydata.labels
review_data = list(zip(mydata.reviews, mydata.labels))

positive_vocabulary = []
negative_vocabulary = []
#Iterate through review_data:
for tuple in review_data:
    #Check if the label is positive, aka: 1
    if tuple[1] == 1:
        #Since we don't care about the order of the words, just
        # that the words are in our positive vocabulary list, we
        # can add all the items in tuple[0] to positive_vocabulary
        # in one fell swoop using += as opposed to the
        # .append() function.
        positive_vocabulary+=tuple[0]
    #And if the review is not positive, we add its contents to
    # the negative_vocabulary list.
    else:
        negative_vocabulary+=tuple[0]

feature_vocab = set(positive_vocabulary) ^ set(negative_vocabulary) #The '^' symbol means symmetric difference in Python

fs_dic = {}
ct=0
for word in feature_vocab:
    fs_dic[word]=ct
    ct+=1

#review is the review being converted, and dic is the dictionary we're using to convert words into indeces.
def data_builder(review, dic):
    #We create an array of 0s as long as the dictionary's length.
    data = [0 for _ in range(len(dic))]
    #Now, we go through every word in the review.
    for word in review:
        #And if the word is in our dictionary, in other words is used
        # as a feature of either a good or bad review.
        if word in dic.keys():
            #We add 1 at the index in our array that represents that word.
            data[dic[word]] += 1
    return data

review_representations = []
for review in mydata.reviews:
    review_representations.append(data_builder(review, fs_dic))

review_representations = np.array(review_representations).reshape(-1, len(fs_dic))

model = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=review_representations, y=mydata.y, epochs=10, verbose=2)
