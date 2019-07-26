import pandas as pd
import numpy as np
import gensim
from nltk.stem import SnowballStemmer
import keras
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.layers import Input, Concatenate

df_classics = pd.read_csv('NLP/classicLit/classicLit/the-classics.csv/datasets/classic-novels.csv', skipinitialspace=True)

def encoder_decoder(n_features_in, n_features_out, vocabulary_size_novels, vocabulary_size_summaries, rnn_units):
    encoder_inputs = Input(batch_shape=(1, n_features_in))
    encoder_embeddings = Embedding(input_dim=vocabulary_size_novels, output_dim=rnn_units)
    input_data = encoder_embeddings(encoder_inputs)
    encoder = LSTM(rnn_units, return_state=True)
    encoder_outputs, h, c = encoder(input_data)
    encoder_states = [h, c]

    decoder_inputs = Input(shape=(None, n_features_out))
    decoder_embeddings = Embedding(input_dim=vocabulary_size_summaries, output_dim=rnn_units)
    decoder_data = decoder_embeddings(decoder_inputs)
    decoder_lstm = LSTM(rnn_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_data, initial_state=encoder_states)
    decoder_dense = Dense(n_features_out, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    training_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    encoder_model = Model(encoder_inputs, encoder_states)

    inf_decoder_states_input_h = Input(batch_shape=(None, rnn_units))
    inf_decoder_states_input_c = Input(batch_shape=(None, rnn_units))
    inf_decoder_states_inputs = [inf_decoder_states_input_h, inf_decoder_states_input_c]
    inf_decoder_outputs, decoder_h, decoder_c = decoder_lstm(decoder_data, initial_state=inf_decoder_states_inputs)
    decoder_states = [decoder_h, decoder_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + inf_decoder_states_inputs, [decoder_outputs] + decoder_states)

    return training_model, encoder_model, decoder_model


from NLPhelperGEN.seq2seq import *
novels_data = seq2seq('NLP/classicLit/classicLit/the-classics.csv/datasets/classic-novels.csv')
novels_data.encoder_inputs('chap')
novels_data.decoder_outputs('text', length=15)

training_model, encoder_model, decoder_model = encoder_decoder(None, 15, len(novels_data.encoder_dic), len(novels_data.decoder_dic),  300)

novels_data.batch_training(training_model, 10)
