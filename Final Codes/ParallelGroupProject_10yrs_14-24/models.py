import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from sklearn.metrics import mean_squared_error

def model_LSTM(input_shape,units=50):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # model.summary()
    return model


def model_BiLSTM(input_shape,units=50):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=units, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=units, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # model.summary()
    return model


def model_GRU(input_shape,units=50):
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # model.summary()
    return model

def model_Hybrid(input_shape,units=50):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # model.summary()
    return model