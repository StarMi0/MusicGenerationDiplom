import os
import random

import numpy as np
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

output_path = "result"
data_folder = "data/dataset"
len_data = 25
n_prev = 30


def create_model():
    model = Sequential()
    model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
    model.add(Dropout(0.6))
    model.add(LSTM(128, input_shape=(n_prev, 1), return_sequences=True))
    model.add(Dropout(0.6))
    model.add(LSTM(64, input_shape=(n_prev, 1), return_sequences=False))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation("linear"))
    return model


def get_notes(data_: list) -> list:
    notes = []
    notes_ = list(map(get_notes, random.choices(data_, k=len_data)))
    for i in notes_:
        notes += i
    print(f"Lean of dataset is: {len(notes)}")
    return notes


def data_prepare():
    data_ = []

    for dirName, subdirList, fileList in os.walk(data_folder):

        for fileName in fileList:
            if fileName.endswith('.mid'):
                data_.append(os.path.join(dirName, fileName))

    print(f"Dataset contains {len(data_)} mid files")

    notes = get_notes(data_)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array(notes).reshape(-1, 1))
    notes = list(scaler.transform(np.array(notes).reshape(-1, 1)))

    notes = [list(note) for note in notes]

    # subsample data for training and prediction
    X = []

    # number of notes in a batch
    n_prev = 30
    for i in range(len(notes) - n_prev):
        X.append(notes[i: i + n_prev])
    # save a seed to do prediction later
    X_test = X[-300:]

    return scaler, notes, X_test
