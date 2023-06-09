import os.path

import numpy as np
from mido import Message, MidiFile, MidiTrack
from functions import create_model, output_path, data_prepare, weights_path


def save_pred(prediction):
    mid = MidiFile()
    track = MidiTrack()
    t = 0
    for note in prediction:
        msg_on = Message.from_dict({'type': 'note_on', 'channel': 0, 'note': note, 'velocity': 80, 'time': 30})
        # you need to add some pauses "note_off"
        msg_off = Message.from_dict({'type': 'note_off', 'channel': 0, 'note': note, 'velocity': 80, 'time': 60})
        track.append(msg_on)
        track.append(msg_off)
        track.append(msg_off)
    mid.tracks.append(track)
    mid.save(os.path.join(output_path, "LSTM_music.mid"))


def gan_melody(model):
    """
    Функция генерирует мелодию с помощью нейронной сети.

    :return: audio
    """
    scaler, notes, X_test = data_prepare()
    prediction = model.predict(np.array(X_test))
    prediction = np.squeeze(prediction)
    prediction = np.squeeze(scaler.inverse_transform(prediction.reshape(-1, 1)))
    prediction = [int(i) for i in prediction]
    return prediction


def load_trained_model(weights_path):
    print("loading weights")
    model = create_model()
    model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    model = load_trained_model(weights_path)
    save_pred(gan_melody(model))
