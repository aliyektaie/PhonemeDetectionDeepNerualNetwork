import sys
import librosa
import Constants
import numpy as np
import TrainConvLSTMKeras
import NormalizeFeatures
from keras import Model
from keras import backend as K
from python_speech_features import mfcc
import FeatureExtractor
import editdistance
from DataSet import static_get_phonetics_char_array
import os


class Entry:
    def __init__(self):
        self.path = ''
        self.label = ''
        self.word = ''


def load_features_from_audio_file(path):
    audio, _ = librosa.core.load(path, mono=True, sr=None)
    mfccs = mfcc(audio,
                 Constants.AUDIO_FILES_SAMPLING_RATE,
                 numcep=Constants.MFCC_COEFFICIENT_COUNT,
                 winlen=0.008,
                 winstep=0.008).T

    delta_added = FeatureExtractor.add_delta_features_to_mfcc(mfccs)
    return delta_added


def pad_features_to_model_input_size(features):
    result = np.zeros(TrainConvLSTMKeras.PADDED_FEATURE_SHAPE_INPUT)
    result[0:features.data.shape[0], :, :] = features

    return result


def decode_predict_ctc(out, top_paths=3):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
                                          greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        # TODO: the last character has a problem, it is always the same. Should be checked.
        text = TrainConvLSTMKeras.labels_to_text(lables[0:-1])
        results.append(text)

    return results


def load_scales():
    return np.load('./final_model/feature_scale_mean.npy'), np.load('./final_model/feature_scale_std.npy')


def load_file_paths(file_path):
    path = file_path[-4:]
    if path == '.wav':
        e = Entry()
        e.path = file_path
        return [e]
    elif path == '.txt':
        result = []
        with open(file_path) as file:
            line = file.readline()
            while line:
                entry = Entry()
                parts = line.strip().split('\t')
                entry.word = parts[1]
                entry.label = parts[2]
                entry.path = parts[3]

                if os.path.isfile(entry.path):
                    result.append(entry)

                line = file.readline()

        return result


def main(file_path):
    print('Loading input file(s) list')
    file_paths = load_file_paths(file_path)
    print('Loading normalization scale')
    mean, std = load_scales()

    print('Loading model')
    init_model()
    model_prediction = load_model()

    results = []
    results.append('Word,Phonetics,Best Prediction,Best Prediction Distance,'
                   '2nd Best Prediction,2nd Best Prediction Distance,'
                   '3rd Best Prediction,3rd Best Prediction Distance')

    print('Start predictions')
    for i, entry in enumerate(file_paths):
        print(f'{i+1} of {len(file_paths)}')
        features = load_features_from_audio_file(entry.path)
        NormalizeFeatures.normalize_single_data(features, mean, std)

        features = pad_features_to_model_input_size(features)
        features = np.expand_dims(features, axis=0)

        net_out_value = model_prediction.predict(features)
        r = decode_predict_ctc(net_out_value)
        phonetics = ''.join(static_get_phonetics_char_array(entry.label))
        ds = [
            editdistance.eval(static_get_phonetics_char_array(r[0]), static_get_phonetics_char_array(entry.label)),
            editdistance.eval(static_get_phonetics_char_array(r[1]), static_get_phonetics_char_array(entry.label)),
            editdistance.eval(static_get_phonetics_char_array(r[2]), static_get_phonetics_char_array(entry.label))
        ]
        line = f'{entry.word},{phonetics},{r[0]},{ds[0]},{r[1]},{ds[1]},{r[2]},{ds[2]}'
        results.append(line)

        if i % 100 == 0:
            save_results(results, print_to_console=False)

    save_results(results, print_to_console=True)


def save_results(results, print_to_console=False):
    final_result = '\n'.join(results)

    if print_to_console:
        print(final_result)

    with open("/Users/yektaie/Desktop/Phoneme Predictions.csv", "w") as text_file:
        text_file.write(final_result)


def init_model():
    TrainConvLSTMKeras.ONE_TO_ONE_ALPHABET_PATH = './final_model/oto_alphabet.txt'
    TrainConvLSTMKeras.init_predict()


def load_model():
    model, test_func, y_pred, input_data = TrainConvLSTMKeras.create_model()
    model_prediction = Model(inputs=input_data, outputs=y_pred)
    model_prediction.load_weights('./final_model/model_weights.h5')
    return model_prediction


if __name__ == '__main__':
    main('./final_model/oto_ds.ts.txt')
