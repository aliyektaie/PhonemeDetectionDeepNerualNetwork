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
import edit_distance
from DataSet import static_get_phonetics_char_array
import matplotlib.pyplot as plt
import os

OUTPUT_FOLDER = '/Users/yektaie/Desktop/'


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


def create_distance_error():
    result = {}

    for i in range(21):
        result[i] = 0

    return result


def plot_error_distance_in_predictions(distances, title):
    values = []
    counts = []

    for i in range(0, 20 + 1):
        if distances[i] > 0:
            values.append(distances[i])
            counts.append(i)

    plt.figure(figsize=(7, 6), dpi=200)
    plt.bar(np.arange(len(counts)), values, align='center', color='green')
    plt.xticks(np.arange(len(counts)), counts)
    plt.ylabel('Frequency')
    plt.title(f'Distance of Predicted to True Phonetics ({title})')
    plt.savefig(OUTPUT_FOLDER + title + '.png')

    plt.show()


def add_to_confusion_matrix(confusion_matrix, opcodes, predicted, label):
    insert_delete_idx = len(TrainConvLSTMKeras.ALPHABET)

    for code in opcodes:
        try:
            op = code[0]

            if op == 'equal':
                l_index = TrainConvLSTMKeras.ALPHABET_CHAR_TO_INDEX[label[code[1]]]
                confusion_matrix[l_index, l_index] += 1
            elif op == 'replace':
                l_index = TrainConvLSTMKeras.ALPHABET_CHAR_TO_INDEX[label[code[1]]]
                p_index = TrainConvLSTMKeras.ALPHABET_CHAR_TO_INDEX[predicted[code[1]]]

                confusion_matrix[l_index, p_index] += 1
                confusion_matrix[p_index, l_index] += 1
            elif op == 'delete':
                ch = predicted[code[1]]
                p_index = TrainConvLSTMKeras.ALPHABET_CHAR_TO_INDEX[ch]

                confusion_matrix[insert_delete_idx, p_index] += 1
                confusion_matrix[p_index, insert_delete_idx] += 1
            elif op == 'insert':
                ch = label[code[3]]
                l_index = TrainConvLSTMKeras.ALPHABET_CHAR_TO_INDEX[ch]

                confusion_matrix[insert_delete_idx, l_index] += 1
                confusion_matrix[l_index, insert_delete_idx] += 1
        except:
            pass

# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    target_names = [c for c in TrainConvLSTMKeras.ALPHABET]
    target_names.append('r/d')

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(60, 60), dpi=200)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for ii in range(cm.shape[0]):
        for jj in range(cm.shape[1]):
            if np.isnan(cm[ii, jj]):
                cm[ii, jj] = 0

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(OUTPUT_FOLDER + 'confusion matrix.png')
    plt.show()


def main(file_path):
    print('Loading input file(s) list')
    file_paths = load_file_paths(file_path)
    # file_paths = file_paths[0:50]
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
    distance_error_count = [create_distance_error(),
                            create_distance_error(),
                            create_distance_error()]

    confusion_matrix = np.zeros((len(TrainConvLSTMKeras.ALPHABET) + 1, len(TrainConvLSTMKeras.ALPHABET) + 1))

    for ii, entry in enumerate(file_paths):
        print(f'{ii + 1} of {len(file_paths)}')
        features = load_features_from_audio_file(entry.path)
        NormalizeFeatures.normalize_single_data(features, mean, std)

        features = pad_features_to_model_input_size(features)
        features = np.expand_dims(features, axis=0)

        net_out_value = model_prediction.predict(features)
        r = decode_predict_ctc(net_out_value)
        phonetics = ''.join(static_get_phonetics_char_array(entry.label))

        sm = []
        for i in range(0, 3):
            sm.append(edit_distance.SequenceMatcher(a=static_get_phonetics_char_array(r[i]),
                                                    b=static_get_phonetics_char_array(entry.label)))

        add_to_confusion_matrix(confusion_matrix,
                                sm[0].get_opcodes(),
                                static_get_phonetics_char_array(r[0]),
                                static_get_phonetics_char_array(entry.label))

        ds = [sm[0].distance(), sm[1].distance(), sm[2].distance()]

        for i in range(0, 3):
            if ds[i] in distance_error_count[i]:
                distance_error_count[i][ds[i]] += 1

        line = f'{entry.word},{phonetics},{r[0]},{ds[0]},{r[1]},{ds[1]},{r[2]},{ds[2]}'
        results.append(line)

        if ii % 100 == 0:
            save_results(results, print_to_console=False)

    save_results(results, print_to_console=True)

    chart_titles = ['Best Prediction', '2nd Best Prediction', '3rd Best Prediction']
    for i in range(0, 3):
        plot_error_distance_in_predictions(distance_error_count[i], chart_titles[i])

    plot_confusion_matrix(confusion_matrix, normalize=True)
    np.save(OUTPUT_FOLDER + 'confusion matrix', confusion_matrix)


def save_results(results, print_to_console=False):
    final_result = '\n'.join(results)

    if print_to_console:
        print(final_result)

    with open(OUTPUT_FOLDER + "Phoneme Predictions.csv", "w") as text_file:
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
