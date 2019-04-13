import editdistance
import itertools

from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Reshape, Dense, GRU, add, Activation, concatenate, Lambda, LSTM
from keras.optimizers import SGD
import matplotlib.pyplot as plt

from DataSet import *
import Constants
from random import shuffle
import numpy as np
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import ctc_label_dense_to_sparse
from keras.backend.common import epsilon
from keras import backend as K
import sys

CONV_LAYERS_COUNT = 3
USE_RNN_AT_ALL = True
POOL_SIZE = 2
KEEP_SMALL_EXAMPLES = False
PADDED_FEATURE_SHAPE_INPUT = (735, 20, 3)
# NUMBER_OF_SAMPLE_TO_TRAIN_ON = 420790
# NUMBER_OF_SAMPLE_TO_TRAIN_ON = 00000
NUMBER_OF_RNN_LAYERS = 2
RNN_COUNT = 20
BATCH_SIZE = 32
VALIDATION_PORTION = 0.2
MAX_PHONETICS_LEN = 30
MIN_PHONETICS_LEN = 1
FEATURE_FOLDER_NAME = 'mfcc_balanced'
ONE_TO_ONE_DATA_SET_PATH_TRAIN = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_ds.tr.txt'
ONE_TO_ONE_DATA_SET_PATH_VAL = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_ds.val.txt'
ONE_TO_ONE_DATA_SET_PATH_TEST = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_ds.ts.txt'
ONE_TO_ONE_ALPHABET_PATH = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_alphabet.txt'
OUTPUT_DIR = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Output'

ALPHABET_CHAR_TO_INDEX = {}
ALPHABET_INDEX_TO_CHAR = {}
ALPHABET = []

NETWORK_INPUT_LAYER = 'INPUT_LAYER'


class TrainingEntry:
    def __init__(self, line=None):
        self.word = None
        self.phonetics = None
        self.id = None
        self.data_path = None
        self.data = None
        self.label = None

        if line is not None:
            parts = line.strip().split('\t')
            self.id = parts[0]
            self.word = parts[1]
            self.phonetics = parts[2]
            self.data_path = parts[3]

    def to_str_line(self):
        return f'{self.id}\t{self.word}\t{self.phonetics}\t{self.data_path}'


class TrainDataGenerator(keras.utils.Sequence):
    # Initialization
    def __init__(self, samples, max_output_length, input_shape, batch_size=1, incremental=False):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_output_length = max_output_length
        self.data_set = samples

        self.samples = None
        if not incremental:
            self.samples = samples

        self.indexes = None
        self.incremental = incremental
        self.max_size = MIN_PHONETICS_LEN + 1
        self.on_epoch_end()

    # Denotes the number of batches per epoch'
    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        samples_in_batch = [self.samples[i] for i in indexes]
        X_data = self.generate_inputs_X(samples_in_batch)
        labels = self.generate_target_labels(samples_in_batch)
        input_length = self.generate_input_dimensions_after_convolution(samples_in_batch)
        label_length = self.generate_target_labels_length(samples_in_batch)

        inputs = {
            NETWORK_INPUT_LAYER: X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'samples_in_batch': samples_in_batch,
        }

        outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function
        return inputs, outputs

    # Updates indexes after each epoch
    def on_epoch_end(self, epoch=0, logs={}):
        if self.incremental:
            self.max_size += 1

            self.samples = []
            while self.max_size < MAX_PHONETICS_LEN and len(self.samples) == 0:
                self.samples = []

                for entry in self.data_set:
                    if len(entry.label) <= self.max_size:
                        self.samples.append(entry)

                if len(self.samples) == 0:
                    self.max_size += 1

            do_print(f'On epoch end: Next epoch will train with {len(self.samples)} samples.' +
                     f' Max label length: {self.max_size}')

        self.indexes = np.arange(len(self.samples))
        np.random.shuffle(self.indexes)

    def generate_inputs_X(self, samples_in_batch):
        X_data = np.zeros((self.batch_size, *self.input_shape))

        for i, sample in enumerate(samples_in_batch):
            X_data[i, 0:sample.data.shape[0], :, :] = sample.data

        return X_data

    def generate_target_labels(self, samples_in_batch):
        target_data = np.ones((self.batch_size, self.max_output_length)) * -1

        for i, sample in enumerate(samples_in_batch):
            target_data[i, 0:len(sample.label)] = sample.label

        return target_data

    def generate_input_dimensions_after_convolution(self, samples_in_batch):
        dimensions = np.zeros((self.batch_size, 1))

        for i, sample in enumerate(samples_in_batch):
            l = sample.data.shape[0] // (POOL_SIZE ** CONV_LAYERS_COUNT)
            # if l <= len(sample.label):
            #     l = len(sample.label) + 1

            dimensions[i, 0] = l

        return dimensions

    def generate_target_labels_length(self, samples_in_batch):
        labels_length = np.zeros((self.batch_size, 1))

        for i, sample in enumerate(samples_in_batch):
            labels_length[i, 0] = len(sample.label)

        return labels_length

    def set_model(self, model):
        pass

    def set_params(self, params):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class VisualizationCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, data_gen):
        self.test_func = test_func
        self.output_dir = os.path.join(OUTPUT_DIR, run_name)
        self.data_gen = data_gen
        self.num_display_words = BATCH_SIZE
        self.epochs_result = {}
        self.accuracy = []
        self.all_phonetics = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, epoch):
        result = {}
        num = len(self.data_gen)

        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        mean_ed_sq = 0.0
        count = 0
        while num_left > 0:
            word_batch, _ = self.data_gen[num_left - 1]
            inputs = word_batch[NETWORK_INPUT_LAYER]
            shape = inputs.shape
            decoded_res = decode_batch(self.test_func, word_batch)
            for j in range(shape[0]):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['samples_in_batch'][j].phonetics)
                mean_ed += float(edit_dist)
                mean_ed_sq += (float(edit_dist) * float(edit_dist))
                mean_norm_ed += float(edit_dist) / max(len(word_batch['samples_in_batch'][j].phonetics), 1)
                count += 1

                phonetics = word_batch['samples_in_batch'][j].phonetics + '      [' + str(
                    word_batch['samples_in_batch'][j].id) + ']'
                if phonetics not in self.all_phonetics:
                    self.all_phonetics.append(phonetics)

                result[phonetics] = {
                    'prediction': decoded_res[j],
                    'edit_distance': float(edit_dist)
                }

            num_left -= 1
        mean_norm_ed = mean_norm_ed / max(count, 1)
        mean_ed = mean_ed / max(count, 1)
        st_dev = np.sqrt(mean_ed_sq / max(count + (mean_ed * mean_ed), 1))
        do_print('\nOut of %d samples:  Mean edit distance: %.3f +- %.3f Mean normalized edit distance: %0.3f'
                 % (count, mean_ed, st_dev, mean_norm_ed))
        self.accuracy.append('%d,%.3f,%.3f' % (epoch, mean_ed, st_dev))

        return result

    def on_epoch_end(self, epoch=0, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        self.epochs_result[epoch] = self.show_edit_distance(epoch)
        self.save_results_as_csv(epoch)

    def save_results_as_csv(self, epoch):
        epoch += 1
        path = os.path.join(self.output_dir, 'predictions.csv')

        file = open(path, "w")
        file.write('Original Phonetics')

        for i in range(epoch):
            file.write(f',Prediction Epoch {i + 1},Distance Epoch {i + 1}')

        file.write('\n')

        for phonetics in self.all_phonetics:
            file.write(phonetics)
            for e in range(epoch):
                pred = ''
                distance = ''

                if phonetics in self.epochs_result[e]:
                    pred = self.epochs_result[e][phonetics]['prediction']
                    pred = pred.replace('"', '').replace(',', '')
                    distance = self.epochs_result[e][phonetics]['edit_distance']
                file.write(f',"{pred}",{distance}')

            file.write('\n')

        file.close()

        path = os.path.join(self.output_dir, 'accuracy.csv')
        file = open(path, "w")

        for line in self.accuracy:
            file.write(line + '\n')

        file.close()


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(ALPHABET_CHAR_TO_INDEX[char])

    ret.append(len(ALPHABET))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(ALPHABET):  # CTC Blank
            ret.append("")
        else:
            ret.append(ALPHABET_INDEX_TO_CHAR[c])
    return "".join(ret)


def get_subset(all_entries, start, end):
    start = int(len(all_entries) * start)
    end = int(len(all_entries) * end)

    return all_entries[start:end]


def create_one_to_one_dataset():
    if os.path.isfile(ONE_TO_ONE_DATA_SET_PATH_TRAIN):
        return

    do_print('Creating one-to-one dataset')
    ds = DataSet(Constants.TRAINING_FOLDER_PATH)
    all_entries = []
    dropped = 0
    max_label_length = 0
    min_label_length = 100
    zero_length_labels = 0
    short_length_labels = 0
    invalid_symbols = [';', '+', '4', '5', '7', ' ']

    for index, entry in enumerate(ds.entries):
        if index % 1000 == 0:
            do_print('   -> %d of %d' % (index, len(ds.entries)))

        if entry.phonetics == '':
            zero_length_labels += 1
            continue

        ph = static_get_phonetics_char_array(entry.phonetics)
        for symbol in invalid_symbols:
            if symbol in ph:
                continue

        for i, path in enumerate(entry.get_audio_paths()):
            path = path.replace('Audio Files', 'Features/' + FEATURE_FOLDER_NAME).replace('.wav', '.npy')

            if os.path.isfile(path):
                data = np.load(path)
                label_length = len(static_get_phonetics_char_array(entry.phonetics))

                if (data.shape[0] > label_length) \
                        and (label_length > 0) \
                        and (label_length > MIN_PHONETICS_LEN) \
                        and (data.shape[0] > NUMBER_OF_RNN_LAYERS):

                    max_label_length = max(max_label_length, label_length)
                    min_label_length = min(min_label_length, label_length)
                    e = TrainingEntry()
                    e.word = entry.word
                    e.phonetics = entry.phonetics
                    e.id = entry.word + '_' + str(i)
                    e.data_path = path

                    all_entries.append(e)
                elif data.shape[0] <= NUMBER_OF_RNN_LAYERS:
                    do_print('   -> [INFO] Dropped sample because data is too short')
                elif label_length == 0:
                    zero_length_labels += 1
                    do_print('   -> [INFO] Dropped sample because label length is 0')
                elif label_length <= MIN_PHONETICS_LEN:
                    short_length_labels += 1
                    # do_print('   -> [INFO] Dropped sample because label length is less than ' + str(MIN_PHONETICS_LEN))
                else:
                    do_print('   -> [INFO] Dropped sample because label is longer than input')
                    dropped += 1

    shuffle(all_entries)

    alphabet = set()
    create_alphabet_set(all_entries, alphabet)

    with open(ONE_TO_ONE_DATA_SET_PATH_TRAIN, 'w') as file:
        for entry in get_subset(all_entries, 0.0, 0.5):
            file.write(entry.to_str_line())
            file.write('\n')

    with open(ONE_TO_ONE_DATA_SET_PATH_VAL, 'w') as file:
        for entry in get_subset(all_entries, 0.5, 0.7):
            file.write(entry.to_str_line())
            file.write('\n')

    with open(ONE_TO_ONE_DATA_SET_PATH_TEST, 'w') as file:
        for entry in get_subset(all_entries, 0.7, 1.0):
            file.write(entry.to_str_line())
            file.write('\n')

    save_alphabet_file(alphabet)

    do_print('   -> Total sample dropped: ' + str(dropped))
    do_print('   -> Total sample missing label: ' + str(zero_length_labels))
    do_print('   -> Total sample with too short label: ' + str(short_length_labels))
    do_print('   -> Max label length: ' + str(max_label_length))
    do_print('   -> Min label length: ' + str(min_label_length))
    do_print('   -> Done')
    do_print('')


def create_alphabet_set(all_entries, alphabet):
    for entry in all_entries:
        symbols = static_get_phonetics_char_array(entry.phonetics)
        for symbol in symbols:
            if symbol not in alphabet:
                alphabet.add(symbol)


def save_alphabet_file(alphabet):
    with open(ONE_TO_ONE_ALPHABET_PATH, 'w') as file:
        file.write('\t'.join(list(alphabet)))


def load_entries_for_training():
    do_print('Loading training entries')
    entries_train = []
    entries_val = []

    with open(ONE_TO_ONE_DATA_SET_PATH_TRAIN) as file:
        line = file.readline()
        while line:
            entry = TrainingEntry(line=line)
            ll = len(static_get_phonetics_char_array(entry.phonetics))
            if (ll >= MIN_PHONETICS_LEN) and (ll <= MAX_PHONETICS_LEN):
                entries_train.append(entry)
            line = file.readline()

    with open(ONE_TO_ONE_DATA_SET_PATH_VAL) as file:
        line = file.readline()
        while line:
            entry = TrainingEntry(line=line)
            ll = len(static_get_phonetics_char_array(entry.phonetics))
            if (ll >= MIN_PHONETICS_LEN) and (ll <= MAX_PHONETICS_LEN):
                entries_val.append(entry)
            line = file.readline()

    null_count = len(entries_train) // 100
    for i in range(null_count):
        e = TrainingEntry()
        e.label = [len(ALPHABET), len(ALPHABET), len(ALPHABET)]
        e.phonetics = ''
        e.data_path = None
        e.data = np.zeros(PADDED_FEATURE_SHAPE_INPUT)
        e.word = ''

        entries_train.append(e)

    shuffle(entries_train)

    alphabet = None
    with open(ONE_TO_ONE_ALPHABET_PATH) as file:
        alphabet = file.readline().split('\t')

    do_print('   -> Done')
    do_print('')

    return entries_train, entries_val, alphabet


def load_training_data_into_memory(dataset):
    do_print('Loading dataset in memory')
    result = []
    count = 0

    for i, entry in enumerate(dataset):
        if i % 1000 == 0 and i > 0:
            do_print(f'   -> {i} of {len(dataset)} [dropped: {count}]')

        data = None
        if entry.data_path is not None:
            data = np.load(entry.data_path)
            entry.label = text_to_labels(static_get_phonetics_char_array(entry.phonetics))
        else:
            data = np.zeros(PADDED_FEATURE_SHAPE_INPUT)
            entry.label = entry.label

        if KEEP_SMALL_EXAMPLES or len(entry.label) * (POOL_SIZE ** CONV_LAYERS_COUNT) < data.shape[0]:
            entry.data = data
            result.append(entry)
        else:
            # do_print(f'   -> Dropped sample {entry.data.shape} -> {len(entry.label)}')
            count += 1

    do_print(f'   -> Total sample dropped: {count}')
    do_print('   -> Done')
    do_print('')

    return result


def do_print(s):
    print(s, flush=True)


def load_alphabet_indices():
    global ALPHABET_INDEX_TO_CHAR
    global ALPHABET_CHAR_TO_INDEX

    ALPHABET_CHAR_TO_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}
    ALPHABET_INDEX_TO_CHAR = {i: ch for i, ch in enumerate(ALPHABET)}


def get_network_output_size():
    return len(ALPHABET) + 1


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 0:MAX_PHONETICS_LEN, :]

    # return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


# copied code from Keras to be able to add ignore_longer_outputs_than_inputs=True
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.

    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    """
    label_length = tf.to_int32(tf.squeeze(label_length, axis=-1))
    input_length = tf.to_int32(tf.squeeze(input_length, axis=-1))
    sparse_labels = tf.to_int32(ctc_label_dense_to_sparse(y_true, label_length))

    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + epsilon())

    loss = tf.nn.ctc_loss(inputs=y_pred,
                          labels=sparse_labels,
                          sequence_length=input_length,
                          preprocess_collapse_repeated=True,
                          ignore_longer_outputs_than_inputs=True)

    return tf.expand_dims(loss, 1)


def decode_batch(test_func, word_batch):
    out = test_func([word_batch[NETWORK_INPUT_LAYER]])
    out = out[0]

    ret_1 = []

    for j in range(out.shape[0]):
        single_out = out[j, 0:MAX_PHONETICS_LEN]
        argmax_1 = np.argmax(single_out, 1)

        out_best_1 = list(argmax_1)
        out_best_1 = [k for k, g in itertools.groupby(out_best_1)]
        outstr_1 = labels_to_text(out_best_1)
        ret_1.append(outstr_1)

    return ret_1


def create_model():
    # Input Parameters
    feature_count_per_entry = PADDED_FEATURE_SHAPE_INPUT[1]

    # Network parameters
    conv_filters = 16
    kernel_2d_size = (3, 3)
    pool_size = POOL_SIZE
    time_dense_size = 64
    rnn_size = RNN_COUNT

    act = 'relu'
    input_data = Input(name=NETWORK_INPUT_LAYER,
                       # shape=PADDED_FEATURE_SHAPE_INPUT,
                       dtype='float32', shape=PADDED_FEATURE_SHAPE_INPUT)

    inner = Conv2D(conv_filters, kernel_2d_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(input_data)

    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)

    for i in range(1, CONV_LAYERS_COUNT):
        inner = Conv2D(conv_filters, kernel_2d_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv' + str(i + 2))(inner)

        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max' + str(i + 2))(inner)

    time_steps = PADDED_FEATURE_SHAPE_INPUT[0]
    conv_to_rnn_dims = (time_steps // (pool_size ** CONV_LAYERS_COUNT),
                        (feature_count_per_entry // (pool_size ** CONV_LAYERS_COUNT)) * conv_filters)

    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    if USE_RNN_AT_ALL:
        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)

        inner = concatenate([gru_2, gru_2b])

    # transforms RNN output to character activations:
    inner = Dense(get_network_output_size(),
                  kernel_initializer='he_normal',
                  name='dense2')(inner)

    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[MAX_PHONETICS_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-5, momentum=0.5, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['acc'])

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    return model, test_func


def main():
    global ALPHABET

    create_one_to_one_dataset()
    dataset_train, dataset_val, ALPHABET = load_entries_for_training()
    do_print('len alphabet: ' + str(len(ALPHABET)))
    load_alphabet_indices()

    dataset_train = load_training_data_into_memory(dataset_train)
    dataset_val = load_training_data_into_memory(dataset_val)

    try_training_model(dataset_train, dataset_val)
    do_print('Done training!')


def try_training_model(train_entries, validation_entries):
    keras.regularizers.l1_l2(l1=0.01, l2=0.01)

    model, test_func = create_model()
    run_name = 'first run'
    train_gen = TrainDataGenerator(train_entries, MAX_PHONETICS_LEN, PADDED_FEATURE_SHAPE_INPUT, BATCH_SIZE)
    validation_gen = TrainDataGenerator(validation_entries, MAX_PHONETICS_LEN, PADDED_FEATURE_SHAPE_INPUT, BATCH_SIZE)
    viz_cb = VisualizationCallback(run_name, test_func, validation_gen)
    result = True
    # try:
    model.fit_generator(generator=train_gen,
                        validation_data=validation_gen,
                        callbacks=[viz_cb, train_gen],
                        steps_per_epoch=None,
                        epochs=100)
    # except:
    #     result = False

    return result


if __name__ == '__main__':
    main()
