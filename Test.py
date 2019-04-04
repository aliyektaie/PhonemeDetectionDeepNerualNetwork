import os
import datetime
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Activation
from keras.layers import Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
import keras.callbacks
import matplotlib.pyplot as plt

import Constants
from DataSet import DataSet

TRAIN_INDEX_FILE = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/index_sample.txt'
FEATURE_PATH = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/Features/mfcc/'
TRAIN_ALPHABET_FILE = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/alphabet_sample.txt'
OUTPUT_DIR = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Output'
MAX_LENGTH_IN_TIME = 185
FEATURE_COUNT = 15
DATASET_SIZE = 1024


def load_alphabet():
    file = open(TRAIN_ALPHABET_FILE, 'r')

    temp = file.readline()
    _alphabet = temp.split(',,')
    _alphabet_char_to_index = {ch: i for i, ch in enumerate(_alphabet)}
    _alphabet_index_to_char = {i: ch for i, ch in enumerate(_alphabet)}

    return _alphabet, _alphabet_char_to_index, _alphabet_index_to_char


alphabet, alphabet_char_to_index, alphabet_index_to_char = load_alphabet()

np.random.seed(55)


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet_char_to_index[char])
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet_index_to_char[c])
    return "".join(ret)


class TrainEntry:
    def __init__(self):
        self.word = None
        self.phonetics = None
        self.data_path = None


# Uses generator functions to supply train/test with
# data.
class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, dataset_index_file, mini_batch_size, val_split, absolute_max_string_len=57):
        self.mini_batch_size = mini_batch_size
        self.img_w = FEATURE_COUNT
        self.img_h = MAX_LENGTH_IN_TIME
        self.dataset_index_file = dataset_index_file
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

        self.num_example = 0
        self.entries_list = None
        self.Y_data = None
        self.Y_len = None
        self.cur_val_index = 0
        self.cur_train_index = 0

    def get_output_size(self):
        return len(alphabet) + 1

    # num_example can be independent of the epoch size due to the use of generators
    def build_sample_list(self, num_example):
        assert num_example % self.mini_batch_size == 0
        assert (self.val_split * num_example) % self.mini_batch_size == 0

        self.num_example = num_example
        self.entries_list = [None] * self.num_example

        self.Y_data = np.ones([self.num_example, self.absolute_max_string_len]) * -1
        self.Y_len = [0] * self.num_example

        ds = DataSet(self.dataset_index_file)
        num_sample_loaded = 0
        for i, entry in enumerate(ds.entries):
            if i == self.num_example or num_sample_loaded >= self.num_example:
                break

            for j in range(0, entry.audioCount):
                if num_sample_loaded >= self.num_example:
                    break

                sample = TrainEntry()
                sample.word = entry.word
                sample.phonetics = entry.get_phonetics_char_array()
                folder = FEATURE_PATH + sample.word[0: min(2, len(sample.word))] + Constants.SLASH
                sample.data_path = folder + sample.word + '_' + str(j) + '.npy'
                self.entries_list[num_sample_loaded] = sample
                num_sample_loaded += 1

        if num_sample_loaded != self.num_example:
            raise IOError('Could not pull enough sample from the dataset. ')

        for i, entry in enumerate(self.entries_list):
            phonetics_length = len(entry.phonetics)
            self.Y_len[i] = phonetics_length
            self.Y_data[i, 0:phonetics_length] = text_to_labels(entry.phonetics)

        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    def get_batch(self, index, size):
        X_data = np.ones([size, self.img_h, self.img_w])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        for i in range(size):
            data = np.load(self.entries_list[index + i].data_path).T
            X_data[i, 0:data.shape[0], :] = data
            labels[i, :] = self.Y_data[index + i]
            input_length[i] = data.shape[0]
            label_length[i] = self.Y_len[index + i]

        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
        }

        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return inputs, outputs

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.mini_batch_size)
            self.cur_train_index += self.mini_batch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.mini_batch_size)
            self.cur_val_index += self.mini_batch_size
            if self.cur_val_index >= self.num_example:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_sample_list(DATASET_SIZE)

    def on_epoch_begin(self, epoch, logs={}):
        # # rebind the paint function to implement curriculum learning
        # if 3 <= epoch < 6:
        #     self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
        #                                               rotate=False, ud=True, multi_fonts=False)
        # elif 6 <= epoch < 9:
        #     self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
        #                                               rotate=False, ud=True, multi_fonts=True)
        # elif epoch >= 9:
        #     self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
        #                                               rotate=True, ud=True, multi_fonts=True)
        # if epoch >= 21 and self.max_string_len < 12:
        #     self.build_sample_list(32000, 12, 0.5)
        pass


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# def decode_batch(test_func, word_batch):
#     out = test_func([word_batch])[0]
#     ret = []
#     for j in range(out.shape[0]):
#         out_best = list(np.argmax(out[j, 2:], 1))
#         out_best = [k for k, g in itertools.groupby(out_best)]
#         outstr = labels_to_text(out_best)
#         ret.append(outstr)
#     return ret
#
#
# class VizCallback(keras.callbacks.Callback):
#
#     def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
#         self.test_func = test_func
#         self.output_dir = os.path.join(
#             OUTPUT_DIR, run_name)
#         self.text_img_gen = text_img_gen
#         self.num_display_words = num_display_words
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)
#
#     def show_edit_distance(self, num):
#         num_left = num
#         mean_norm_ed = 0.0
#         mean_ed = 0.0
#         while num_left > 0:
#             word_batch = next(self.text_img_gen)[0]
#             num_proc = min(word_batch['the_input'].shape[0], num_left)
#             decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
#             for j in range(num_proc):
#                 edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
#                 mean_ed += float(edit_dist)
#                 mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
#             num_left -= num_proc
#         mean_norm_ed = mean_norm_ed / num
#         mean_ed = mean_ed / num
#         print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
#               % (num, mean_ed, mean_norm_ed))
#
#     def on_epoch_end(self, epoch, logs={}):
#         self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
#         self.show_edit_distance(256)
#         word_batch = next(self.text_img_gen)[0]
#         res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
#         if word_batch['the_input'][0].shape[0] < 256:
#             cols = 2
#         else:
#             cols = 1
#         for i in range(self.num_display_words):
#             plt.subplot(self.num_display_words // cols, cols, i + 1)
#             if K.image_data_format() == 'channels_first':
#                 the_input = word_batch['the_input'][i, 0, :, :]
#             else:
#                 the_input = word_batch['the_input'][i, :, :, 0]
#             plt.imshow(the_input.T, cmap='Greys_r')
#             plt.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
#         fig = pylab.gcf()
#         fig.set_size_inches(10, 13)
#         plt.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
#         plt.close()


def train(run_name, start_epoch, stop_epoch):
    # Input Parameters
    img_h = MAX_LENGTH_IN_TIME
    example_per_epoch = DATASET_SIZE
    val_split = 0.125
    val_words = int(example_per_epoch * (val_split))

    # Network parameters
    rnn_size = 64
    mini_batch_size = 32

    input_shape = (None, FEATURE_COUNT)

    data_gen = TextImageGenerator(TRAIN_INDEX_FILE, mini_batch_size, example_per_epoch - val_words)

    act = 'relu'
    network_output = Input(name='the_input', shape=input_shape, dtype='float32')
    input_data = network_output

    # inner = Dense(time_dense_size, activation=act, name='dense1')(input_shape)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(network_output)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        network_output)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    # transforms RNN output to character activations:
    network_output = Dense(data_gen.get_output_size(), kernel_initializer='he_normal',
                           name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(network_output)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[data_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    return model.fit_generator(generator=data_gen.next_train(),
                               steps_per_epoch=(example_per_epoch - val_words) // mini_batch_size,
                               epochs=stop_epoch,
                               validation_data=data_gen.next_val(),
                               validation_steps=val_words // mini_batch_size,
                               callbacks=[data_gen],
                               initial_epoch=start_epoch)


run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')

history = train(run_name, 0, 2)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
