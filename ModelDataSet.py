import numpy as np
import Constants
import keras
from DataSet import static_get_phonetics_char_array


# Base code is coming from:
#    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#
# Also, changes were made inspired by:
#    https://github.com/Tony607/keras-image-ocr/blob/master/image-ocr.ipynb
#
# Changes made to adapt to the current project.

class ModelDataSet(keras.utils.Sequence):
    def __init__(self, data_path, samples_id_list, labels, labels_alphabet, max_phonetics_length, dim, n_classes,
                 normalization_scale, batch_size=1, n_channels=1, shuffle=True):
        'Initialization'
        self.data_path = data_path
        self.dim = (dim[1], dim[0])
        self.batch_size = batch_size
        self.labels = {ID: phonetic for ID, phonetic in zip(samples_id_list, labels)}
        self.samples_id_list = samples_id_list
        self.labels_alphabet = labels_alphabet
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.normalization_scale = normalization_scale
        self.max_phonetics_length = max_phonetics_length
        self.on_epoch_end()
        self.cache_dataset = True
        self.cache = {}
        self.cache_length = {}
        self.alphabet_index = {ch: i for i, ch in enumerate(labels_alphabet)}
        self.label_encoding_type = ''

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.samples_id_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.samples_id_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.samples_id_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.zeros([len(list_IDs_temp), self.max_phonetics_length])
        input_length = np.zeros([len(list_IDs_temp), 1])
        label_length = np.zeros([len(list_IDs_temp), 1])

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            example, input_len = self.get_example(ID)
            X[i,] = example

            # Store class
            phonetic = self.labels[ID]
            input_length[i] = input_len
            label = text_to_labels(phonetic, self.labels_alphabet, self.max_phonetics_length)
            label_length[i] = len(static_get_phonetics_char_array(phonetic))

            y[i, 0:len(label)] = label

        inputs = {
            'input_layer': X,
            'phonetics': y,
            'input_length': input_length,
            'label_length': label_length,
        }

        outputs = {'ctc_lambda': np.zeros([self.batch_size])}  # dummy data for dummy loss function????

        return inputs, outputs

    # def encode_phonetic_label(self, phonetic):
    #     result = np.zeros((self.max_phonetics_length, len(self.labels_alphabet)), dtype=float)
    #
    #     for i, ch in enumerate(static_get_phonetics_char_array(phonetic)):
    #         result[i, self.alphabet_index[ch]] = 1.0
    #
    #     return result

    def get_example(self, ID):
        if self.cache_dataset and ID in self.cache:
            return self.cache[ID], self.cache_length[ID]

        path = self.get_data_folder(ID) + ID + '.npy'
        example, length = self.scale_and_pad_to_meet_dim(ID, np.load(path))

        if self.cache_dataset:
            self.cache[ID] = example
            self.cache_length[ID] = length

        return example, length

    def scale_and_pad_to_meet_dim(self, ID,  array):
        for i in range(len(self.normalization_scale)):
            array[i,] -= self.normalization_scale[i][Constants.TUPLE_INDEX_MEAN]
            array[i,] /= self.normalization_scale[i][Constants.TUPLE_INDEX_STD]

        length = array.shape[1]
        # print('--------------- ' + ID + ' -> '+str(length))
        array = array.T
        result = np.zeros(self.dim, dtype=np.float64)

        for j in range(array.shape[1]):
            for i in range(array.shape[0]):
                result[i, j] = array[i, j]

        return result, length

    def get_data_folder(self, id):
        if '_' in id:
            id = id[0:id.index('_')]

        return self.data_path + id[0:min(2, len(id))] + Constants.SLASH


# Translation of characters to unique integer values
def text_to_labels(phonetic, alphabet, max_length):
    ret = np.ones(max_length) * -1
    for i, char in enumerate(static_get_phonetics_char_array(phonetic)):
        ret[i] = alphabet.index(char)
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels, alphabet):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)
