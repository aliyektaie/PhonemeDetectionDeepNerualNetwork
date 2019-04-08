from DataSet import *
import Constants
from random import shuffle
import numpy as np
import keras

PADDED_FEATURE_SHAPE_INPUT = (185, 20, 3)
NUMBER_OF_SAMPLE_TO_TRAIN_ON = 1024
BATCH_SIZE = 64
FEATURE_FOLDER_NAME = 'mfcc_balanced'
ONE_TO_ONE_DATA_SET_PATH = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_ds.txt'
ONE_TO_ONE_ALPHABET_PATH = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_alphabet.txt'

ALPHABET_CHAR_TO_INDEX = {}
ALPHABET_INDEX_TO_CHAR = {}
ALPHABET = []


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
    def __init__(self, samples, max_output_length, input_shape, batch_size=1):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_output_length = max_output_length
        self.samples = samples
        self.indexes = None
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
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'samples_in_batch': samples_in_batch,
        }

        outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function
        return inputs, outputs

    # Updates indexes after each epoch
    def on_epoch_end(self):
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
            # TODO: Fix this
            dimensions[i, 0] = sample.data.shape[0]

        return dimensions

    def generate_target_labels_length(self, samples_in_batch):
        labels_length = np.zeros((self.batch_size, 1))

        for i, sample in enumerate(samples_in_batch):
            labels_length[i, 0] = len(sample.label)

        return labels_length


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(ALPHABET_CHAR_TO_INDEX[char])
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


def create_one_to_one_dataset():
    if os.path.isfile(ONE_TO_ONE_DATA_SET_PATH):
        return

    print('Creating one-to-one dataset')
    ds = DataSet(Constants.TRAINING_FOLDER_PATH)
    all_entries = []

    for index, entry in enumerate(ds.entries):
        if index % 10000 == 0:
            print('   -> %d of %d' % (index, len(ds.entries)))

        for i, path in enumerate(entry.get_audio_paths()):
            path = path.replace('Audio Files', 'Features/' + FEATURE_FOLDER_NAME).replace('.wav', '.npy')

            if os.path.isfile(path):
                e = TrainingEntry()
                e.word = entry.word
                e.phonetics = entry.phonetics
                e.id = entry.word + '_' + str(i)
                e.data_path = path

                all_entries.append(e)

    alphabet = set()
    with open(ONE_TO_ONE_DATA_SET_PATH, 'w') as file:
        for entry in all_entries:
            file.write(entry.to_str_line())
            file.write('\n')

            symbols = static_get_phonetics_char_array(entry.phonetics)
            for symbol in symbols:
                if symbol not in alphabet:
                    alphabet.add(symbol)

    with open(ONE_TO_ONE_ALPHABET_PATH, 'w') as file:
        file.write('\t'.join(list(alphabet)))

    print('   -> Done')
    print('')


def load_entries_for_training():
    print('Loading training entries')
    entries = []

    with open(ONE_TO_ONE_DATA_SET_PATH) as file:
        line = file.readline()
        while line:
            entries.append(TrainingEntry(line=line))
            line = file.readline()

    shuffle(entries)

    alphabet = None
    with open(ONE_TO_ONE_ALPHABET_PATH) as file:
        alphabet = file.readline().split('\t')

    print('   -> Done')
    print('')

    return entries[0:min(NUMBER_OF_SAMPLE_TO_TRAIN_ON, len(entries))], alphabet


def load_training_data_into_memory(dataset):
    print('Loading dataset in memory')
    for i, entry in enumerate(dataset):
        if i % 1000 == 0 and i > 0:
            print(f'   -> {i} of {len(dataset)}')
        entry.data = np.load(entry.data_path)
        entry.label = text_to_labels(static_get_phonetics_char_array(entry.phonetics))

    print('   -> Done')
    print('')

def load_alphabet_indices():
    global ALPHABET_INDEX_TO_CHAR
    global ALPHABET_CHAR_TO_INDEX

    ALPHABET_CHAR_TO_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}
    ALPHABET_INDEX_TO_CHAR = {i: ch for i, ch in enumerate(ALPHABET)}


def main():
    global ALPHABET

    create_one_to_one_dataset()
    dataset, ALPHABET = load_entries_for_training()
    load_alphabet_indices()
    load_training_data_into_memory(dataset)



if __name__ == '__main__':
    main()
