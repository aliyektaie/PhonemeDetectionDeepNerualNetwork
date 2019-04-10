import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from random import shuffle
from ConvRnnKeras import TrainingEntry

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

PADDED_FEATURE_SHAPE_INPUT = (185, 20, 3)
# NUMBER_OF_SAMPLE_TO_TRAIN_ON = 404355
NUMBER_OF_SAMPLE_TO_TRAIN_ON = 1000
NUMBER_OF_RNN_LAYERS = 2
BATCH_SIZE = 256
VALIDATION_PORTION = 0.0
MAX_PHONETICS_LEN = 30
MIN_PHONETICS_LEN = 3
FEATURE_FOLDER_NAME = 'mfcc_balanced'
ONE_TO_ONE_DATA_SET_PATH = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_ds.txt'
ONE_TO_ONE_ALPHABET_PATH = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/oto_alphabet.txt'
OUTPUT_DIR = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Output'

ALPHABET_CHAR_TO_INDEX = {}
ALPHABET_INDEX_TO_CHAR = {}
ALPHABET = []


def load_entries_for_training():
    print('Loading training entries')
    entries = []

    with open(ONE_TO_ONE_DATA_SET_PATH) as file:
        line = file.readline()
        while line:
            entries.append(TrainingEntry(line=line))
            line = file.readline()

    null_count = NUMBER_OF_SAMPLE_TO_TRAIN_ON // 100
    for i in range(null_count):
        e = TrainingEntry()
        e.label = [len(ALPHABET), len(ALPHABET), len(ALPHABET)]
        e.phonetics = ''
        e.data_path = None
        e.data = np.zeros(PADDED_FEATURE_SHAPE_INPUT)
        e.word = ''

        entries.append(e)

    shuffle(entries)

    alphabet = None
    with open(ONE_TO_ONE_ALPHABET_PATH) as file:
        alphabet = file.readline().split('\t')

    print('   -> Done')
    print('')

    return entries[0:min(NUMBER_OF_SAMPLE_TO_TRAIN_ON + null_count, len(entries))], alphabet


def load_training_data_into_memory(dataset):
    print('Loading dataset in memory')
    for i, entry in enumerate(dataset):
        if i % 1000 == 0 and i > 0:
            print(f'   -> {i} of {len(dataset)}')

        if entry.data_path is not None:
            entry.data = np.load(entry.data_path)
            entry.label = None # text_to_labels(static_get_phonetics_char_array(entry.phonetics))

    print('   -> Done')
    print('')


def load_alphabet_indices():
    global ALPHABET_INDEX_TO_CHAR
    global ALPHABET_CHAR_TO_INDEX

    ALPHABET_CHAR_TO_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}
    ALPHABET_INDEX_TO_CHAR = {i: ch for i, ch in enumerate(ALPHABET)}


class PhonemeRecognizerModel(nn.Module):
    def __init__(self):
        super(PhonemeRecognizerModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # inner = Conv2D(conv_filters, kernel_2d_size, padding='same',
        #                activation=act, kernel_initializer='he_normal',
        #                name='conv2')(input_data)
        #
        # inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        #
        # inner = Conv2D(conv_filters, kernel_2d_size, padding='same',
        #                activation=act, kernel_initializer='he_normal',
        #                name='conv3')(inner)
        #
        # inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

        # 64 input features, 10 output features for our 10 defined classes

    def forward(self, x):
        # make it channel first
        x = x.permute([2, 0, 1])

        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool1(x)

        # Computes the activation of the second convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv2(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool2(x)

def main():
    global ALPHABET

    dataset, ALPHABET = load_entries_for_training()
    load_alphabet_indices()

    load_training_data_into_memory(dataset)

    model = PhonemeRecognizerModel()
    model.forward(torch.from_numpy(dataset[0].data))


if __name__ == '__main__':
    main()
