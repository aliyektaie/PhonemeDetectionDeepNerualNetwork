from AutoEncoderNeuralModel import AutoEncoderNeuralModel
from DataSet import *
from GruCTCNeuralModel import GruCTCNeuralModel
from ModelDataSet import ModelDataSet
from random import shuffle
import FeatureExtractor
import matplotlib.pyplot as plt

'''
Constants Section
'''
MODEL_NAME_RNN_DENSE_CTC = 'rnn_dense_ctc'
MODEL_NAME_AUTO_ENCODERS = 'auto_encoder'
VALIDATION_PORTION = 0.15
SHOULD_KEEP_DATA_SET_IN_MEMORY = True
BATCH_SIZE = 32

''''
Execution Parameters
'''
MODEL_NAME = MODEL_NAME_AUTO_ENCODERS
FEATURES_FOLDER = Constants.TRAINING_FOLDER_PATH + Constants.FEATURES_FILES_FOLDER + Constants.SLASH + 'mfcc' + Constants.SLASH

TRAINING_SET_ENTRIES_PATH = ''
VALIDATION_SET_ENTRIES_PATH = ''
MODEL_SAVE_LOCATION = ''
MIN_SAMPLE_PHONEME_FREQUENCY = 100
MAX_SAMPLE_PHONEME_FREQUENCY = 10


def create_keras_model(model_name, input_shape, alphabet_size, max_phonetics_length):
    model = None

    if model_name == MODEL_NAME_RNN_DENSE_CTC:
        model = GruCTCNeuralModel()
    elif model_name == MODEL_NAME_AUTO_ENCODERS:
        model = AutoEncoderNeuralModel()

    model.init_model(input_shape, alphabet_size, max_phonetics_length)

    return model


def get_database_phonetics_alphabets(dataset):
    result = set()

    for entry in dataset.entries:
        symbols = set(entry.get_phonetics_char_array())
        for symbol in symbols:
            if symbol not in result:
                result.add(symbol)

    temp = list(result)
    result = [' ']
    for t in temp:
        result.append(t)

    return result


def load_dataset():
    print('Loading Data Set Information')
    dataset, _ = DataSet.sample_from_data_set(Constants.TRAINING_FOLDER_PATH,
                                              MIN_SAMPLE_PHONEME_FREQUENCY,
                                              MAX_SAMPLE_PHONEME_FREQUENCY)

    alphabet = get_database_phonetics_alphabets(dataset)
    size = round(dataset.get_audio_file_size() / (1024 * 1024 * 1024), 2)
    print(f'   -> Data Set: {len(dataset.entries)} entries, {size}GB')
    print('   -> Done')
    print('')

    return dataset, alphabet


def get_tuple_array_dimension(list, i):
    result = []

    for t in list:
        result.append(t[i])

    return result


def get_max_phonetics_length(entries):
    result = 0

    for entry in entries:
        result = max(len(entry.get_phonetics_char_array()), result)

    return result


def load_model_dataset(dataset, alphabet):
    print('Load Network Data Provider')
    phonetics, audio_ids = dataset.get_entries_id_label_list()
    max_phonetics_length = get_max_phonetics_length(dataset.entries)
    print('   -> Loading normalization factors')
    input_shape, mean_variance = FeatureExtractor.get_input_shape_and_normalizers_of_entry_list(FEATURES_FOLDER,
                                                                                                audio_ids)

    entries_list = [(p, a) for p, a in zip(phonetics, audio_ids)]
    shuffle(entries_list)

    div = int(len(entries_list) * (1.0 - VALIDATION_PORTION))
    train_list = entries_list[0:div]
    validation_list = entries_list[div:]

    model_dataset_train = ModelDataSet(FEATURES_FOLDER,
                                       get_tuple_array_dimension(train_list, 1),
                                       get_tuple_array_dimension(train_list, 0),
                                       alphabet,
                                       max_phonetics_length,
                                       input_shape,
                                       len(alphabet),
                                       mean_variance,
                                       batch_size=BATCH_SIZE)

    model_dataset_validation = ModelDataSet(FEATURES_FOLDER,
                                            get_tuple_array_dimension(validation_list, 1),
                                            get_tuple_array_dimension(validation_list, 0),
                                            alphabet,
                                            max_phonetics_length,
                                            input_shape,
                                            len(alphabet),
                                            mean_variance,
                                            batch_size=BATCH_SIZE)

    model_dataset_train.cache_dataset = SHOULD_KEEP_DATA_SET_IN_MEMORY
    model_dataset_validation.cache_dataset = SHOULD_KEEP_DATA_SET_IN_MEMORY
    input_shape = (input_shape[1], input_shape[0])

    print('   -> Network input shape: ' + str(input_shape))
    print('   -> Done')
    print('')

    return model_dataset_train, model_dataset_validation, input_shape, mean_variance, max_phonetics_length


def main():
    dataset, alphabet = load_dataset()
    train_set, validation_set, input_shape, features_mean_variance, max_phonetics_length = load_model_dataset(dataset,
                                                                                                              alphabet)

    model = create_keras_model(MODEL_NAME, input_shape, len(alphabet), max_phonetics_length)
    model.prepare_data_provider(train_set)
    model.prepare_data_provider(validation_set)

    print('Start training')
    history = model.train(train_set, validation_set)

    print('Training finished')
    # list all data in history
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


if __name__ == '__main__':
    main()
