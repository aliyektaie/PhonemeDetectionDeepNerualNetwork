from keras import *
from keras.layers import *
from DataSet import *
from ModelDataSet import ModelDataSet
from random import shuffle
import FeatureExtractor

'''
Constants Section
'''
MODEL_NAME_RNN_DENSE_CTC = 'rnn_dense_ctc'
MODEL_NAME_CNN_DENSE_CTC = 'cnn_dense_ctc'
MODEL_NAME_CNN_DENSE_RNN_DENSE_CTC = 'cnn_dense_rnn_dense_ctc'
VALIDATION_PORTION = 0.15
SHOULD_KEEP_DATA_SET_IN_MEMORY = True

''''
Execution Parameters
'''
MODEL_NAME = MODEL_NAME_RNN_DENSE_CTC
FEATURES_FOLDER = Constants.TRAINING_FOLDER_PATH + Constants.FEATURES_FILES_FOLDER + Constants.SLASH + 'mfcc' + Constants.SLASH

TRAINING_SET_ENTRIES_PATH = ''
VALIDATION_SET_ENTRIES_PATH = ''
MODEL_SAVE_LOCATION = ''
MIN_SAMPLE_PHONEME_FREQUENCY = 100
MAX_SAMPLE_PHONEME_FREQUENCY = 10


def create_rnn_dense_ctc_model(model, input_shape, alphabet_size):
    # base code extracted from:
    #   https://github.com/Tony607/keras-image-ocr/blob/master/image-ocr.ipynb
    input_dense_nc = 32
    rnn_nc = 256

    input_data = Input(name='input_layer', shape=input_shape, dtype='float32')
    after_input_layer = Dense(input_dense_nc, activation='relu', name='dense_after_input')(input_data)

    # Two layers of bidirectional GRUs
    gru_1 = GRU(rnn_nc,
                return_sequences=True,
                kernel_initializer='he_normal',
                name='gru1')(after_input_layer)

    gru_1b = GRU(rnn_nc, return_sequences=True,
                 go_backwards=True,
                 kernel_initializer='he_normal',
                 name='gru1_b')(after_input_layer)

    gru1_merged = add([gru_1, gru_1b])

    gru_2 = GRU(rnn_nc,
                return_sequences=True,
                kernel_initializer='he_normal',
                name='gru2')(gru1_merged)

    gru_2b = GRU(rnn_nc,
                 return_sequences=True,
                 go_backwards=True,
                 kernel_initializer='he_normal',
                 name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    after_rnns = Dense(alphabet_size + 1,  # +1 for empty token
                       kernel_initializer='he_normal',
                       name='final_sense_layer')(concatenate([gru_2, gru_2b]))

    y_pred = Activation('softmax', name='softmax')(after_rnns)

    Model(input_data, y_pred).summary()
    return model


def create_cnn_dense_ctc_model(model, input_shape, alphabet_size):
    return model


def create_cnn_dense_rnn_dense_ctc_model(model, input_shape, alphabet_size):
    return model


def create_keras_model(model_name, input_shape, alphabet_size):
    model = Sequential()

    if model_name == MODEL_NAME_RNN_DENSE_CTC:
        model = create_rnn_dense_ctc_model(model, input_shape, alphabet_size)
    elif model_name == MODEL_NAME_CNN_DENSE_RNN_DENSE_CTC:
        model = create_cnn_dense_ctc_model(model, input_shape, alphabet_size)
    elif model_name == MODEL_NAME_CNN_DENSE_CTC:
        model = create_cnn_dense_ctc_model(model, input_shape, alphabet_size)

    return model


def load_dataset():
    print('Loading Data Set Information')
    dataset, alphabet = DataSet.sample_from_data_set(Constants.TRAINING_FOLDER_PATH,
                                                     MIN_SAMPLE_PHONEME_FREQUENCY,
                                                     MAX_SAMPLE_PHONEME_FREQUENCY)
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


def get_max_phonetics_length(phonetics):
    result = 0

    for p in phonetics:
        result = max(len(p), result)

    return result


def load_model_dataset(dataset, alphabet):
    print('Load Network Data Provider')
    phonetics, audio_ids = dataset.get_entries_id_label_list()
    max_phonetics_length = get_max_phonetics_length(phonetics)
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
                                       mean_variance)

    model_dataset_validation = ModelDataSet(FEATURES_FOLDER,
                                            get_tuple_array_dimension(validation_list, 1),
                                            get_tuple_array_dimension(validation_list, 0),
                                            alphabet,
                                            max_phonetics_length,
                                            input_shape,
                                            len(alphabet),
                                            mean_variance)

    model_dataset_train.cache_dataset = SHOULD_KEEP_DATA_SET_IN_MEMORY
    model_dataset_validation.cache_dataset = SHOULD_KEEP_DATA_SET_IN_MEMORY
    input_shape = (None, input_shape[0])

    print('   -> Network input shape: ' + str(input_shape))
    print('   -> Done')
    print('')

    return model_dataset_train, model_dataset_validation, input_shape, mean_variance


def main():
    dataset, alphabet = load_dataset()
    train_set, validation_set, input_shape, features_mean_variance = load_model_dataset(dataset, alphabet)

    model = create_keras_model(MODEL_NAME, input_shape, len(alphabet))


if __name__ == '__main__':
    main()
