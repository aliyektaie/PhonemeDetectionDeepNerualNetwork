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

''''
Execution Parameters
'''
MODEL_NAME = MODEL_NAME_RNN_DENSE_CTC
FEATURES_FOLDER = Constants.TRAINING_FOLDER_PATH + Constants.FEATURES_FILES_FOLDER + Constants.SLASH + 'mfcc' + Constants.SLASH

TRAINING_SET_ENTRIES_PATH = ''
VALIDATION_SET_ENTRIES_PATH = ''
MODEL_SAVE_LOCATION = ''
MIN_SAMPLE_PHONEME_FREQUENCY = 100
MAX_SAMPLE_PHONEME_FREQUENCY = 120


def create_rnn_dense_ctc_model(model, input_shape):
    return model


def create_cnn_dense_ctc_model(model, input_shape):
    return model


def create_cnn_dense_rnn_dense_ctc_model(model, input_shape):
    return model


def create_keras_model(model_name, input_shape):
    model = Sequential()

    if model_name == MODEL_NAME_RNN_DENSE_CTC:
        model = create_rnn_dense_ctc_model(model, input_shape)
    elif model_name == MODEL_NAME_CNN_DENSE_RNN_DENSE_CTC:
        model = create_cnn_dense_ctc_model(model, input_shape)
    elif model_name == MODEL_NAME_CNN_DENSE_CTC:
        model = create_cnn_dense_ctc_model(model, input_shape)

    return model


def load_dataset():
    print('Loading Data Set Information')
    dataset, alphabet = DataSet.sample_from_data_set(Constants.TRAINING_FOLDER_PATH,
                                                     MIN_SAMPLE_PHONEME_FREQUENCY,
                                                     MAX_SAMPLE_PHONEME_FREQUENCY)
    size = round(dataset.get_audio_file_size() / (1024 * 1024 * 1024), 2)
    print(f'   -> Data Set: {len(dataset.entries)} entries, {size}GB')
    print('   -> Done')

    return dataset, alphabet


def get_tuple_array_dimension(list, i):
    result = []

    for t in list:
        result.append(t[i])

    return result


def load_model_dataset(dataset, alphabet):
    phonetics, audio_ids = dataset.get_entries_id_label_list()
    input_shape, mean_variance = FeatureExtractor.get_input_shape_and_normalizers_of_entry_list(FEATURES_FOLDER,
                                                                                                audio_ids)
    print('input shape')
    print(input_shape)

    entries_list = [(p, a) for p, a in zip(phonetics, audio_ids)]
    shuffle(entries_list)

    div = int(len(entries_list) * (1.0 - VALIDATION_PORTION))
    train_list = entries_list[0:div]
    validation_list = entries_list[div:]

    model_dataset_train = ModelDataSet(dataset.path,
                                       get_tuple_array_dimension(train_list, 1),
                                       get_tuple_array_dimension(train_list, 0),
                                       input_shape,
                                       len(alphabet),
                                       mean_variance)

    model_dataset_validation = ModelDataSet(dataset.path,
                                            get_tuple_array_dimension(validation_list, 1),
                                            get_tuple_array_dimension(validation_list, 0),
                                            input_shape,
                                            len(alphabet),
                                            mean_variance)

    return model_dataset_train, model_dataset_validation, input_shape, mean_variance


def main():
    dataset, alphabet = load_dataset()
    train_set, validation_set, input_shape, features_mean_variance = load_model_dataset(dataset, alphabet)
    model = create_keras_model(MODEL_NAME, input_shape)


if __name__ == '__main__':
    main()
