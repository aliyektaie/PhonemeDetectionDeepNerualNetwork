from keras import *
from keras.layers import *
from DataSet import *

'''
Constants Section
'''
MODEL_NAME_RNN_DENSE_CTC = 'rnn_dense_ctc'
MODEL_NAME_CNN_DENSE_CTC = 'cnn_dense_ctc'
MODEL_NAME_CNN_DENSE_RNN_DENSE_CTC = 'cnn_dense_rnn_dense_ctc'

''''
Execution Parameters
'''
MODEL_NAME = MODEL_NAME_RNN_DENSE_CTC
TRAINING_SET_ENTRIES_PATH = ''
VALIDATION_SET_ENTRIES_PATH = ''
MODEL_SAVE_LOCATION = ''


def create_rnn_dense_ctc_model(model, input_shape):
    return model


def create_cnn_dense_ctc_model(model, input_shape):
    return model


def create_cnn_dense_rnn_dense_ctc_model(model, input_shape):
    return model


def create_keras_model(model_name, input_shape):
    model = Sequential()

    if model == MODEL_NAME_RNN_DENSE_CTC:
        model = create_rnn_dense_ctc_model(model, input_shape)
    elif model == MODEL_NAME_CNN_DENSE_RNN_DENSE_CTC:
        model = create_cnn_dense_ctc_model(model, input_shape)
    elif model == MODEL_NAME_CNN_DENSE_CTC:
        model = create_cnn_dense_ctc_model(model, input_shape)

    return model


def main():
    dataset = DataSet.sample_from_data_set(Constants.TRAINING_FOLDER_PATH, 100, 200)
    input_shape = (1, 2)
    model = create_keras_model(MODEL_NAME, input_shape)


if __name__ == '__main__':
    main()
