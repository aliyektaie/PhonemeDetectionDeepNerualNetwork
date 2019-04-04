from keras import *
from keras.layers import *
from keras.optimizers import SGD


from BaseNeuralModel import BaseNeuralModel


class GruCTCNeuralModel(BaseNeuralModel):
    def init_model(self, input_shape, alphabet_size, max_phonetics_length):
        # base code extracted from:
        #   https://github.com/Tony607/keras-image-ocr/blob/master/image-ocr.ipynb
        rnn_nc = 256

        input_data = Input(name='input_layer', shape=input_shape, dtype='float32')

        # Two layers of bidirectional GRUs
        gru_1 = GRU(rnn_nc,
                    return_sequences=True,
                    kernel_initializer='he_normal',
                    name='gru1')(input_data)

        gru_1b = GRU(rnn_nc, return_sequences=True,
                     go_backwards=True,
                     kernel_initializer='he_normal',
                     name='gru1_b')(input_data)

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

        y_pred_layer = Activation('softmax', name='softmax')(after_rnns)

        # adding the CTC part
        labels = Input(name='phonetics', shape=[max_phonetics_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_lambda')(
            [y_pred_layer, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        # model.summary()

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc_lambda': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        # model.compile(loss={'ctc_lambda': lambda y_true, y_pred: y_pred}, optimizer='adam')

        self.model = model

    def prepare_data_provider(self, provider):
        provider.label_encoding_type = 'sequence'

    def train(self, train_set, validation_set):
        return self.model.fit_generator(generator=train_set,
                                        epochs=10,

                                        validation_data=validation_set,
                                        use_multiprocessing=True,
                                        workers=6,
                                        verbose=2)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    ret = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    return ret
