from keras import Input, Model
from keras.layers import LSTM, Dense

from BaseNeuralModel import BaseNeuralModel


class AutoEncoderNeuralModel(BaseNeuralModel):
    def init_model(self, input_shape, alphabet_size, max_phonetics_length):
        latent_dim = 256

        # Define an input sequence and process it.
        encoder_inputs = Input(name='input_layer', shape=(None, input_shape[1]))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, alphabet_size))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(alphabet_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.model = model

    def prepare_data_provider(self, provider):
        provider.label_encoding_type = 'sparse_matrix'

    def train(self, train_set, validation_set):
        return self.model.fit_generator(generator=train_set,
                                        epochs=10,

                                        validation_data=validation_set,
                                        use_multiprocessing=True,
                                        workers=6,
                                        verbose=2)
