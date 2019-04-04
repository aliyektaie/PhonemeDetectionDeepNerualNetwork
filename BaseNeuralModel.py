class BaseNeuralModel:
    def __init__(self):
        self.model = None

    def init_model(self,input_shape, alphabet_size, max_phonetics_length):
        pass

    def prepare_data_provider(self, provider):
        pass

    def train(self, train_set, validation_set):
        pass
