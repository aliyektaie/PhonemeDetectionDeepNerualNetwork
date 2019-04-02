import numpy as np
import Constants
import keras


# Base code is coming from:
#    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# Changes made to adapt to the current project.

class ModelDataSet(keras.utils.Sequence):
    def __init__(self, data_path, samples_id_list, labels, dim, n_classes, normalization_scale,
                 batch_size=32, n_channels=1, shuffle=True):
        'Initialization'
        self.data_path = data_path
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.samples_id_list = samples_id_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.normalization_scale = normalization_scale
        self.on_epoch_end()

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
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            path = self.get_data_folder(ID) + ID + '.npy'
            example = self.scale(np.load(path))
            X[i, ] = example

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def scale(self, array):
        for i in range(len(self.normalization_scale)):
            array[i, ] -= self.normalization_scale[i][Constants.TUPLE_INDEX_MEAN]
            array[i, ] /= self.normalization_scale[i][Constants.TUPLE_INDEX_STD]

        return array

    def get_data_folder(self, id):
        return self.data_path + id[0, min(2, len(id))] + Constants.SLASH
