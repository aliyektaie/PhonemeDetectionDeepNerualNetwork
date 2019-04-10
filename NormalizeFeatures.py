import os
import numpy as np

INPUT_FEATURE_FOLDER_NAME = 'mfcc'
OUTPUT_FEATURE_FOLDER_NAME = 'mfcc_balanced'
FEATURE_FOLDER = '/Volumes/Files/Georgetown/AdvancedMachineLearning/Project Data/DataSet/Features/' \
                 + INPUT_FEATURE_FOLDER_NAME + '/'


def load_all_files(path):
    result = []

    for dirname, dirnames, filenames in os.walk(path):
        # traverse sub directories
        for subdirname in dirnames:
            files = load_all_files(os.path.join(dirname, subdirname))
            for file in files:
                result.append(file)

        # load all file names
        for filename in filenames:
            if '.npy' not in filename:
                continue

            result.append(os.path.join(dirname, filename))

    return result


def save_normalized_features(files, means, stds):
    print('Loading all features files')
    count = len(files)

    for ind, file in enumerate(files):
        if ind % 1000 == 0:
            print('   -> Processed %.1f%s' % (ind * 100.0 / count, '%'))

        data = np.load(file)
        for i in range(means.shape[0]):
            for j in range(means.shape[1]):
                data[:, i, j] -= means[i, j]
                if stds[i, j] != 0.:
                    data[:, i, j] /= stds[i, j]

        path = file.replace(INPUT_FEATURE_FOLDER_NAME, OUTPUT_FEATURE_FOLDER_NAME)
        folder = path[0:path.rfind('/')]
        if not os.path.isdir(folder):
            os.mkdir(folder)

        np.save(path, data)


def load_features_mean_std(files, count):
    means = None
    stds = None
    print('Loading all features files')
    files = files[0:count]
    concatinated = []
    lenght = 0

    for i, file in enumerate(files):
        if i % 100 == 0:
            print(i)
        data = np.load(file)
        concatinated.append(data)
        lenght += data.shape[0]

        if means is None:
            means = np.zeros((data.shape[1], data.shape[2]))
            stds = np.zeros((data.shape[1], data.shape[2]))

    all = np.zeros((lenght, means.shape[0], means.shape[1]))
    start = 0

    for data in concatinated:
        all[start:start+data.shape[0],:,:] = data
        start += data.shape[0]

    concatinated = all

    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            data = concatinated[:, i, j]
            means[i, j] = np.mean(data)
            stds[i, j] = np.std(data)

    return means, stds


def main():
    files = load_all_files(FEATURE_FOLDER)
    means, stds = load_features_mean_std(files, 10000)

    save_normalized_features(files, means, stds)


if __name__ == '__main__':
    main()