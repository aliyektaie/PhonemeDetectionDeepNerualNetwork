import os
import numpy as np
import Constants

INPUT_FEATURE_FOLDER_NAME = 'mfcc2'
OUTPUT_FEATURE_FOLDER_NAME = 'mfcc_balanced'
FEATURE_FOLDER = Constants.TRAINING_FOLDER_PATH + 'Features/' \
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
    min_ts = 100
    max_ts = 0
    tss = []

    for ind, file in enumerate(files):
        try:
            if ind % 1000 == 0:
                print('   -> Processed %.1f%s' % (ind * 100.0 / count, '%'))

            if not os.path.isfile(file):
                continue

            data = np.load(file)

            # if data.shape[0] < 125:
            #     continue

            min_ts = min(min_ts, data.shape[0])
            max_ts = max(max_ts, data.shape[0])
            tss.append(data.shape[0])
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
            os.remove(file)
        except:
            continue

    print('Min time step: ' + str(min_ts))
    print('Max time step: ' + str(max_ts))
    print('Time step (avg): ' + str(np.average(np.array(tss))))
    print('Time step (std): ' + str(np.std(np.array(tss))))


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
        all[start:start + data.shape[0], :, :] = data
        start += data.shape[0]

    concatinated = all

    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            data = concatinated[:, i, j]
            means[i, j] = np.mean(data)
            stds[i, j] = np.std(data)

    return means, stds


def main(featureFolder=None):
    global FEATURE_FOLDER
    if featureFolder is not None:
        FEATURE_FOLDER = featureFolder

    files = load_all_files(FEATURE_FOLDER)
    means, stds = load_features_mean_std(files, 10000)

    folder = FEATURE_FOLDER.replace(INPUT_FEATURE_FOLDER_NAME, OUTPUT_FEATURE_FOLDER_NAME)
    np.save(folder + 'mean', means)
    np.save(folder + 'std', stds)

    save_normalized_features(files, means, stds)


if __name__ == '__main__':
    main()
