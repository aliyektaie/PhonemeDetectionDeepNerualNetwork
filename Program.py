import argparse
import os
import Constants
from DataSet import DataSet
import FeatureExtractor


def init_arguments():
    parser = argparse.ArgumentParser(description='Advanced Machine Learning Project')
    parser.add_argument('--extract-feature', metavar='method', type=str, nargs=1, dest='method',
                        help='Extract features of training set. The feature extraction method must be specified.'
                             ' \'method\' include: mfcc')

    return parser.parse_args()


def create_folder_if_not_exists(folder):
    try:
        os.makedirs(folder)
    except:
        pass


def extract_features(methods):
    print('Loading DataSet from: ' + Constants.TRAINING_FOLDER_PATH)
    dataset = DataSet(Constants.TRAINING_FOLDER_PATH)

    for method in methods:
        feature_shapes = []
        print('Extracting Features: ' + method)

        current = 0
        count = len(dataset.entries)

        for entry in dataset.entries:
            index = 0
            for audioFile in entry.get_audio_paths():
                inputFile = audioFile
                outputFile = Constants.TRAINING_FOLDER_PATH + Constants.FEATURES_FILES_FOLDER + Constants.SLASH
                outputFile = outputFile + method + Constants.SLASH
                outputFile = outputFile + entry.word[0: min(len(entry.word), 2)] + Constants.SLASH

                create_folder_if_not_exists(outputFile)
                outputFile = outputFile + entry.word + '_' + str(index)

                FeatureExtractor.extract_features(method, inputFile, outputFile, feature_shapes)
                index += 1

            current += 1
            if current % 50 == 0:
                p = current * 100.0 / count
                p = str(round(p, 2))
                print(f'{current} of {count} (%{p})')

        max_shape = [0, 0]
        for shape in feature_shapes:
            column = shape[1]
            max_shape[1] = max(max_shape[1], column)
            column = shape[0]
            max_shape[0] = max(max_shape[0], column)

        print(f'Max Shape ({method}): ({max_shape[0]}, {max_shape[1]})')

def main():
    args = init_arguments()

    if args.method:
        extract_features(args.method)


if __name__ == '__main__':
    main()
