import argparse
import sys
import os
import progressbar
import Constants
from DataSet import DataSet
from DataSet import TrainingEntry
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
        print('Extracting Features: ' + method)

        bar = progressbar.ProgressBar(maxval=len(dataset.entries), fd=sys.stdout,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        current = 0
        for entry in dataset.entries:
            index = 0
            for audioFile in entry.get_audio_paths():
                inputFile = audioFile
                outputFile = Constants.TRAINING_FOLDER_PATH + Constants.FEATURES_FILES_FOLDER + Constants.SLASH
                outputFile = outputFile + method + Constants.SLASH
                outputFile = outputFile + entry.word[0: min(len(entry.word), 2)] + Constants.SLASH

                create_folder_if_not_exists(outputFile)
                outputFile = outputFile + entry.word + '_' + str(index) + ".feature"

                FeatureExtractor.extract_features(method, inputFile, outputFile)
                index += 1

            bar.update(current)
            current += 1

        bar.finish()


def main():
    args = init_arguments()

    if args.method:
        extract_features(args.method)


if __name__ == '__main__':
    main()
