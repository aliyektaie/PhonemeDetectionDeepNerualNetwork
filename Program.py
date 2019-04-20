import argparse
import os
import Constants
from DataSet import DataSet
import FeatureExtractor
import numpy as np
from matplotlib import pyplot as plt
import NormalizeFeatures


def init_arguments():
    parser = argparse.ArgumentParser(description='Advanced Machine Learning Project')
    parser.add_argument('--extract-feature', metavar='method', type=str, nargs=1, dest='method',
                        help='Extract features of training set. The feature extraction method must be specified.'
                             ' \'method\' include: mfcc')
    parser.add_argument('--normalize-feature', metavar='normalize', type=str, nargs=1, dest='normalize',
                        help='Normalize features of training set. This normalization scales features to z score.' +
                             ' The parameter defines the feature input folder.')

    return parser.parse_args()


def create_folder_if_not_exists(folder):
    try:
        os.makedirs(folder)
    except:
        pass


def make_histogram(values):
    result = np.zeros(101)

    for v in values:
        i = int(v / 20)
        result[i + 50] += 1

    return result


def extract_features(methods):
    print('Loading DataSet from: ' + Constants.TRAINING_FOLDER_PATH)
    dataset = DataSet(Constants.TRAINING_FOLDER_PATH)

    for method in methods:
        feature_shapes = []
        print('Extracting Features: ' + method)

        current = 0
        count = len(dataset.entries)
        feature_folder = ''
        sumFeatures = 0.0
        sumOfSquareFeatures = 0.0
        bins = np.linspace(-1000, 1000, 101)
        historgram = np.zeros(101)
        minFeature = 0
        maxFeature = 0

        for entry in dataset.entries:
            index = 0
            for audioFile in entry.get_audio_paths():
                inputFile = audioFile
                outputFile = Constants.TRAINING_FOLDER_PATH + Constants.FEATURES_FILES_FOLDER + Constants.SLASH
                outputFile = outputFile + method + Constants.SLASH
                feature_folder = outputFile
                outputFile = outputFile + entry.word[0: min(len(entry.word), 2)] + Constants.SLASH
                create_folder_if_not_exists(outputFile)
                outputFile = outputFile + entry.word + '_' + str(index)

                values = FeatureExtractor.extract_features(method, inputFile, outputFile, feature_shapes)
                historgram = np.add(historgram, make_histogram(values))
                for v in values:
                    sumFeatures += v
                    sumOfSquareFeatures += (v * v)

                    if sumOfSquareFeatures < 0:
                        print('sum of square became negative')

                    minFeature = min(minFeature, v)
                    maxFeature = max(maxFeature, v)

                index += 1

            current += 1
            if current % 50 == 0:
                p = current * 100.0 / count
                p = str(round(p, 2))
                print(f'{current} of {count} (%{p})  -> {entry.word}')

        max_shape = [0, 0]
        for shape in feature_shapes:
            column = shape[1]
            max_shape[1] = max(max_shape[1], column)
            column = shape[0]
            max_shape[0] = max(max_shape[0], column)

        mu = sumFeatures / len(dataset.entries)
        sigma_2 = sumOfSquareFeatures / len(dataset.entries) - mu
        save_feature_extraction_report(feature_folder, historgram, max_shape, method, bins, mu, sigma_2, minFeature,
                                       maxFeature)


def save_feature_extraction_report(folder, histogram, max_shape, method, bins, mu, sigma_2, _min, _max):
    report = f'Max Shape ({method}): ({max_shape[0]}, {max_shape[1]})\n' \
             + f'Mean: {mu}\n' \
             + f'Variance: {sigma_2}\n' \
             + f'Min: {_min}\n' \
             + f'Max: {_max}'
    print(report)

    report += '\n\nFeature Value Distribution'
    report += '\n\n'
    binsPlus = []
    for b, v in zip(bins, histogram):
        report += f'{abs(b)}-{abs(b + 20)},{v}\n'
        binsPlus.append(b + 1000)
    fig = plt.figure(figsize=(15, 6), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    labels = []
    for s in range(0, 101):
        start = (s - 50) * 20
        if s % 5 == 0:
            labels.append(f'{start}')
        else:
            labels.append('')

    plt.bar(range(0, 101), np.add(histogram, np.ones(101)), log=True, tick_label=labels)
    plt.title("Feature Values Distribution")
    ax.set_yscale('log')
    plt.savefig(folder + "distribution.png")

    file = open(folder + 'report.txt', "w")

    file.write(report)

    file.close()


def normalize_features(features):
    NormalizeFeatures.main(features[0])


def main():
    args = init_arguments()

    if args.method:
        # extract_features(args.method)
        print('Extracting Features')
    elif args.normalize:
        # normalize_features(args.normalize)
        print('Normalizing Features')


if __name__ == '__main__':
    main()
