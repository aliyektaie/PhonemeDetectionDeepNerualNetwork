import librosa
import Constants
import librosa.display
import numpy
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os
from python_speech_features import mfcc


def extract_features(method, input_file, output_file, feature_shapes):
    values = None
    if method == 'mfcc':
        values = extract_mfcc_features(input_file, output_file, feature_shapes)
    elif method == 'mfcc+bandpass':
        values = extract_mfcc_bandpass_features(input_file, output_file, feature_shapes)
    elif method == 'mfcc2':
        values = extract_mfcc_alternative_features(input_file, output_file, feature_shapes)
    else:
        raise ValueError('Invalid method type')

    return values


def apply_audio_filter(signal):
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = Constants.AUDIO_FILES_SAMPLING_RATE
    lowcut = Constants.FEATURE_EXTRACTION_LOW_CUT_OFF_FREQUENCY
    highcut = Constants.FEATURE_EXTRACTION_HIGH_CUT_OFF_FREQUENCY

    # y = butter_lowpass_filter(signal, highcut, fs, order=Constants.FEATURE_EXTRACTION_BAND_PASS_ORDER)
    y = butter_bandpass_filter(signal, lowcut, highcut, fs, order=Constants.FEATURE_EXTRACTION_BAND_PASS_ORDER)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_mfcc_bandpass_features(input_file, output_file, feature_shapes):
    audio, _ = librosa.core.load(input_file, mono=True, sr=None)
    audio = apply_audio_filter(audio)

    mfccs = librosa.feature.mfcc(y=audio,
                                 sr=Constants.AUDIO_FILES_SAMPLING_RATE,
                                 n_mfcc=Constants.MFCC_COEFFICIENT_COUNT)

    numpy.save(output_file, mfccs)
    feature_shapes.append(mfccs.shape)

    if Constants.SAVE_MFCC_IMAGE:
        plt.clf()
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        path = output_file[output_file.rfind(Constants.SLASH) + 1:]
        plt.title(f'MFCC ({path})')
        plt.tight_layout()
        plt.savefig(output_file + '.png')
        plt.close()

    return flatten_features(mfccs)


def add_delta_features_to_mfcc(mfccs):
    result = numpy.zeros([mfccs.shape[1], Constants.MFCC_COEFFICIENT_COUNT, 3], dtype='float')

    mfccs = mfccs.T
    for i in range(mfccs.shape[0]):
        for j in range(mfccs.shape[1]):
            result[i, j, 0] = mfccs[i, j]

    # delta of mfcc
    for i in range(1, mfccs.shape[0]):
        for j in range(mfccs.shape[1]):
            result[i, j, 1] = mfccs[i, j] - mfccs[i - 1, j]

    # delta-delta of mfcc
    for i in range(mfccs.shape[0]):
        for j in range(1, mfccs.shape[1]):
            result[i, j, 2] = result[i, j, 1] - result[i - 1, j, 1]

    return result


def extract_mfcc_features(input_file, output_file, feature_shapes):
    if not os.path.isfile(input_file):
        return []

    audio, _ = librosa.core.load(input_file, mono=True, sr=None)
    mfccs = librosa.feature.mfcc(y=audio,
                                 sr=Constants.AUDIO_FILES_SAMPLING_RATE,
                                 n_mfcc=Constants.MFCC_COEFFICIENT_COUNT)

    delta_added = add_delta_features_to_mfcc(mfccs)
    numpy.save(output_file, delta_added)
    feature_shapes.append(mfccs.shape)

    if Constants.SAVE_MFCC_IMAGE:
        plt.clf()
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        path = output_file[output_file.rfind(Constants.SLASH) + 1:]
        plt.title(f'MFCC ({path})')
        plt.tight_layout()
        plt.savefig(output_file + '.png')
        plt.close()

    return flatten_features(mfccs)


def extract_mfcc_alternative_features(input_file, output_file, feature_shapes):
    if not os.path.isfile(input_file):
        return []

    audio, _ = librosa.core.load(input_file, mono=True, sr=None)
    mfccs = mfcc(audio,
                 Constants.AUDIO_FILES_SAMPLING_RATE,
                 numcep=Constants.MFCC_COEFFICIENT_COUNT,
                 winlen=0.006,
                 winstep=0.006).T

    delta_added = add_delta_features_to_mfcc(mfccs)
    numpy.save(output_file, delta_added)
    feature_shapes.append(mfccs.shape)

    if Constants.SAVE_MFCC_IMAGE:
        plt.clf()
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        path = output_file[output_file.rfind(Constants.SLASH) + 1:]
        plt.title(f'MFCC ({path})')
        plt.tight_layout()
        plt.savefig(output_file + '.png')
        plt.close()

    return flatten_features(mfccs)


def flatten_features(mfccs):
    values = []
    for i in range(0, mfccs.shape[0]):
        for j in range(mfccs.shape[1]):
            values.append(mfccs[i, j])
    return values


def get_input_shape_and_normalizers_of_entry_list(feature_folder, ids):
    x = 0
    y = 0
    stats = None

    SUM = 0
    SUM_OF_SQUARES = 1
    N = 0

    for id in ids:
        word = id[0:id.index('_')]
        path = feature_folder + word[0: min(2, len(word))] + Constants.SLASH + id + '.npy'

        data = numpy.load(path)[:, :, 0].T
        _x, _y = data.shape

        if x == 0:
            x = _x
            stats = numpy.zeros((x, 2), dtype=numpy.float64)
        else:
            x = max(_x, x)
            assert x == _x

        y = max(_y, y)

        N += _y
        for i in range(0, x):
            for j in range(0, _y):
                value = data[i, j]
                stats[i, SUM] += value
                stats[i, SUM_OF_SQUARES] += (value * value)

    mean_variance = []
    for i in range(0, x):
        mean = stats[i, SUM] / N
        variance = (stats[i, SUM_OF_SQUARES] / N) - (mean * mean)
        std = numpy.sqrt(variance)

        mean_variance.append((mean, std))

    return (x, y), mean_variance

# if __name__ == '__main__':
#     for i in range(0, 10):
#         input = f'/Users/yektaie/Desktop/untitled folder/test/tea_{i}.wav'
#         output = f'/Users/yektaie/Desktop/untitled folder/test/tea_{i}f.wav'
#
#         audio, _ = librosa.core.load(input, mono=True, sr=None)
#         audio = apply_audio_filter(audio)
#
#         soundfile.write(output, audio, Constants.AUDIO_FILES_SAMPLING_RATE, subtype=None, endian=None, format=None, closefd=True)
#         # librosa.output.write_wav(output, audio, Constants.AUDIO_FILES_SAMPLING_RATE, norm=False)
