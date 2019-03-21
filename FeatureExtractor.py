import librosa
import Constants
import librosa.display
import numpy
import matplotlib.pyplot as plt


def extract_features(method, input_file, output_file, feature_shapes):
    if method == 'mfcc':
        extract_mfcc_features(input_file, output_file, feature_shapes)
    else:
        raise ValueError('Invalid method type')


def extract_mfcc_features(input_file, output_file, feature_shapes):
    audio, _ = librosa.core.load(input_file, mono=True, sr=None)
    mfccs = librosa.feature.mfcc(y=audio,
                                 sr=Constants.AUDIO_FILES_SAMPLING_RATE,
                                 n_mfcc=Constants.MFCC_COEFFICIENT_COUNT)

    numpy.save(output_file, mfccs)
    feature_shapes.append(mfccs.shape)

    plt.clf()
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    path = output_file[output_file.rfind(Constants.SLASH) + 1:]
    plt.title(f'MFCC ({path})')
    plt.tight_layout()
    plt.savefig(output_file + '.png')
    plt.close()
