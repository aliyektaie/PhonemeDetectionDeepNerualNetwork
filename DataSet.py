import Constants
import random
import os


def static_get_phonetics_char_array(phonetics):
    result = []
    prefix = ''

    for ch in phonetics:
        if ch == 'ˌ':
            prefix = 'ˌ'
        elif ch == 'ː':
            result[len(result) - 1] += 'ː'
            prefix = ''
        else:
            result.append(prefix + ch)
            prefix = ''

    return result


class TrainingEntry:
    def __init__(self, dataset):
        self.word = ''
        self.phonetics = ''
        self.audioCount = 0
        self.dataSet = dataset
        self.audioPaths = None

    def get_audio_paths(self):
        if self.audioPaths is not None:
            return self.audioPaths

        path = self.dataSet.path + Constants.AUDIO_FILES_FOLDER + Constants.SLASH
        folder = self.word[0: min(len(self.word), 2)]
        path = path + folder + Constants.SLASH

        result = []

        for i in range(0, self.audioCount):
            p = path + self.word + '_' + str(i) + '.wav'
            if os.path.isfile(p):
                result.append(p)

        self.audioPaths = result
        return result

    def get_phonetics_char_array(self):
        return static_get_phonetics_char_array(self.phonetics)


class DataSet:
    def __init__(self, path):
        self.path = path
        self.entries = []

        self.load()

    def load(self):
        result = []

        indexPath = self.path
        if not indexPath.endswith('.txt'):
            indexPath = indexPath + 'index.txt'

        with open(indexPath) as fileHandle:
            line = fileHandle.readline()
            while line:
                e = self.loadEntryFromIndexFileLine(line)
                if e is not None:
                    self.entries.append(e)
                line = fileHandle.readline()

        return result

    def loadEntryFromIndexFileLine(self, line):
        result = TrainingEntry(self)
        parts = line.strip().split('\t')

        result.word = parts[0]
        result.phonetics = self.prepare_phonetics(parts[1])
        result.audioCount = int(parts[2])

        if ',' in result.phonetics:
            return None

        return result

    def get_audio_file_size(self):
        result = 0
        for entry in self.entries:
            for path in entry.get_audio_paths():
                result += os.path.getsize(path)

        return result

    def get_entries_id_label_list(self):
        phonetics = []
        audio_id = []

        for entry in self.entries:
            i = 0
            for path in entry.get_audio_paths():
                phonetics.append(entry.phonetics)
                audio_id.append(entry.word + '_' + str(i))
                i += 1

        return phonetics, audio_id

    @classmethod
    def sample_from_data_set(cls, path, min_count_of_each_phoneme, max_count_of_each_phoneme):
        symbols_to_word_mapper = {}
        dataset = DataSet(path)
        entries = []

        indexPath = path + 'index.txt'
        alphabet = []
        with open(indexPath) as fileHandle:
            line = fileHandle.readline()
            while line:
                entry = dataset.loadEntryFromIndexFileLine(line)
                if entry is not None:
                    entries.append(entry)
                    symbols = set(entry.get_phonetics_char_array())
                    for symbol in symbols:
                        if symbol not in alphabet:
                            alphabet.append(symbol)
                            symbols_to_word_mapper[symbol] = []

                        symbols_to_word_mapper[symbol].append(entry)

                line = fileHandle.readline()

        dataset, alphabet = cls.keep_samples_to_meet_min_count(dataset, alphabet, min_count_of_each_phoneme,
                                                               max_count_of_each_phoneme, symbols_to_word_mapper)
        return dataset, alphabet

    @classmethod
    def prepare_phonetics(cls, phonetics):
        phonetics = phonetics.replace('ˈ', '')
        phonetics = phonetics.replace('¦', '')
        # phonetics = phonetics.replace('ː', '')

        return phonetics

    @classmethod
    def keep_samples_to_meet_min_count(cls, dataset, alphabet, min_count_of_each_phoneme, max_count_of_each_phoneme,
                                       symbols_to_word_mapper):

        phonemes_count = [(symbol, len(symbols_to_word_mapper[symbol])) for symbol in alphabet]
        sample_count = {symbol: 0 for symbol in alphabet}
        phonemes_count.sort(key=lambda tup: tup[1])
        new_alphabet = []

        entries = []
        for symbol, count in phonemes_count:
            if count > min_count_of_each_phoneme:
                entries_with_symbol = symbols_to_word_mapper[symbol]
                random.Random(0).shuffle(entries_with_symbol)

                for entry in entries_with_symbol:
                    if len(entry.get_audio_paths()) == 0:
                        continue

                    symbols = set(entry.get_phonetics_char_array())

                    should_add = True
                    for s in symbols:
                        should_add = should_add and sample_count[s] < max_count_of_each_phoneme
                        should_add = should_add and len(symbols_to_word_mapper[s]) > min_count_of_each_phoneme

                    if should_add:
                        entries.append(entry)

                        for s in symbols:
                            sample_count[s] += 1

                            if s not in new_alphabet:
                                new_alphabet.append(s)

        reduced_dataset = DataSet(dataset.path)
        reduced_dataset.entries = entries

        for entry in entries:
            entry.dataSet = reduced_dataset

        return reduced_dataset, new_alphabet
