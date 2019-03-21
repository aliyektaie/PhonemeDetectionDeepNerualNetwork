import Constants


class TrainingEntry:
    def __init__(self, dataset):
        self.word = ''
        self.phonetics = ''
        self.audioCount = 0
        self.dataSet = dataset

    def get_audio_paths(self):
        path = self.dataSet.path + Constants.AUDIO_FILES_FOLDER + Constants.SLASH
        folder = self.word[0: min(len(self.word), 2)]
        path = path + folder + Constants.SLASH

        result = []

        for i in range(0, self.audioCount):
            result.append(path + self.word + '_' + str(i) + '.wav')

        return result


class DataSet:
    def __init__(self, path):
        self.path = path
        self.entries = []

        self.load()

    def load(self):
        result = []

        indexPath = self.path + 'index.txt'
        with open(indexPath) as fileHandle:
            line = fileHandle.readline()
            while line:
                self.entries.append(self.loadEntryFromIndexFileLine(line))
                line = fileHandle.readline()

        return result

    def loadEntryFromIndexFileLine(self, line):
        result = TrainingEntry(self)
        parts = line.strip().split('\t')

        result.word = parts[0]
        result.phonetics = parts[1]
        result.audioCount = int(parts[2])

        return result

