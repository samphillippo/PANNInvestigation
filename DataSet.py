from time import time
import numpy as np

class Base(object):
    def __init__(self, data):
        self.data = data

        # Extract necessary information
        self.audio_names = [item["filename"] for item in self.data]
        self.waveforms = [item["waveform"] for item in self.data]
        self.targets = [item["target"] for item in self.data]
        self.folds = [item["fold"] for item in self.data]

        self.audios_num = len(self.data)
        self.classes_num = self.targets[0].shape[1]  # Assuming targets are tensors

        # Log information

        print('Training number: {}'.format(self.audios_num))

class TrainSampler(object):
    def __init__(self, data, holdout_fold, batch_size, random_seed=1234):
        self.data = data
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Select indexes for training based on fold information
        self.indexes = [idx for idx, item in enumerate(self.data) if item["fold"] != holdout_fold]
        self.audios_num = len(self.indexes)

        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        self.pointer = 0

    def __iter__(self):
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)

                batch_meta.append({
                    'filename': self.data[index]["filename"],
                    'waveform': self.data[index]["waveform"],
                    'target': self.data[index]["target"],
                    'fold': self.data[index]["fold"]
                })
                i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer
        }
        return state

    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']

class EvaluateSampler(object):
    def __init__(self, data, holdout_fold, batch_size, random_seed=1234):
        self.data = data
        self.batch_size = batch_size

        self.indexes = [idx for idx, item in enumerate(self.data) if item["fold"] == holdout_fold]
        self.audios_num = len(self.indexes)

    def __iter__(self):
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for i in batch_indexes:
                batch_meta.append({
                    'filename': self.data[self.indexes[i]]["filename"],
                    'waveform': self.data[self.indexes[i]]["waveform"],
                    'target': self.data[self.indexes[i]]["target"],
                    'fold': self.data[self.indexes[i]]["fold"]
                })

            pointer += batch_size
            yield batch_meta

def collate_fn(list_data_dict):
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict


# class GTZANDataset(object):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         print(idx)
#         return self.data[idx]


#TODO: this is now pointless...
class GTZANDataset(object):
    def __init__(self, data):
        """This class takes the meta of an audio clip as input, and return
        the waveform and target of the audio clip. This class is used by DataLoader.
        Args:
          clip_samples: int
          classes_num: int
        """
        self.data = data

    def __getitem__(self, meta):
        return { 'filename': meta['filename'], 'waveform': meta['waveform'], 'target': meta['target'] }
