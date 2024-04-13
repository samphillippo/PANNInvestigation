
#TODO: sampler???

#SHOULD RETURN
"""
'audio_name': str,
'waveform': (clip_samples,),
'target': (classes_num,)}           one-hot? wants as float32???
"""

class GTZANDataset(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
