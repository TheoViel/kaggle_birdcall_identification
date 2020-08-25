import os
import librosa
import soundfile
import numpy as np
from torch.utils.data import Dataset

from params import MEAN, STD, CLASSES
from data.transforms import *


ONE_HOT = np.eye(len(CLASSES))


def compute_melspec(y, params):
    # print('Computing melspec')
    melspec = librosa.feature.melspectrogram(
        y,
        sr=params.sr,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax,
        # hop_length=params.hop_length,
        # n_fft=params.n_fft,
    )

    # print('Moving to db scale')
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    
    return melspec


class BirdDataset(Dataset):
    def __init__(self, df, params, audio_path='', train=True):
        self.train = train
        self.params = params
        self.audio_path = audio_path

        self.wav_transfos = get_wav_transforms(train=train) 
        self.spec_transfos = None

        self.y = np.array([CLASSES.index(c) for c in df['ebird_code']])
        self.paths = df['file_path'].values
        
        self.sample_len = params.duration * params.sr

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        y, sr = soundfile.read(self.audio_path + self.paths[idx])
        
        y = crop_or_pad(y, self.sample_len, train=self.train)

        if self.wav_transfos is not None:
            y = self.wav_transfos(y, self.params.sr)
        
        melspec = compute_melspec(y, self.params)

        if self.spec_transfos is not None:
            melspec = self.spec_transfos(melspec)

        image = mono_to_color(melspec)
        image = resize_and_transpose(image, self.params.img_size)

        return image, ONE_HOT[self.y[idx]]