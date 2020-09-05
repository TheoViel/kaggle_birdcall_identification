import os
import pickle
import librosa
import soundfile
import numpy as np
from torch.utils.data import Dataset

from data.transforms import *
from params import MEAN, STD, CLASSES


ONE_HOT = np.eye(len(CLASSES))
CONF_PATH = "../output/preds_oof.pkl"


def compute_melspec(y, params):
    melspec = librosa.feature.melspectrogram(
        y,
        sr=params.sr,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec


class BirdDataset(Dataset):
    def __init__(self, df, params, audio_path="", train=True, use_conf=False):
        self.train = train
        self.params = params
        self.audio_path = audio_path

        self.wav_transfos = get_wav_transforms() if train else None
        # self.wav_transfos = AudioAugmentation(p_effects=0.5, p_noise=0.5) if train else None

        self.spec_transfos = None

        self.y = np.array([CLASSES.index(c) for c in df["ebird_code"]])
        self.paths = df["file_path"].values

        self.sample_len = params.duration * params.sr

        self.use_conf = use_conf
        if use_conf:
            with open(CONF_PATH, "rb") as file:
                self.confidences = pickle.load(file)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        y, sr = soundfile.read(self.audio_path + self.paths[idx])

        if self.use_conf:
            confs = self.confidences[self.paths[idx]][:, self.y[idx]]
            if len(confs):
                confs = confs / np.sum(confs)
            else:
                confs = None
        else:
            confs = None

        y = crop_or_pad(
            y, self.sample_len, sr=self.params.sr, train=self.train, probs=confs
        )

        if self.wav_transfos is not None:
            y = self.wav_transfos(y, self.params.sr)

        melspec = compute_melspec(y, self.params)

        if self.spec_transfos is not None:
            melspec = self.spec_transfos(melspec)

        image = mono_to_color(melspec)
        image = resize(image, self.params.img_size)
        image = normalize(image, mean=None, std=None)

        return image, ONE_HOT[self.y[idx]]
