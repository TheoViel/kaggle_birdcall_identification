# import cv2
import pysndfx
import numpy as np
# import albumentations as albu
from audiomentations import *
from params import BACKGROUND_PATH


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def resize(image, size=None):
    if size is not None:
        h, w, _ = image.shape
        new_w, new_h = int(w * size / h), size
        # image = cv2.resize(image, (new_w, new_h))

    return image


def normalize(image, mean=None, std=None):
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)


def crop_or_pad(y, length, sr, train=True, probs=None):
    # if len(y) > 0:
    # y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (
                np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            )
            start = int(sr * (start))

        y = y[start : start + length]

    return y.astype(np.float32)


def get_wav_transforms():
    transforms = Compose(
        [
            AddGaussianSNR(max_SNR=0.5, p=0.5),
            AddBackgroundNoise(
                sounds_path=BACKGROUND_PATH, min_snr_in_db=0, max_snr_in_db=2, p=0.5
            ),
        ]
    )

    return transforms


class AudioAugmentation:
    def __init__(self, p_effects=0.5, p_noise=0.5):
        self.p_effects = p_effects

        self.noise_transfos = Compose(
            [
                AddGaussianSNR(max_SNR=0.5, p=p_noise),
                AddBackgroundNoise(
                    sounds_path=BACKGROUND_PATH, min_snr_in_db=0, max_snr_in_db=2, p=p_noise
                ),
            ]
        )

    def __call__(self, y, sr):
        y = self.noise_transfos(y, sr)

        if np.random.uniform() < self.p_effects:
            effects_chain = (
                pysndfx.AudioEffectsChain()
                .reverb(
                    reverberance=random.randrange(50),
                    room_scale=random.randrange(50),
                    stereo_depth=random.randrange(50),
                )
                .pitch(shift=random.randrange(-300, 300))
                .overdrive(gain=random.randrange(2, 20))
            )

            y = effects_chain(y)

        return y
