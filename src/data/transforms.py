# import cv2
import numpy as np
# import albumentations as albu

from audiomentations import *
from data.audiomentations import *

from params import BACKGROUND_PATH


def mono_to_color(X, eps=1e-6):
    X = np.stack([X, X, X], axis=-1)
    
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
    image = (image / 255.0)
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)


def crop_or_pad(y, length, train=True):
    # if len(y) > 0:
        # y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    if len(y) <= length:
        y = np.concatenate([y , np.zeros(length - len(y))])
    else:
        start = np.random.randint(len(y) - length) if train else 0
        y = y[start : start + length]
        
    return y.astype(np.float32)


def get_wav_transforms(train=True):
    if train:
        transforms = Compose([
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),  # Too slow
            AddGaussianSNR(max_SNR=0.5, p=0.5),
            AddBackgroundNoise(sounds_path=BACKGROUND_PATH, min_snr_in_db=0, max_snr_in_db=2, p=0.5),
        ])
    else:
        transforms = None

    return transforms