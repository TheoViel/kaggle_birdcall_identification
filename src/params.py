import os
import torch
import warnings
import numpy as np


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 2020

DATA_PATH = "../input/"
AUDIO_PATH = "../../../data/audio/"
EXTRA_AUDIO_PATH = "../../../data/extra_audios/"

BACKGROUND_PATH = "../../../data/backgrounds/"

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4
VAL_BS = 32

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

IMG_SIZE = (224, 224)

NUM_CLASSES = 264


CLASSES = sorted(os.listdir(AUDIO_PATH))

# try:
#     CLASSES = sorted(os.listdir(AUDIO_PATH))
# except:
#     AUDIO_PATH = "../input/birdsong-recognition/train_audio/"
#     CLASSES = sorted(os.listdir(AUDIO_PATH))

