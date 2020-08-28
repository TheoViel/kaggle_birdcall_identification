import os
import torch
import warnings
import numpy as np


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 2020

DATA_PATH = "../input/"
AUDIO_PATH = "../../../data/audio/"

BACKGROUND_PATH = "../../../data/backgrounds/"

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4
VAL_BS = 32

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

IMG_SIZE = (224, 224)

CLASSES = sorted(os.listdir(AUDIO_PATH))
NUM_CLASSES = len(CLASSES)
