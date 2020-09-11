import os
import datetime
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold

from util import *
from params import *
from logger import create_logger
from data.dataset import BirdDataset
from model_zoo.models import get_model
from training.train import fit, predict


class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = 32000
    duration = 5

    # Melspectrogram
    n_mels = 128
    fmin = 20
    fmax = 16000


def train(config, df_train, df_val, fold):
    """
    Trains and validate a model

    Arguments:
        config {Config} -- Parameters
        df_train {pandas dataframe} -- Training metadata
        df_val {pandas dataframe} -- Validation metadata
        fold {int} -- Selected fold

    Returns:
        np array -- Validation predictions
    """

    print(f"    -> {len(df_train)} training birds")
    print(f"    -> {len(df_val)} validation birds")

    seed_everything(config.seed)

    model = get_model(
        config.selected_model, use_msd=config.use_msd, num_classes=NUM_CLASSES
    ).cuda()
    model.zero_grad()

    train_dataset = BirdDataset(df_train, AudioParams, use_conf=config.use_conf)
    val_dataset = BirdDataset(df_val, AudioParams, train=False)

    n_parameters = count_parameters(model)
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        alpha=config.alpha,
        mixup_proba=config.mixup_proba,
        verbose_eval=config.verbose_eval,
        epochs_eval_min=config.epochs_eval_min,
    )

    if config.save:
        save_model_weights(
            model,
            f"{config.selected_model}_{config.name}_{fold}.pt",
            cp_folder=CP_TODAY,
        )

    return pred_val


def k_fold(config, df, df_extra=None):
    """
    Performs a k-fold cross validation

    Arguments:
        config {Config} -- Parameters
        df {pandas dataframe} -- Metadata

    Keyword Arguments:
        df_extra {pandas dataframe or None} -- Metadata of the extra samples to use (default: {None})

    Returns:
        np array -- Out-of-fold predictions
    """

    skf = StratifiedKFold(n_splits=config.k, random_state=config.random_state)
    splits = list(skf.split(X=df, y=df["ebird_code"]))

    pred_oof = np.zeros((len(df), NUM_CLASSES))

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            if df_extra is not None:
                df_train = pd.concat((df_train, df_extra), 0).reset_index(drop=True)

            pred_val = train(config, df_train, df_val, i)
            pred_oof[val_idx] = pred_val

    return pred_oof


class Config:
    """
    Parameter used for training
    """
    # General
    seed = 2020
    verbose = 1
    verbose_eval = 1
    epochs_eval_min = 25
    save = True

    # k-fold
    k = 5
    random_state = 42
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    # selected_model = "resnest50_fast_1s1x64d"
    selected_model = "resnext101_32x8d_wsl"
    # selected_model = 'resnext50_32x4d'
    # selected_model = 'resnest50'

    use_conf = False
    use_extra = True

    # Training
    batch_size = 64
    epochs = 30
    lr = 1e-3
    warmup_prop = 0.05
    val_bs = 64

    if "101" in selected_model or "b5" in selected_model or "b6" in selected_model:
        batch_size = batch_size // 2
        lr = lr / 2

    mixup_proba = 0.5
    alpha = 5

    name = "extra"


if __name__ == "__main__":

    # Data

    df_train = pd.read_csv(DATA_PATH + "train.csv")

    paths = []
    for c, file in df_train[["ebird_code", "filename"]].values:
        path = f"{AUDIO_PATH}{c}/{file[:-4]}.wav"
        paths.append(path)
    df_train["file_path"] = paths

    # Extra Data

    df_extra = pd.read_csv(DATA_PATH + "train_extended.csv")

    paths = []
    for c, file in df_extra[["ebird_code", "filename"]].values:
        path = f"{EXTRA_AUDIO_PATH}{c}/{file[:-4]}.wav"
        paths.append(path)
    df_extra["file_path"] = paths

    if not Config.use_extra:
        df_extra = None

    # Checkpoints folder

    TODAY = str(datetime.date.today())
    CP_TODAY = f"../checkpoints/{TODAY}/"

    if not os.path.exists(CP_TODAY):
        os.mkdir(CP_TODAY)

    # Logger

    create_logger("../output/", f"{TODAY}_{Config.selected_model}_{Config.name}")

    # Training

    pred_oof = k_fold(Config, df_train, df_extra=df_extra)
