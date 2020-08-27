import os 
import torch 
import random
import numpy as np 
import torch.nn as nn

from sklearn.metrics import f1_score

from params import NUM_CLASSES


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True #False


def save_model_weights(model, filename, verbose=1, cp_folder=''):
    """
    Saves the weights of a PyTorch model
    
    Arguments:
        model {torch module} -- Model to save the weights of
        filename {str} -- Name of the checkpoint
    
    Keyword Arguments:
        verbose {int} -- Whether to display infos (default: {1})
        cp_folder {str} -- Folder to save to (default: {''})
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder=''):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities
    
    Arguments:
        model {torch module} -- Model to load the weights to
        filename {str} -- Name of the checkpoint
    
    Keyword Arguments:
        verbose {int} -- Whether to display infos (default: {1})
        cp_folder {str} -- Folder to load from (default: {''})
    
    Returns:
        torch module -- Model with loaded weights
    """
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )
    return model


def count_parameters(model, all=False):
    """
    Count the parameters of a model
    
    Arguments:
        model {torch module} -- Model to count the parameters of
    
    Keyword Arguments:
        all {bool} -- Whether to include not trainable parameters in the sum (default: {False})
    
    Returns:
        int -- Number of parameters
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


ONE_HOT = np.eye(NUM_CLASSES)

def f1(truth, pred, threshold=0.5, avg='samples'):

    if len(truth.shape) == 1:
        truth = ONE_HOT[truth]

    pred = (pred > threshold).astype(int)

    return f1_score(truth, pred, average=avg)