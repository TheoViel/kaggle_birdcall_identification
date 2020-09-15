class Config:
    """
    ResNest50 - LB 0.612
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
    selected_model = "resnest50_fast_1s1x64d"

    use_conf = False
    use_extra = False

    # Training
    batch_size = 64
    epochs = 30 if use_extra else 40
    lr = 1e-3
    warmup_prop = 0.05
    val_bs = 64

    mixup_proba = 0.5
    alpha = 5

    name = "mixup5"


class Config:
    """
    ResNest50 - LB 0.617
    Uses confidences computed by the LB 0.612 model
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
    selected_model = "resnest50_fast_1s1x64d"

    use_conf = True
    use_extra = False

    # Training
    batch_size = 64
    epochs = 30 if use_extra else 40
    lr = 1e-3
    warmup_prop = 0.05
    val_bs = 64

    mixup_proba = 0.5
    alpha = 5

    name = "conf"


class Config:
    """
    ResNext50 - LB 0.606
    Uses extra data
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
    selected_model = 'resnext50_32x4d'

    use_conf = False
    use_extra = True

    # Training
    batch_size = 64
    epochs = 30 if use_extra else 40
    lr = 1e-3
    warmup_prop = 0.05
    val_bs = 64

    mixup_proba = 0.5
    alpha = 5

    name = "conf"

class Config:
    """
    ResNext101 - LB 0.606
    Uses extra data
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
    selected_model = "resnext101_32x8d_wsl"

    use_conf = False
    use_extra = True

    # Training
    batch_size = 32
    epochs = 40
    lr = 5e-4
    warmup_prop = 0.05
    val_bs = 64

    mixup_proba = 0.5
    alpha = 5

    name = "extra"