# Kaggle Birdcall Identification

Code for the [Cornell Birdcall Identification competition](https://www.kaggle.com/c/birdsong-recognition)

## Context

> Do you hear the birds chirping outside your window? Over 10,000 bird species occur in the world, and they can be found in nearly every environment, from untouched rainforests to suburbs and even cities. Birds play an essential role in nature. They are high up in the food chain and integrate changes occurring at lower levels. As such, birds are excellent indicators of deteriorating habitat quality and environmental pollution. However, it is often easier to hear birds than see them. With proper sound detection and classification, researchers could automatically intuit factors about an area’s quality of life based on a changing bird population.

The aim of the competition is to identify birds in audio recordings. The main challenges are the following ones : 
- The training data is weakly labeled, i.e. we know which birds there are in the recordings, but we don't know when
- The test data is much more noisier that the training data

There are three sites in the test data, two of them requires models to identify which birds out of the 264 species are present in every 5 seconds interval, whereas the third one only requires weak labelling. The metric used to assess performances is the [F1-score](https://en.wikipedia.org/wiki/F1_score). 


## Solution Overview

Our solution has three main aspects : data augmentation, modeling and post-processing

### Data Augmentation

Data augmentation is the key to reduce the discrepancy between train and test. We start by randomly cropping 5 seconds of the audio and then add aggressive noise augmentations :
- Gaussian noise

With a soud to noise ratio up to 0.5

- Background noise

We randomly chose 5 seconds of a sample in the background dataset available [here](https://www.kaggle.com/theoviel/bird-backgrounds). This dataset contains samples without bircall from the example test audios from the competition data, and some samples from the freesound bird detection challenge that were manually selected.

- Modified Mixup

Mixup creates a combination of a batch `x1` and its shuffled version `x2` : `x = a * x1 + (1 - a) * x2` where `a` is samples with a beta distribution. 
Then, instead of using the classical objective for mixup, we define the target associated to `x` as the union of the original targets. 
This forces the model to correctly predict both labels.
Mixup is applied with probability 0.5 and I used 5 as parameter for the beta disctribution, which forces `a` to be close to 0.5.

- Improved cropping 

Instead of randomly selecting the crops, selecting them based on out-of-fold confidence was also used. The confidence at time `t` is the probability of the ground truth class predicted on the 5 second crop starting from `t`.

### Modeling

We used 4 models in the final blend :

- resnext50 [0.606 LB] - trained with the additional audio recordings.
- resnext101 [0.606 LB] - trained with the additional audio recordings as well.
- resnest50 [0.612 LB] 
- resnest50 [0.617 LB] - trained with improved crops


They were trained for 40 epochs (30 if the external data is used), with a linear scheduler with 0.05 warmup proportion. Learning rate is 0.001 with a batch size of 64 for the small models, and both are divided by two for the resnext101 one, in order to fit in a single 2080Ti.

We had no reliable validation strategy, and used stratified 5 folds where the prediction is made on the 5 first second of the validation audios.

### Post-processing

We used 0.5 as our threshold `T`.

- First step is to zero the predictions lower than `T`
- Then, we aggregate the predictions
  - For the sites 1 and 2, the prediction of a given window is summed with those of the two neighbouring windows. 
  - For the site 3, we aggregate using the max
- The `n` most likely birds with probability higher than `T` are kept
  - `n = 3` for the sites 1 and 2
  - `n` is chose according to the audio length for the site 3.



## Data

- Competition data is available on the [competition page](https://www.kaggle.com/c/birdsong-recognition/data)

Audio samples aren't actually used and the `csv`files are already in the `input` folder.

- Resampled data in `.wav` resampled at 32 kHz format is available in the following Kaggle datasets : 
  - [[Part 1](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-00)], 
 [[Part 2](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-01)], 
 [[Part 3](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-02)], 
 [[Part 4](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-03)], 
 [[Part 5](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-04)]

- Extra samples in `.wav` resampled at 32 kHz are available in the following Kaggle datasets :
  - [[Part 1](https://www.kaggle.com/ludovick/xenoexternalwav0)],
[[Part 2](https://www.kaggle.com/ludovick/xenoexternalwav1)]

- Files used for the background augmentations are available [on Kaggle](https://www.kaggle.com/theoviel/bird-backgrounds) as well.

The folder structure for the data is the following, where the folders `AUDIO_PATH`, `EXTRA_AUDIO_PATH`,  and `BACKGROUND_PATH` are specified in `params.py`

```
AUDIO_PATH
├── bird_class_1
│   ├── id1.wav
│   └── ...
├── bird_class_2
└── ...
EXTRA_AUDIO_PATH
├── bird_class_1
│   ├── extra_id1.wav
│   └── ...
├── bird_class_2
│   ├── extra_id2.wav
│   └── ...
└── ...
BACKGROUND_PATH
├── background_1.wav
└── ...
```

## Repository structure

- `input` : Input metadata
- `kept_logs`: Training logs of the 4 models used in the ensemble. Associated configs are in `configs.py`
- `notebooks` : Notebook to compute confidence for improved sampling
- `output` : More logs and outputs of the training
- `src` : Source code

## Inference

- To reproduce our final score, fork this notebook [notebook](https://www.kaggle.com/theoviel/inference-theo) in the kaggle kernels.
- Model weights are available on kaggle : [[Part 1]](https://www.kaggle.com/theoviel/birds-cp-1) , [[Part 2]](https://www.kaggle.com/theoviel/birds-cp-2), [[Part 3]](https://www.kaggle.com/theoviel/birds-checkpoints-3), 
- Weights used in the final ensemble are the following, where `IDX` is the fold number and varies from 0 to 4 :
  - `resnext50_32x4d_extra_IDX.pt`
  - `resnext101_32x8d_wsl_extra_IDX.pt`
  - `resnest50_fast_1s1x64d_mixup5_IDX.pt`
  - `resnest50_fast_1s1x64d_conf_IDX.pt`
