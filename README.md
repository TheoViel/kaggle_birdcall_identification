# Kaggle Birdcall Identification

Code for the [Cornell Birdcall Identification competition](https://www.kaggle.com/c/birdsong-recognition)

## Context

> Do you hear the birds chirping outside your window? Over 10,000 bird species occur in the world, and they can be found in nearly every environment, from untouched rainforests to suburbs and even cities. Birds play an essential role in nature. They are high up in the food chain and integrate changes occurring at lower levels. As such, birds are excellent indicators of deteriorating habitat quality and environmental pollution. However, it is often easier to hear birds than see them. With proper sound detection and classification, researchers could automatically intuit factors about an area’s quality of life based on a changing bird population.

The aim of the competition is to identify birds in audio recordings. The main challenges are the following ones : 
- The training data is weakly labeled, i.e. we know which birds there are in the recordings, but we don't know when
- The test data is much more noisier that the training data

There are three sites in the test data, two of them requires models to identify which birds out of the 264 species are present in every 5 seconds interval, whereas the third one only requires weak labelling. The metric used to assess performances is the [F1-score](https://en.wikipedia.org/wiki/F1_score). 


## Data

- Competition data is available on the [competition page](https://www.kaggle.com/c/birdsong-recognition/data)

Audio samples aren't actually used but the metadata are already in the `input` folder.

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
