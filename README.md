# Kaggle Birdcall Identification

Code for the [Cornell Birdcall Identification competition](https://www.kaggle.com/c/birdsong-recognition)



## Data

- Competition data is available on the [competition page](https://www.kaggle.com/c/birdsong-recognition/data)

Audio samples aren't actually used but the metadata are already in the `input` folder.

- Resampled data in `.wav` resampled at 32 kHz format is available in the following Kaggle datasets : 
  - [[Part 1](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-00)], 
 [[Part 2](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-01)], 
 [[Part 3](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-02)], 
 [[Part 4](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-03)], 
 [[Part 5](https://www.kaggle.com/ttahara/birdsong-resampled-train-audio-04)]
  
There is one folder per class, they are expected to be put together in the `AUDIO_PATH` folder specified in `params.py`.

- Extra samples in `.wav` resampled at 32 kHz are available in the following Kaggle datasets :
  - [[Part 1](https://www.kaggle.com/ludovick/xenoexternalwav0)],
[[Part 2](https://www.kaggle.com/ludovick/xenoexternalwav1)]
  
There is one folder per class, they are expected to be put together in the `EXTRA_AUDIO_PATH` folder specified in `params.py`

- Files used for the background augmentations are available [on Kaggle](https://www.kaggle.com/theoviel/bird-backgrounds) as well.

They are expected to be put in the `BACKGROUND_PATH` folder specified in `params.py`.


## Repository structure


```
├── input
│   ├── 
│   ├── 
│   └── 
├── output
└── src

```
