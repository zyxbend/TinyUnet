# TinyUnet
The code repository of ChinaMM2021 paper TinyUnet:结合多层特征及空间信息蒸馏的医学影像分割


# Structure of this repository
This repository is organized as:

- [dataloaders](/datasets/) contains the dataloader for different datasets
- [modeling](/modeling/) contains a model zoo for network models
- [models](/models/) contains pretrained models for network models
- [scripts](/scripts/) coontains scripts for preparing data
- [utils](/utils/) contains api for training and processing data
- [test_em.py](/test_em.py) test student/teacher model on  the em datasets
- [test_nih.py](/test_nih.py) test student/teacher model on  the nih datasets
- [test_tooth.py](/test_tooth.py) this dataset is private,so can't offer the related code
# Usage Guide

## Requirements

 All the codes are tested in the following environment:

- pytorch 1.6.0
- OpenCV
- natsort
- tqdm

## Dataset Preparation

### EM datasets
The ISBI challenge for segmentation of neuronal structures in Electron Microscopic (EM)
Download data [here](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000502)

Please follow the instructions and the data/ directory should then be structured as follows
```
em_challenge
├── data
|   ├── 0.PNG
|   └── 1.PNG
...
├── mask
|   ├── 0.PNG
|   └── 1.PNG
...
├── val_img
|   ├── 78.PNG
|   └── 79.PNG
...
├── val_mask
|   ├── 78.PNG
|   └── 79.PNG
...
```

### NIH datasets
Similar to EM dataset.

## Running
### Test Teacher/Student Model
After knowledge distillation, a well-trained teacher model is required.

[UNet](https://github.com/nizhenliang/RAUNet) is chosen to be our teacher model.

```
python -u /path/test_Tooth.py -dataset em_dataset -eval_interval 1 -no_val False
```
If you want to try different datasets, use --dataset with following choices
```
'em_dataset','nih_dataset','tooth'
```

