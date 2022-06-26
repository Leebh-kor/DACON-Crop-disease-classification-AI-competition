# Crop disease classification AI competition -1st-place-solution
This repository is the 1st place solution for [DACON 작물 병해 분류 AI 경진대회](https://dacon.io/competitions/official/235842/overview/description). This competition is a task to classify the disease through the images of the crop affected by the disease.
DACON: Crop disease classification AI competition. 1st place solution
***
## Overview
- Data Augmentation
- RegNet040 5Fold ensemble
- Pseudo labeling

## Pseudo labeling Process
- The training dataset are a small amount of 250 images. So I conducted Pseudo labeling for the relatively large amount Test dataset(4750 images).
- Pseudo labeling was performed 6 times in total process.
- The first Pseudo labeling used the optimized regnet 5fold ensemble(Public Score: 0.989).
- After that, I did a total of 5 Pseudo label set updates in the same model config.
- Finally, I submitted a regnet 5 fold ensemble that was trained for Pseudo label set + Train Dataset, which was updated 5 times in total, and achieved 1st place based on Private Score.

## Pseudo label set sampling
- For the reliability of the pseudo label set, the softmax value for the 5fold ensemble model was limited to samples greater than 0.9.
- Class 0 with high proportions in the pseudo label set was random sampling(n=500).

## Public Score by Pseudo label set update count
- 0step(Only Trainset) : 0.989
- 1step : 0.996
- 4step : 0.998
- 6step(Submission) : 0.999 / Private : 0.99885

## Requirements
```python
pip install -r requirements.txt
```
- Ubuntu 18.04, Cuda 11.1
- opencv-python  
- numpy  
- pandas
- timm
- torch==1.8.0 torchvision 0.9.0 with cuda 11.1
- natsort
- scikit-learn
- pillow
- torch_optimizer
- tqdm
- ptflops
- easydict
- matplotlib

<br>

## Data Structure
- data
    - train.csv
    - test.csv
    - sample_submission.csv
    - train_imgs
        - 10000.jpg
        - ...
        - 10249.jpg
    - test_imgs
        - 20000.jpg
        - ...
        - 24749.jpg

<br>

## Data Preprocessing
- Since the original image size is large, we saved it after resize for learning time
```python
python make_data.py
```

<br>

## Train & inference with bestsetting
- Training and inference are conducted at the same time.
```python
python main.py --batch_size=16 --drop_path_rate=0.2 --encoder_name=regnety_040\
			   --img_size=288 --aug_ver=2 --scheduler=cycle\
               --weight_decay=1e-3 --initial_lr=5e-6\
               --max_lr=1e-3 --epochs=70 --warm_epoch=5 --image_type=train_1024
```
