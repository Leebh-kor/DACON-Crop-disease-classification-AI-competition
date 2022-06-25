# 작물 병해 분류 AI competition -1st-place-solution
This repository is the 1st place solution for [DACON 작물 병해 분류 AI 경진대회](https://dacon.io/competitions/official/235842/overview/description).

***
## Overview
- RegNet040 5Fold ensemble
- Pseudo labeling
- 이전 대회에서 활용한 Framework을 기반으로 진행했습니다.

## Total Process
- Train Dastset이 많지 않기 때문에 Test Dataset에 대해 Pseudo labeling을 진행 후, Train Dataset에 추가하여 모델을 Training시켰습니다.
- 총 6번의 Pseudo labeling을 진행 하였습니다.
- 첫 Pseudo labeling은 최적화 과정을 거친 regnet 5fold ensemble(Public Score : 0.989)을 이용했습니다.
- 이후 동일한 Model config에서 총 5번의 Pseudo label set update를 진행하였습니다.
- 최종적으로 총 5번의 update가 된 Pseudo label set + Train Dataset에 대해 Training한 regnet 5fold ensemble을 제출 하였고 Private Score기준 1등을 달성했습니다.

## Pseudo label set sampling
- Pseudo label set의 신뢰성을 위해, 5fold ensembel logic에 대한 softmax결과값이 0.9보다 큰 sample로 한정했습니다.
- Pseudo label set에서 비율이 많은 0번 class는 random sampling(n=500)했습니다.

## Pseudo label set업데이트에 따른 Regnet 5fold ensemble Public Score
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

## Directory Structure
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
- src
    - config.py
    - dataloader.py
    - ...
    - ...
    - main.py

<br>

## Data Preparation
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

## Make a prediction without Training
- Inference using saved weights
- Data Preparation must proceed first.
```python
python test.py
```


