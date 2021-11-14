# 작물 병해 분류 AI competition -1st-place-solution
This repository is the 1st place solution for [DACON 작물 병해 분류 AI 경진대회](https://dacon.io/competitions/official/235842/overview/description).


<br>

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


