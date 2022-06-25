import os
import cv2
import pandas as pd
from os.path import join as opj

df = pd.read_csv('../data/train.csv')

# Make Train
save_path = '../data/train_1024'
os.makedirs(save_path, exist_ok=True)
for img in df['img_path']:
    name = os.path.basename(img)
    img = cv2.imread(opj('../data/', img))
    img = cv2.resize(img, dsize=(1024, 1024))
    img = cv2.imwrite(opj(save_path, name), img)

# Make Test
df = pd.read_csv('../data/test.csv')
save_path = '../data/test_1024'
os.makedirs(save_path, exist_ok=True)
for img in df['img_path']:
    name = os.path.basename(img)
    img = cv2.imread(opj('../data/', img))
    img = cv2.resize(img, dsize=(1024, 1024))
    img = cv2.imwrite(opj(save_path, name), img)
