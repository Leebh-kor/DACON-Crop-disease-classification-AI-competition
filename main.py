import os
import pprint
import random
import warnings
import torch
import numpy as np
from config import getConfig
from trainer import Trainer
import pandas as pd
from dataloader import *
from network import *
from ensemble import predict, ensemble_5fold, make_pseudo_df
warnings.filterwarnings('ignore')
args = getConfig()

def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.model_path, (args.exp_num).zfill(3))
    
    # Create model directory
    os.makedirs(save_path, exist_ok=True)
    Trainer(args, save_path)

    return save_path

if __name__ == '__main__':
    img_size = 288
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sub = pd.read_csv('../data/sample_submission.csv')
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    df_test['img_path'] = df_test['img_path'].apply(lambda x:x.replace('test_imgs', 'test_1024'))
    test_transform = get_train_augmentation(img_size=img_size, ver=1)
    test_dataset = Test_dataset(df_test, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    start = 0 # first time : Only Trainset
    steps = 6 # Number of pseudo labeling times 
    for step in range(start, steps+1): 
        models_path = []
        args.step = step
        for s_fold in range(5): # 5fold
            args.fold = s_fold
            args.exp_num = str(s_fold)
            save_path = main(args)
            models_path.append(save_path)
        ensemble = ensemble_5fold(models_path, test_loader, device)
        make_pseudo_df(df_train, df_test, ensemble, step+1)

    # For submission
    # sub.iloc[:, 1] = ensemble.argmax(axis=1)
    # sub.to_csv(f'./submission.csv', index=False)


