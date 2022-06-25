import pandas as pd
from ensemble import predict, ensemble_5fold, make_pseudo_df
from dataloader import *
from network import *

img_size = 288

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sub = pd.read_csv('../data/sample_submission.csv')

df_test = pd.read_csv('../data/test.csv')
df_test['img_path'] = df_test['img_path'].apply(lambda x:x.replace('test_imgs', 'test_1024'))
test_transform = get_train_augmentation(img_size=img_size, ver=1)
test_dataset = Test_dataset(df_test, test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

models_path = ['./results/000', './results/001', './results/002', './results/003', './results/004']
ensemble = ensemble_5fold(models_path, test_loader, device)
sub.iloc[:, 1] = ensemble.argmax(axis=1)
sub.to_csv(f'./submission.csv', index=False)