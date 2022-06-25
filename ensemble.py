from os.path import join as opj
from tqdm import tqdm
from dataloader import *
from network import *


def predict(encoder_name, test_loader, device, model_path):
    model = Network_test(encoder_name).to(device)
    model.load_state_dict(torch.load(opj(model_path, 'best_model.pth'))['state_dict'])
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = torch.as_tensor(images, device=device, dtype=torch.float32)
            preds = model(images)
            preds = torch.softmax(preds, dim=1)
            preds_list.extend(preds.cpu().tolist())

    return np.array(preds_list)

def ensemble_5fold(model_path_list, test_loader, device):
    predict_list = []
    for model_path in model_path_list:
        prediction = predict(encoder_name= 'regnety_040', test_loader = test_loader, device = device, model_path = model_path)
        predict_list.append(prediction)
    ensemble = (predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4])/len(predict_list)

    return ensemble

def make_pseudo_df(train_df, test_df, ensemble, step, threshold = 0.9, z_sample = 500):
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()

    test_df_copy['disease'] = np.nan
    test_df_copy['disease_code'] = ensemble.argmax(axis=1)
    pseudo_test_df = test_df_copy.iloc[np.where(ensemble > threshold)[0]].reset_index(drop=True)
    z_idx  = pseudo_test_df[pseudo_test_df['disease_code'] == 0].sample(n=z_sample, random_state=42).index.tolist()
    ot_idx = pseudo_test_df[pseudo_test_df['disease_code'].isin([*range(1,8)])].index.tolist()
    pseudo_test_df = pseudo_test_df.iloc[z_idx + ot_idx]

    train_df_copy = train_df_copy.append(pseudo_test_df, ignore_index=True).reset_index(drop=True)
    print(f'Make train_{step}step.csv')
    train_df_copy.to_csv(f'../data/train_{step}step.csv', index=False)