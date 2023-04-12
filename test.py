import os
import pandas as pd
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from matplotlib import pyplot
from numpy import sqrt, argmax

import pytorch_lightning as pl

from models.lightning_model import Model
from models.loader_new import *
from factory.config import *

import argparse

"""
	Test Code
"""

def find_best_model(model_root):
    model_list = os.listdir(model_root)
    model_score = []

    for model in model_list:
        if 'ckpt' in model:
            scores = model.replace('.ckpt','').split('-')
            acc = float(scores[2].split('=')[1])
            sen = float(scores[4].split('=')[1])
            spe = float(scores[5].split('=')[1])
            los = float(scores[6].split('=')[1])
            score = ( acc * sen * spe ) / los
            model_score.append(score)
        else:
            model_score.append(-10000)
    max_score = max(model_score)

    return model_list[model_score.index(max_score)]

def metrics(gts, preds):
    tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[0,1]).ravel()
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average='binary', pos_label=1)

    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    return acc, sen, spe, ppv, npv, tn, fp, fn, tp, f1

def plot_roc_curve(gts, probs, model_name, folder):
    auc = roc_auc_score(gts, probs)

    ns_probs = [0 for _ in range(len(gts))]
    lr_probs = probs

    ns_auc = roc_auc_score(gts, ns_probs)
    lr_auc = roc_auc_score(gts, lr_probs)

    ns_fpr, ns_tpr, _ = roc_curve(gts, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(gts, lr_probs)

    # calculate g-mean for each threshold
    gmeans = sqrt(lr_tpr * (1-lr_fpr))
    ix = argmax(gmeans)

    # plot True, Predict, Best
    pyplot.scatter(lr_fpr[ix], lr_tpr[ix], marker='*', color='black', label='Best')
    pyplot.text(lr_fpr[ix] + 0.05, lr_tpr[ix] - 0.05, "FPR: {}\nTPR: {}".format(lr_fpr[ix], lr_tpr[ix]), fontsize=7)

    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='True')
    pyplot.plot(lr_fpr, lr_tpr, marker=',', label='Predict (auc={})'.format(round(auc, 3)))

    pyplot.xlabel('False Positive Rate (1 - Specificity)')
    pyplot.ylabel('True Positive Rate (Sensitivity)')

    pyplot.legend()

    pyplot.savefig('test_{}_f{}_roc.png'.format(model_name, folder), dpi=600)
    
    return auc

def test(config):
    random.seed(config.folder)

    # Patch Data Path
    data_path ='{}/{}'.format(config.data_path , config.folder)

    # Trained Model
    model_root = '{}/{}/model'.format(config.save_path, config.folder)

    # Model Path
    if config.model_name == None:
        model_path = '{}/{}'.format(model_root, config.model_name)
    else:
        model_path = '{}/{}'.format(model_root, find_best_model(model_root))
    print("Model_path : {}\n".format(model_path))

    test_info, file_name = data_load(path=os.path.join(data_path, "test"))

    test = custom_dataset(
        test_info,
        transforms=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ]
        ),
    )

    test_loader = DataLoader(
                         test,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size
                        )

    model = Model(model_name=model_name_list[config.model_num])

    model = model.load_from_checkpoint(model_name=model_name_list[config.model_num], checkpoint_path=model_path)

    trainer = pl.Trainer(gpus=[config.gpu_num])
    
    trainer.test(model, test_loader)

    with open('test_{}_f{}_metrics.csv'.format(model_name_list[config.model_num], config.folder), 'w') as fOut:
        fOut.write("CV_index,Accuracy,F1,AUC,Sensitivity,Specificity,PPV,NPV,TN,FP,FN,TP\n")

    dfData = pd.concat([pd.DataFrame(file_name), pd.read_csv("./test.csv".format(model_name_list[config.model_num]))], axis=1)
    dfData.columns = ['img_name', 'gts', 'preds', 'probs']
    dfData.to_csv('./test_{}_f{}_probs.csv'.format(model_name_list[config.model_num], config.folder), index=False)

    acc, sen, spe, ppv, npv, tn, fp, fn, tp, f1 = metrics(dfData['gts'], dfData['preds'])
    auc = plot_roc_curve(dfData['gts'], dfData['probs'], model_name_list[config.model_num], config.folder)

    with open('test_{}_f{}_metrics.csv'.format(model_name_list[config.model_num], config.folder), 'a') as fOut:
        fOut.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(acc, f1, auc, sen, spe, ppv, npv, tn, fp, fn, tp))
    os.remove('./test.csv'.format(model_name_list[config.model_num]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=int, default=0, help='(default=0)')
    parser.add_argument('--data_path', type=str, default='./data', help='(default=./data)')
    parser.add_argument('--save_path', type=str, default='./results', help='(default=./result)')

    parser.add_argument('--batch_size', type=int, default=1, help='(default=1)')
    parser.add_argument('--num_workers', type=int, default=8, help='(default=8)')

    parser.add_argument('--gpu_num', type=int, default=0, help='(default=0)')
    parser.add_argument('--model_num', type=int, default=0, help='0: efficinetnet-b0, 1: efficinetnet-b1, 2: efficinetnet-b2, 3: efficinetnet-b3, 4: efficinetnet-b4, 5: efficinetnet-b5, 6: vit, 7: cait, 8: deepvit, 9: resnet50, 10: resnet101, 11: resnet152, 12: densenet121, 13: densenet161, 14: densenet169, 15: densenet201, (default=0, efficinetnet-b0)')
    parser.add_argument('--model_name', type=str, default='None', help='(default=None)')
   
    config = parser.parse_args()
    print("### Parameters ###")
    print(config)

    test(config)
