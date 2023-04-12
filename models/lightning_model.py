import os
import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

import torch.nn as nn

from numpy import sqrt, argmax
from torch.optim import lr_scheduler
from .model import CNN
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from matplotlib import pyplot
from backbone.vit_pytorch import cait, vit, deepvit
from backbone.torchvision.models_orig import resnet, densenet, inception
from factory.config import *



class Model(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()

        self.model_name = model_name
     

        # efficientnet-b0 ~ efficientnet-b5
        if model_name == 'efficientnet-b0':
            self.net = CNN(backbone="efficientnet-b0", freeze=False)
        if model_name == 'efficientnet-b1':
            self.net = CNN(backbone="efficientnet-b1", freeze=False)
        if model_name == 'efficientnet-b2':
            self.net = CNN(backbone="efficientnet-b1", freeze=False)
        if model_name == 'efficientnet-b3':
            self.net = CNN(backbone="efficientnet-b2", freeze=False)
        if model_name == 'efficientnet-b4':
            self.net = CNN(backbone="efficientnet-b4", freeze=False)
        if model_name == 'efficientnet-b5':
            self.net = CNN(backbone="efficientnet-b5", freeze=False)

        #naive vit
        elif model_name == 'vit':
            self.net = vit.ViT(image_size=IMG_SIZE , patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)

        #Cait
        elif model_name == 'cait':
            self.net = cait.CaiT(image_size=IMG_SIZE, patch_size=32, num_classes=2, dim=1024, depth=12, cls_depth=2, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, layer_dropout=0.05)

        #deep vit
        elif model_name == 'deepvit':
            self.net = deepvit.DeepViT(image_size=IMG_SIZE, patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
      
        #resnet50
        elif model_name == 'resnet50':
                self.net = resnet.resnet50(pretrained=True)
    
        #resnet101
        elif model_name == 'resnet101':
            self.net = resnet.resnet101(pretrained=True)
    
        #resnet152
        elif model_name == 'resnet152':
            self.net = resnet.resnet152(pretrained=True)
    
        #densenet121
        elif model_name == 'densenet121':
            self.net = densenet.densenet121(pretrained=True)
    
        #densenet161
        elif model_name == 'densenet161':
            self.net = densenet.densenet161(pretrained=True)
    
        #densenet169
        elif model_name == 'densenet169':
            self.net = densenet.densenet169(pretrained=True)
    
        #densenet201
        elif model_name == 'densenet201':
            self.net = densenet.densenet201(pretrained=True)

        #inception_v3
        elif model_name == 'inception_v3':
            self.net = inception.inception_v3(pretrained=True)

        hidden_dim1 = 256
        hidden_dim2 = 64
        num_classes = 2
        dropout = 0.1        

        self.classifier = nn.Sequential(
            nn.Linear(1000, hidden_dim1),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes)
        )

        self.train_preds = []
        self.train_gts = []

        self.valid_preds = []
        self.valid_gts = []

        self.test_preds = []
        self.test_probs = []
        self.test_gts = []

    def forward(self, x):
        if 'efficientnet' in self.model_name:
            return self.net(x)
        elif 'inception' in self.model_name:
            x = self.net(x)
            return self.classifier(x.logits)
        else:
            x = self.net(x)
            return self.classifier(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        for gy in y:
            self.train_gts.append(gy.cpu().item())
        for py in y_hat:
            c = torch.argmax(py)
            self.train_preds.append(c.cpu().item())

        self.log("loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        acc, sen, spe, ppv, npv, tn, fp, fn, tp = self.calculate_metrics(
                                                      self.train_gts, self.train_preds
                                                  )
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log("train_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        self.log("train_sensitivity(recall)", sen, on_epoch=True, prog_bar=True)
        self.log("train_specificity", spe, on_epoch=True, prog_bar=True)
        self.log("train_ppv(precision)", ppv, on_epoch=True, prog_bar=True)
        self.log("train_npv", npv, on_epoch=True, prog_bar=True)
        self.log("train_tn", tn , on_epoch=True, prog_bar=True)
        self.log("train_fp", fp, on_epoch=True, prog_bar=True)
        self.log("train_fn", fn, on_epoch=True, prog_bar=True)
        self.log("train_tp", tp, on_epoch=True, prog_bar=True)

        self.train_preds = []
        self.train_gts = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        for gy in y:
            self.valid_gts.append(gy.cpu().item())
        for py in y_hat:
            c = torch.argmax(py)
            self.valid_preds.append(c.cpu().item())

        acc, sen, spe, ppv, npv, tn, fp, fn, tp = self.calculate_metrics(
            self.valid_gts, self.valid_preds
        )

        self.log("val_bat_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_sensitivity(recall)", sen, on_epoch=True, prog_bar=True)
        self.log("val_specificity", spe, on_epoch=True, prog_bar=True)
        self.log("val_ppv(precision)", ppv, on_epoch=True, prog_bar=True)
        self.log("val_npv", npv, on_epoch=True, prog_bar=True)
        self.log("val_tn", tn , on_epoch=True, prog_bar=True)
        self.log("val_fp", fp, on_epoch=True, prog_bar=True)
        self.log("val_fn", fn, on_epoch=True, prog_bar=True)
        self.log("val_tp", tp, on_epoch=True, prog_bar=True)
           
        return {
            "val_bat_loss": loss, "val_acc": acc,
            "val_sensitivity(recall)": sen, "val_specificity": spe,
            "val_ppv(precision)":ppv, "val_npv": npv,
            "val_tn": tn, "val_fp": fp, "val_fn": fn, "val_tp": tp,
        }

    def validation_epoch_end(self, outputs):
        acc, sen, spe, ppv, npv, tn, fp, fn, tp = self.calculate_metrics(
                                                      self.valid_gts, self.valid_preds
                                                  )
        avg_loss = torch.stack([x['val_bat_loss'] for x in outputs]).mean()

        self.log("val_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_sensitivity(recall)", sen, on_epoch=True, prog_bar=True)
        self.log("val_specificity", spe, on_epoch=True, prog_bar=True)
        self.log("val_ppv(precision)", ppv, on_epoch=True, prog_bar=True)
        self.log("val_npv", npv, on_epoch=True, prog_bar=True)
        self.log("val_tn", tn , on_epoch=True, prog_bar=True)
        self.log("val_fp", fp, on_epoch=True, prog_bar=True)
        self.log("val_fn", fn, on_epoch=True, prog_bar=True)
        self.log("val_tp", tp, on_epoch=True, prog_bar=True)

        self.valid_preds = []
        self.valid_gts = []


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        for gy in y:
            self.test_gts.append(gy.cpu().item())
        for py in y_hat:
            c = torch.argmax(py)
            p = F.softmax(py, dim=0)[1]
            self.test_probs.append(p.cpu().item())
            self.test_preds.append(c.cpu().item())
        
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        acc, sen, spe, ppv, npv, tn, fp, fn, tp = self.calculate_metrics(
            self.test_gts, self.test_preds
        )
        auc = self.calculate_auc(self.test_gts, self.test_probs)
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log("test_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_sensitivity(recall)", sen, on_epoch=True, prog_bar=True)
        self.log("test_specificity", spe, on_epoch=True, prog_bar=True)
        self.log("test_ppv(precision)", ppv, on_epoch=True, prog_bar=True)
        self.log("test_npv", npv, on_epoch=True, prog_bar=True)
        self.log("test_auc", auc, on_epoch=True, prog_bar=True)
        self.log("test_tn", tn , on_epoch=True, prog_bar=True)
        self.log("test_fp", fp, on_epoch=True, prog_bar=True)
        self.log("test_fn", fn, on_epoch=True, prog_bar=True)
        self.log("test_tp", tp, on_epoch=True, prog_bar=True)
        
        print('============' * 5)
        print('Accuracy : {:.4f}, Recall(Sensitivity) : {:.4f}, Specificity :{:.4f}, PPV(Precision) : {:.4f}, NPV : {:.4f}, Auc : {:.4f}, Confusion : ( TP-{} | FP-{} | FN-{} | TN-{} )'.format(acc, sen, spe, ppv, npv, auc, tp, fp, fn, tn))
        print('============' * 5)

       
        dfGTs = pd.DataFrame(np.round_(np.array(self.test_gts)))
        dfPreds = pd.DataFrame(np.round_(np.array(self.test_preds)))
        dfProbs = pd.DataFrame(np.round_(np.array(self.test_probs) * 100, 3))

        pd.concat([dfGTs, dfPreds, dfProbs], axis=1).to_csv('./test.csv', index=False)

        
    def calculate_metrics(self, gts, preds):
        tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[0,1]).ravel()

        if math.isnan(tn): tn = 0
        if math.isnan(fp): fp = 0
        if math.isnan(fn): fn = 0
        if math.isnan(tp): tp = 0

        acc = (tp + tn) / (tn + fp + fn + tp)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)

        if math.isnan(acc): acc = 0
        if math.isnan(sen): sen = 0
        if math.isnan(spe): spe = 0
        if math.isnan(ppv): ppv = 0
        if math.isnan(npv): npv = 0

        return np.float32(acc), np.float32(sen), np.float32(spe), np.float32(ppv), np.float32(npv), tn, fp, fn, tp

    def calculate_auc(self, gts, probs):
        try:
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

            pyplot.savefig('test_roc.png', dpi=600)
        except ValueError:
            auc=0
        return auc
