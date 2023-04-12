import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models.lightning_model import Model
from models.loader import *

import argparse

"""
    테스트 코드
"""
    
if __name__ == "__main__":
    #데이터 위치
    folder = '0'
    data_path ='./data/marginal/{}'.format(folder)
	#모델 루트
    model_root = './results/marginal/{}'.format(folder)
    #모델 위치
    model_path = '{}/{}'.format(model_root, 'model/marginal-epoch=2-val_acc=0.9436089992523193-val_avg_loss=0.2080.ckpt')
    #테스트 결과 pkl 파일 저장할 위치
    save_path = '{}/{}'.format(model_root, 'result')


    test_info = data_load(path=os.path.join(data_path, "test"))

    test = custom_dataset(
        test_info,
        transforms=transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    test_loader = DataLoader(test, num_workers=24, batch_size=8)
    model = Model(save_path = save_path)
    model = model.load_from_checkpoint(model_path,
                                      save_path=save_path)
    trainer = pl.Trainer(gpus=1)
    trainer.test(model,test_loader)
