import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models.lightning_model import Model
from models.loader import *
from factory.config import *

import argparse

"""
    학습 코드
"""

def train(config):
    if config.model_num == 16: #inception_v3
        IMG_SIZE = 299

    save_path = '{}/{}'.format(config.save_path, config.folder)

    if not os.path.exists(save_path+'/model'):
        os.makedirs(save_path+'/model')
    
    train_info = data_load(
                       path='{}/{}/{}'.format(config.data_path, config.folder, 'train'),
                       under_sampling=False,
                      )
    val_info = data_load(
                     path='{}/{}/{}'.format(config.data_path, config.folder, 'val'),
                     under_sampling=False,
                     )

    train = custom_dataset(
        train_info,
        transforms=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ColorJitter(brightness=(0.2, 2),
                                      contrast=(0.3, 2), 
                                      saturation=(0.2, 2), 
                                      hue=(-0.3, 0.3)),
                transforms.ToTensor(),
            ]
        ),
    )
    val = custom_dataset(
        val_info,
        transforms=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ]
        ),
    )

    train_dataloader = DataLoader(
                          train,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True
                         )
    val_loader = DataLoader(
                        val,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size
                       )

    model = Model(model_name=model_name_list[config.model_num])

    checkpoint_callback = ModelCheckpoint(
                                      dirpath=save_path + '/model', 
                                      filename='cp-{epoch}-{val_acc:.4f}-{val_f1:.4f}-{val_sensitivity(recall):.4f}-{val_specificity:.4f}-{val_avg_loss:.4f}',
                                      save_top_k=5, 
                                      verbose=True, 
                                      monitor="val_acc", mode="max"
                                     )

    early_stop_callback = EarlyStopping(
                                    monitor='val_acc',
                                    patience=20,
                                    mode='max'
                                   )

    trainer = pl.Trainer(
                         gpus=list(map(int,config.gpu_num.split(','))),
                         #accelerator='ddp',
                         strategy='dp',
                         max_epochs=config.epochs,
                         callbacks =[checkpoint_callback, early_stop_callback],
                         num_sanity_val_steps=0,
                         auto_lr_find=True, # find learning rate automatically
                        )

    ## Tune Learning rate
    lr_finder = trainer.tuner.lr_find(
                                  model,
                                  train_dataloader,
                                  val_loader,
                                  #num_training=50
                                 )

    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_{}.png'.format(model_name_list[config.model_num]))

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr

    ## Training
    # Fit
    trainer.fit(
                model,
                train_dataloader,
                val_loader
               )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=int, default=0, help='(default=0)')
    parser.add_argument('--data_path', type=str, default='./data', help='(default=./data)')
    parser.add_argument('--save_path', type=str, default='./results', help='(default=./result)')

    parser.add_argument('--batch_size', type=int, default=16, help='(default=16)')
    parser.add_argument('--num_workers', type=int, default=8, help='(default=8)')
    parser.add_argument('--epochs', type=int, default=150, help='(default=150)')

    parser.add_argument('--gpu_num', type=str, default='0,1', help='(default="0,1")')
    parser.add_argument('--model_num', type=int, default=0, help='0: efficinetnet-b0, 1: efficinetnet-b1, 2: efficinetnet-b2, 3: efficinetnet-b3, 4: efficinetnet-b4, 5: efficinetnet-b5, 6: vit, 7: cait, 8: deepvit, 9: resnet50, 10: resnet101, 11: resnet152, 12: densenet121, 13: densenet161, 14: densenet169, 15: densenet201, 16: inception_v3 (default=0, efficinetnet-b0)'
    )

    config = parser.parse_args()

    print("### Parameters ###")
    print(config)

    train(config)
