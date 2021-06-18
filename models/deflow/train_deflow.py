
import os
import sys
sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl

from torch import Tensor
from argparse import ArgumentParser

from dataset.dmrdenoise import DMRDenoiseDataModule
from models.deflow.deflow import DenoiseFlow
from metric.loss import EarthMoverDistance as EMD

from modules.utils.callback import TimeTrainingCallback
from modules.utils.lightning import LightningModule
from modules.utils.modules import print_progress_log


# -----------------------------------------------------------------------------------------
class TrainerModule(LightningModule):

    def __init__(self, cfg):
        super(TrainerModule, self).__init__()

        self.network = DenoiseFlow()
        self.metric = EMD()

        self.epoch = 1
        self.cfg = cfg
    
    def forward(self, p: Tensor, **kwargs):
        return self.network(p, **kwargs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.sched_patience, factor=self.cfg.sched_factor, min_lr=self.cfg.min_lr)
        return { "optimizer": optimizer, 'scheduler': scheduler }
    
    def training_step(self, batch, batch_idx):

        noise, noiseless = batch['pos'], batch['clean']
        denoised, logpx = self(noise)

        emd = self.metric(denoised, noiseless)
        loss = logpx * 1e-4 + emd * 1e-3

        self.log('EMD', emd.item() * 1e-3, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('logpx', logpx * 1e-4, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss
    
    def validation_step(self, batch, batch_idx):

        output = {}

        for key in self.trainer.datamodule.val_noisy_item_keys:
            noise, noiseless = batch[key], batch['clean']
            denoised, _ = self(noise)
            output[key] = self.metric(denoised, noiseless)
        
        return output

    def validation_epoch_end(self, batch):
        log_dict = {}
        keys = batch[0].keys()

        for key in keys:
            log_dict[key] = torch.tensor([x[key] for x in batch]).sum().detach().cpu()

        print_progress_log(self.epoch, log_dict)
        self.epoch += 1


# -----------------------------------------------------------------------------------------
def model_specific_args():
    parser = ArgumentParser()

    # Network
    parser.add_argument('--net', type=str, default='DenoiseFlow')
    # Optimizer and scheduler
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--sched_patience', default=10, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    # Training
    parser.add_argument('--max_epoch', default=100, type=int)

    return parser

def dataset_specific_args():
    parser = ArgumentParser()

    parser.add_argument('--noise_low', default=0.02, type=float)
    parser.add_argument('--noise_high', default=0.06, type=float, help='-1 for fixed noise level')
    parser.add_argument('--aug_scale', action='store_true', help='Enable scaling augmentation.')
    parser.add_argument('--datasets', type=list, default=[
        'data/DMRDenoise/dataset_train/patches_10k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_20k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_30k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_50k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_80k_1024.h5',
    ])
    parser.add_argument('--subset_size', default=7000, type=int, help='-1 for unlimited')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path=None, begin_checkpoint=None):

    comment = 'Baseline'
    cfg = model_specific_args().parse_args()

    dataset_cfg = dataset_specific_args().parse_args()
    datamodule = DMRDenoiseDataModule(dataset_cfg)

    trainer_config = {
        'default_root_dir'     : './runs/',
        'gpus'                 : 1,  # Set this to None for CPU training
        'fast_dev_run'         : False,
        'max_epochs'           : cfg.max_epoch,
        'precision'            : 32,   # 16
        # 'amp_level'            : 'O1',
        'weights_summary'      : 'top',  # 'top', 'full' or None
        # 'gradient_clip_val'    : 1e-3,
        'deterministic'        : False,
        'num_sanity_val_steps' : 0,  # -1 or 0
        'checkpoint_callback'  : False,
        'callbacks'            : [TimeTrainingCallback()],
    }

    module = TrainerModule(cfg)
    trainer = pl.Trainer(**trainer_config)
    trainer.is_interrupted = False


    if phase == 'Train':
        if comment is not None:
            print(f'\nComment: \033[1m{comment}\033[0m')
        if begin_checkpoint is not None:
            module.network = torch.load(begin_checkpoint, map_location='cpu')

        trainer.fit(model=module, datamodule=datamodule)

        if checkpoint_path is not None and trainer_config['fast_dev_run'] is False and trainer.is_interrupted is False:
            save_path = checkpoint_path + f'-epoch{trainer_config["max_epochs"]}.ckpt'
            torch.save(module.network, save_path)
            print(f'Model has been save to \033[1m{save_path}\033[0m')
    else:  # Test
        module.network = torch.load(checkpoint_path, map_location='cpu')
        trainer.test(model=module, datamodule=datamodule)



# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    checkpoint_path = 'runs/ckpt/DenoiseFlow-baseline'

    # train('Train', None, None)                      # Train from begining, and save nothing after finish
    train('Train', checkpoint_path, None)           # Train from begining, save network params after finish
    # train('Train', checkpoint_path, previous_path)  # Train from previous checkpoint, save network params after finish
    # train('Test', checkpoint_path, None)            # Test with given checkpoint
