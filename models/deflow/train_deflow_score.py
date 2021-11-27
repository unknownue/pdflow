
import os
import sys
sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl

from torch import Tensor
from argparse import ArgumentParser

from dataset.scoredenoise.dataset import ScoreDenoiseDataModule
from models.deflow.deflow import DenoiseFlow, Disentanglement, DenoiseFlowMLP
from models.deflow.denoise import patch_denoise
from metric.loss import MaskLoss, ConsistencyLoss
from metric.loss import EarthMoverDistance as EMD

from modules.utils.score_utils import chamfer_distance_unit_sphere
from modules.utils.callback import TimeTrainingCallback
from modules.utils.lightning import LightningProgressBar
from modules.utils.modules import print_progress_log


# -----------------------------------------------------------------------------------------
class TrainerModule(pl.LightningModule):

    def __init__(self, cfg):
        super(TrainerModule, self).__init__()

        self.disentangle_method = Disentanglement.LCC
        self.network = DenoiseFlow(self.disentangle_method)
        # self.network = DenoiseFlowMLP(self.disentangle_method)

        self.loss_emd = EMD()
        self.mloss = MaskLoss()
        self.closs = ConsistencyLoss()

        self.epoch = 0
        self.cfg = cfg

        self.min_CD = 5.0

    def forward(self, p: Tensor, **kwargs):
        return self.network(p, **kwargs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.sched_patience, factor=self.cfg.sched_factor, min_lr=self.cfg.min_lr)
        # return { 'optimizer': optimizer, 'lr_scheduler': { 'scheduler': scheduler, 'monitor': 'EMD' } }

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.sched_patience, factor=self.cfg.sched_factor, min_lr=self.cfg.min_lr)
        #return { "optimizer": optimizer, 'scheduler': scheduler }
        return optimizer

    def training_step(self, batch, batch_idx):

        pcl_noisy, pcl_clean = batch['pcl_noisy'], batch['pcl_clean']
        denoised, logpx, consistency = self(pcl_noisy, y=pcl_clean)

        emd = self.loss_emd(denoised, pcl_clean)
        loss = logpx * 1e-6 + emd * 0.1 + consistency * 10.0

        self.log('EMD', emd.detach().cpu().item() * 0.1, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('logpx', logpx * 1e-6, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('consistency', consistency * 10.0, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):

        pcl_noisy, pcl_clean = batch['pcl_noisy'], batch['pcl_clean']
        pcl_denoised = patch_denoise(self, pcl_noisy.squeeze(), patch_size=1024)  # Fix patch size
        
        return {
            'denoised': pcl_denoised,
            'clean'  : pcl_clean.squeeze(),
        }

    def validation_epoch_end(self, batch):
        all_denoise = torch.stack([x['denoised'] for x in batch])
        all_clean   = torch.stack([x['clean']    for x in batch])

        avg_chamfer = chamfer_distance_unit_sphere(all_denoise, all_clean, batch_reduction='mean')[0].item() * 1e4

        extra = []
        # if avg_chamfer < self.min_CD:
        #     self.min_CD = avg_chamfer
        #     save_path = f'runs/ckpt/DenoiseFlow-{self.disentangle_method.name}-scoreset-minCD.ckpt'
        #     torch.save(self.network.state_dict(), save_path)
        #     extra.append('CD')
    
        print_progress_log(self.epoch, { 'CD': avg_chamfer }, extra=extra)
        self.epoch += 1


# -----------------------------------------------------------------------------------------
def model_specific_args():
    parser = ArgumentParser()

    # Network
    parser.add_argument('--net', type=str, default='DenoiseFlow')
    # Optimizer and scheduler
    parser.add_argument('--learning_rate', default=2e-3, type=float)
    parser.add_argument('--sched_patience', default=10, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    # Training
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--seed', default=2021, type=int)

    return parser

def dataset_specific_args():
    parser = ArgumentParser()

    parser.add_argument('--noise_min', default=0.005, type=float)  # 0.005
    parser.add_argument('--noise_max', default=0.030, type=float)  # 0.020
    parser.add_argument('--val_noise', default=0.015, type=float)
    parser.add_argument('--aug_rotate', default=True, choices=[True, False])
    parser.add_argument('--dataset_root', default='./data/ScoreDenoise', type=str)
    parser.add_argument('--dataset', default='PUNet', type=str)
    parser.add_argument('--resolutions', default=['10000_poisson', '30000_poisson', '50000_poisson'], type=list)
    # parser.add_argument('--resolutions', default=['10000_poisson'], type=list)
    parser.add_argument('--patch_size', type=int, default=1024)
    parser.add_argument('--num_patches', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path=None, begin_checkpoint=None):

    comment = 'scoreset'
    cfg = model_specific_args().parse_args()
    pl.seed_everything(cfg.seed)

    dataset_cfg = dataset_specific_args().parse_args()
    datamodule = ScoreDenoiseDataModule(dataset_cfg)

    trainer_config = {
        'default_root_dir'     : './runs/',
        'gpus'                 : 1,  # Set this to None for CPU training
        'fast_dev_run'         : False,
        'max_epochs'           : 150, # cfg.max_epoch,
        'precision'            : 32,   # 32, 16, 'bf16'
        'gradient_clip_val'    : 1e-3,
        'deterministic'        : False,
        'num_sanity_val_steps' : 0, # -1,  # -1 or 0
        'enable_checkpointing' : False,
        'callbacks'            : [TimeTrainingCallback(), LightningProgressBar()],
        # 'profiler'             : "pytorch",
    }

    module = TrainerModule(cfg)
    trainer = pl.Trainer(**trainer_config)
    trainer.is_interrupted = False


    if phase == 'Train':
        if comment is not None:
            print(f'\nComment: \033[1m{comment}\033[0m')
        if begin_checkpoint is not None:
            state_dict = torch.load(begin_checkpoint)
            module.network.load_state_dict(state_dict)
            module.network.init_as_trained_state()

        trainer.fit(model=module, datamodule=datamodule)

        if checkpoint_path is not None and trainer_config['fast_dev_run'] is False and trainer.is_interrupted is False:
            if trainer_config["max_epochs"] > 10:
                save_path = checkpoint_path + f'-epoch{trainer_config["max_epochs"]}.ckpt'
                torch.save(module.network.state_dict(), save_path)
                print(f'Model has been save to \033[1m{save_path}\033[0m')
    else:  # Test
        state_dict = torch.load(begin_checkpoint)
        module.network.load_state_dict(state_dict)
        module.network.init_as_trained_state()
        trainer.test(model=module, datamodule=datamodule)



# -----------------------------------------------------------------------------------------
if __name__ == "__main__":

    checkpoint_path = 'runs/ckpt/DenoiseFlow-score-LCC'

    # train('Train', None, None)                      # Train from begining, and save nothing after finish
    train('Train', checkpoint_path, None)           # Train from begining, save network params after finish
    # train('Train', checkpoint_path, previous_path)  # Train from previous checkpoint, save network params after finish
    # train('Test', checkpoint_path, None)            # Test with given checkpoint
