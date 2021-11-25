
import os
import sys
sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl

from torch import Tensor
from argparse import ArgumentParser

from dataset.dmrdenoise.dataset import DMRDenoiseDataModule
from models.deflow.deflow import DenoiseFlow, Disentanglement
from metric.loss import EarthMoverDistance as EMD
# from metric.loss import ChamferDistance as CD

from modules.utils.callback import TimeTrainingCallback
from modules.utils.lightning import LightningModule
from modules.utils.modules import print_progress_log


# -----------------------------------------------------------------------------------------
class TrainerModule(LightningModule):

    def __init__(self, cfg):
        super(TrainerModule, self).__init__()

        self.disentangle_method = Disentanglement.FBM
        self.network = DenoiseFlow(self.disentangle_method, pc_channel=3)
        self.loss_emd = EMD()

        self.epoch = 0
        self.cfg = cfg

        self.min_noisy_v = {
            'noisy_0.01': 0.23,
            'noisy_0.03': 0.40,
            'noisy_0.08': 1.90,
        }

    def forward(self, p: Tensor, **kwargs):
        return self.network(p, **kwargs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.sched_patience, factor=self.cfg.sched_factor, min_lr=self.cfg.min_lr)
        # return { 'optimizer': optimizer, 'lr_scheduler': { 'scheduler': scheduler, 'monitor': 'EMD' } }

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.sched_patience, factor=self.cfg.sched_factor, min_lr=self.cfg.min_lr)
        # return { "optimizer": optimizer, 'scheduler': scheduler }
        return optimizer

    def training_step(self, batch, batch_idx):

        # LCC
        noise, noiseless = batch['pos'], batch['clean']
        denoised, logpx, consistency = self(noise, y=noiseless)

        emd = self.loss_emd(denoised, noiseless)
        loss = logpx * 1e-6 + emd * 0.1 + consistency * 10.0

        self.log('EMD', emd.detach().cpu().item() * 0.1, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('logpx', logpx * 1e-6, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('consistency', consistency * 10.0, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):

        output = {}

        for key in self.trainer.datamodule.val_noisy_item_keys:
            noise, noiseless = batch[key], batch['clean']
            denoised, _, _ = self(noise)
            output[key] = self.loss_emd(denoised, noiseless)

        return output

    def validation_epoch_end(self, batch):
        log_dict = {}
        keys = batch[0].keys()

        for key in keys:
            batch_size = 8
            n = len(batch) * batch_size
            log_dict[key] = torch.tensor([x[key] for x in batch]).sum().detach().cpu() / n

        extra = []
        # if self.epoch % 5 == 0:
        #     save_path = f'runs/ckpt/DenoiseFlow-dmr-epoch{self.epoch}.ckpt'
        #     torch.save(self.network.state_dict(), save_path)
        #     extra.append(str(self.epoch))
        for key in keys:
            if log_dict[key] < self.min_noisy_v[key]:
                self.min_noisy_v[key] = log_dict[key]
                save_path = f'runs/ckpt/DenoiseFlow-dmr-{self.disentangle_method.name}-min_{key}.ckpt'
                torch.save(self.network.state_dict(), save_path)
                extra.append(key)

        print_progress_log(self.epoch, log_dict, extra=extra)
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

    parser.add_argument('--noise_low1', default=0.02, type=float)  # 0.02
    parser.add_argument('--noise_high1', default=0.06, type=float, help='-1 for fixed noise level')  # 0.06
    parser.add_argument('--noise_low2', default=None, type=float)  # 0.02
    parser.add_argument('--noise_high2', default=None, type=float, help='-1 for fixed noise level')  # 0.06
    parser.add_argument('--aug_scale', action='store_true', help='Enable scaling augmentation.')
    parser.add_argument('--datasets', type=list, default=[
        'data/DMRDenoise/dataset_train/patches_20k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_10k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_30k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_50k_1024.h5',
        'data/DMRDenoise/dataset_train/patches_80k_1024.h5',
    ])
    parser.add_argument('--subset_size', default=7000, type=int, help='-1 for unlimited')  # 7000
    parser.add_argument('--batch_size', default=11, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path=None, begin_checkpoint=None):

    comment = 'dmrdenoise set'
    cfg = model_specific_args().parse_args()
    pl.seed_everything(cfg.seed)

    dataset_cfg = dataset_specific_args().parse_args()
    datamodule = DMRDenoiseDataModule(dataset_cfg)

    trainer_config = {
        'default_root_dir'     : './runs/',
        'gpus'                 : 1,  # Set this to None for CPU training
        'fast_dev_run'         : False,
        'max_epochs'           : 200, # cfg.max_epoch,
        'weights_summary'      : 'top',  # 'top', 'full' or None
        'precision'            : 32,   # 16
        # 'amp_level'            : 'O1',
        'gradient_clip_val'    : 1e-3,
        'deterministic'        : False,
        'num_sanity_val_steps' : -1,  # -1 or 0
        'checkpoint_callback'  : False,
        'callbacks'            : [TimeTrainingCallback()],
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

    checkpoint_path = 'runs/ckpt/DenoiseFlow-dmr-FBM'
    # previous_path = 'runs/ckpt/358c73d-DenoiseFlow-baseline-epoch50.ckpt'

    # train('Train', None, None)                      # Train from begining, and save nothing after finish
    train('Train', checkpoint_path, None)           # Train from begining, save network params after finish
    # train('Train', checkpoint_path, previous_path)  # Train from previous checkpoint, save network params after finish
    # train('Test', checkpoint_path, None)            # Test with given checkpoint
