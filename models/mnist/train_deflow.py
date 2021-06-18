
import os
import sys
sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl

from torch import Tensor
from omegaconf import OmegaConf

from dataset.mnist import MNISTDataModule
from models.mnist.deflow import DenoiseFlow
from metric.loss import DenoiseMetrics
from plot.plot_mnist import plot_mnist

from modules.utils.callback import TimeTrainingCallback
from modules.utils.lightning import LightningModule
from modules.utils.modules import print_progress_log



# -----------------------------------------------------------------------------------------
class TrainerModule(LightningModule):

    def __init__(self):
        super(TrainerModule, self).__init__()

        self.network = DenoiseFlow()

        self.train_metric = DenoiseMetrics(dim=2)
        self.valid_metric = DenoiseMetrics(dim=2)
        self.epoch = 1
        
        self.cache = None

    def forward(self, p: Tensor, **kwargs):
        return self.network(p, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):

        clean_pts, noise_pts, noise_probs = batch

        pred_pts, probs, logpx = self(noise_pts)

        metrics = self.train_metric.evaluate(pred_pts, clean_pts, probs, noise_probs)
        loss = logpx * 1e-4 + metrics['CD'] * 10.0 + metrics['Confi'] * 1.0

        self.log('CD', metrics['CD'] * 10.0, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('confi', metrics['Confi'] * 0.5, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('logpx', logpx * 1e-4, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):

        clean_pts, noise_pts, noise_probs = batch
        pred_pts, probs, logpx = self(noise_pts)

        metrics = self.valid_metric.evaluate(pred_pts, clean_pts, probs, noise_probs)

        if self.cache is None:
            self.cache = [
                noise_pts.detach().cpu().numpy(),
                pred_pts.detach().cpu().numpy(),
                probs.detach().cpu().numpy(),
            ]

        return {
            'vloss': logpx.detach().cpu(),
            'CD'   : metrics['CD'],
            'Confi': metrics['Confi'],
        }

    def validation_epoch_end(self, batch):

        log_dict = {
            'vloss': torch.tensor([x['vloss'] * 1e-5 for x in batch]).sum().item(),
            'CD'   : torch.tensor([x['CD']           for x in batch]).sum().item(),
            'Confi': torch.tensor([x['Confi']        for x in batch]).sum().item(),
        }
        
        plot_mnist([
            self.cache[0][0], self.cache[1][0],
            self.cache[0][1], self.cache[1][1],
            self.cache[0][2], self.cache[1][2],
        ], save_path=f'runs/mnist-figures/Epoch-{self.epoch}.png', colors=[
            None, self.cache[2][0],
            None, self.cache[2][1],
            None, self.cache[2][2],
            # None, None,
            # None, None,
            # None, None,
        ], matrix_shape=(3, 2), is_show=None)
        self.cache = None

        print_progress_log(self.epoch, log_dict)
        self.epoch += 1


# -----------------------------------------------------------------------------------------
def train(phase='Train', checkpoint_path: str=None, begin_checkpoint: str=None):

    comment = 'Baseline'

    dataset_cfg = OmegaConf.create({
        'rootdir' : 'data/MNIST/mnist2d-pointcloud-1024.h5',
        'batch_size': 32,  # 32
        'noise_shape': (32, 2),
        'classes': 4,
    })
    datamodule = MNISTDataModule(dataset_cfg)

    trainer_config = {
        'default_root_dir'     : './runs/',
        'gpus'                 : 1,  # Set this to None for CPU training
        'fast_dev_run'         : False,
        'max_epochs'           : 20,
        'precision'            : 32,   # 16
        # 'amp_level'            : 'O1',
        'weights_summary'      : 'top',  # 'top', 'full' or None
        'gradient_clip_val'    : 1e-3,
        'deterministic'        : False,
        'num_sanity_val_steps' : -1,  # -1 or 0
        'checkpoint_callback'  : False,
        'callbacks'            : [TimeTrainingCallback()],
    }

    module = TrainerModule()
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
