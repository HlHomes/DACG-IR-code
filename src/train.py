import sys 
import os
import time
import pathlib
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch import Trainer, seed_everything

from net.model import DACG_IR
from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from utils.loss_utils import FFTLoss


class TeeLogger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class PrintEpochCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("loss/total_epoch")
        lr = trainer.callback_metrics.get("LR Schedule")

        loss_val = f"{loss.item():.6f}" if loss is not None else "N/A"
        lr_val = f"{lr.item():.8f}" if lr is not None else "N/A"
        
        rank_zero_info(f"\n" + "-"*40)
        rank_zero_info(f"Epoch [{trainer.current_epoch}/{trainer.max_epochs}] Completed")
        rank_zero_info(f"Avg Loss: {loss_val} | Learning Rate: {lr_val}")
        rank_zero_info("-" * 40 + "\n")

class TrainingTimeCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None

    def on_train_start(self, trainer, pl_module):

        self.start_time = time.time()
        rank_zero_info(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def on_train_end(self, trainer, pl_module):

        self.end_time = time.time()
        total_time = self.end_time - self.start_time

        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        rank_zero_info(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rank_zero_info(f"Total training time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.save_hyperparameters(self._convert_opt_to_dict(opt))

        self.net = DACG_IR(
                dim=opt.dim,
                num_blocks=opt.num_blocks,
                num_refinement_blocks=opt.num_refinement_blocks,
                heads=opt.heads,
                num_scales=opt.num_scales,
        )

        self.criterion_pixel = nn.L1Loss()
        self.criterion_fft = FFTLoss(loss_weight=0.1)

    def _convert_opt_to_dict(self, opt):
        opt_dict = vars(opt)
        for key, value in opt_dict.items():
            if isinstance(value, (np.ndarray, np.generic)):
                opt_dict[key] = value.tolist()
            elif isinstance(value, pathlib.Path):
                opt_dict[key] = str(value)
        return opt_dict
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        l_pixel = self.criterion_pixel(restored, clean_patch)
        l_fft = self.criterion_fft(restored, clean_patch)

        loss = l_pixel + l_fft

        self.log("loss/total", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss/l1", l_pixel, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/fft", l_fft, on_step=False, on_epoch=True, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
        
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )
        
        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=1,
                max_epochs=self.opt.epochs
            )      
        return [optimizer], [scheduler]


def main(opt):
    time_stamp = opt.time_stamp

    log_dir = os.path.join("single_logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"train_log_{time_stamp}.txt")
    sys.stdout = TeeLogger(log_file_path)

    rank_zero_info("="*50)
    rank_zero_info("Training Options")
    rank_zero_info("="*50)
    for key, value in sorted(vars(opt).items()):
        rank_zero_info(f"{key:<30}: {value}")
    rank_zero_info("="*50)
    rank_zero_info(f"Logging to file: {log_file_path}")

    if opt.wblogger:
        name = f"{opt.model}_{time_stamp}"
        logger = WandbLogger(
            name=name, 
            save_dir=log_dir, 
            config=opt,
            log_model="all"
        )
    else:
        logger = TensorBoardLogger(
            save_dir=log_dir,
            default_hp_metric=False
        )

    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), 
            opt=opt
        )
    else:
        model = PLTrainModel(opt)
    print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, 
        every_n_epochs=5, 
        save_top_k=-1, 
        save_last=True
    )

    time_callback = TrainingTimeCallback()
    print_callback = PrintEpochCallback()

    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)
        
    trainloader = DataLoader(
        trainset, 
        batch_size=opt.batch_size, 
        pin_memory=True, 
        shuffle=True, 
        drop_last=True, 
        num_workers=opt.num_workers
    )

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback, time_callback, print_callback], 
        accumulate_grad_batches=opt.accum_grad,
        deterministic=True,
        log_every_n_steps=10,
    )

    resume_ckpt_path = None
    if opt.resume_from:
        resume_ckpt_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
        rank_zero_info(f"\nResuming training from checkpoint: {resume_ckpt_path}")

    rank_zero_info(f"\nStarting training...")
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        ckpt_path=resume_ckpt_path
    )

    if hasattr(time_callback, 'end_time'):
        total_time = time_callback.end_time - time_callback.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        rank_zero_info(f"\n" + "="*50)
        rank_zero_info("Training Summary")
        rank_zero_info("="*50)
        rank_zero_info(f"Total training time: {hours}h {minutes}m {seconds}s")
        rank_zero_info(f"Epochs completed: {trainer.current_epoch}")
        rank_zero_info(f"Model checkpoints saved to: {checkpoint_path}")
        rank_zero_info(f"Logs saved to: {log_dir}")
        rank_zero_info("="*50)


if __name__ == '__main__':
    train_opt = train_options()

    unique_time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    train_opt.time_stamp = unique_time_stamp
    
    seed_everything(42, workers=True)

    main(train_opt)