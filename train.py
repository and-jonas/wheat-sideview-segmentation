
# includes code from https://www.kaggle.com/code/karthikrangasai/hyperparameter-search-using-optuna-for-flash and
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py

import gc
import os

import optuna

import torch
import torchmetrics

from torch.utils.data.sampler import RandomSampler

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from flash.image import SemanticSegmentation, SemanticSegmentationData

from flash import Trainer

# from transforms import OptSemSegInputTransform  # NOT WORKING
from transforms import set_input_transform_options
from BatchSizeFinder import find_max_batch_size, find_max_batch_size_simple

import glob

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

study_name = "sideview"

metrics = [
    torchmetrics.Accuracy(num_classes=3, mdmc_reduce='global', multiclass=True),
    torchmetrics.F1Score(num_classes=3, mdmc_reduce='global', multiclass=True),
    torchmetrics.Precision(num_classes=3, mdmc_reduce='global',  multiclass=True),
    torchmetrics.Recall(num_classes=3, mdmc_reduce='global', multiclass=True),
    # torchmetrics.JaccardIndex(num_classes=2, multilabel=False)
    ]

# metrics = [torchmetrics.Accuracy(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False),
#            torchmetrics.F1Score(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False),
#            torchmetrics.Precision(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False),
#            torchmetrics.Recall(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False)
#            # torchmetrics.JaccardIndex(num_classes=2, multilabel=False)
#            ]

# predict_files = glob.glob("/projects/segment-sideview/data/prediction/*.png")

# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=2)
loss_fn = torch.nn.CrossEntropyLoss()
crop_factor = 0.737
backbone = "resnet50"
head = "unetplusplus"
size = 500
optimizer = "sgd"
momentum = 0.8
learning_rate = 0.062
strategy = "no_freeze"
p_color_jitter = 0
blur_kernel_size = 1
rand_rot = True
scaling = True

transform = set_input_transform_options(head=head,
                                        size=size,
                                        crop_factor=0.64,
                                        blur_kernel_size=blur_kernel_size,
                                        p_color_jitter=p_color_jitter,
                                        rand_rot=rand_rot,
                                        scaling=scaling)

batch_size = find_max_batch_size_simple(
    backbone=backbone,
    head=head,
    size=size,
    crop_factor=crop_factor,
    max_batch_size=200,
)

datamodule = SemanticSegmentationData.from_folders(
    train_folder="/projects/segment-sideview/data/train/images",
    train_target_folder="/projects/segment-sideview/data/train/masks",
    val_folder="/projects/segment-sideview/data/validation/images",
    val_target_folder="/projects/segment-sideview/data/validation/masks",
    train_transform=transform,
    val_transform=transform,
    test_transform=transform,
    predict_transform=transform,
    num_classes=3,
    batch_size=batch_size,
    num_workers=8,
    sampler=RandomSampler,
)

# Build the task
model = SemanticSegmentation(
    pretrained=True,
    backbone=backbone,
    head=head,
    num_classes=datamodule.num_classes,
    metrics=metrics,
    loss_fn=loss_fn,
    optimizer=(optimizer, {"momentum": momentum}),
    learning_rate=learning_rate,
    # lr_scheduler=("cosineannealinglr", {"T_max": 750}),
)
# model.available_lr_schedulers()

# must be specified inside the objective function
# does NOT overwrite otherwise
early_stopping = EarlyStopping(monitor='val_f1score', mode='max', patience=7)
lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
checkpointing = ModelCheckpoint(
    dirpath="/projects/segment-sideview",
    save_top_k=1,
    monitor="val_f1score",
    mode="max",
    filename="sideview_ears_best-{epoch:02d}-{step:.2f}",
    save_weights_only=False,
    # save_on_train_epoch_end=False,
)

# must be specified inside the objective()
# overwrites otherwise
logger = TensorBoardLogger(save_dir="/projects/segment-sideview",
                           # default_hp_metric=True,
                           name=study_name)

logger.log_hyperparams({
    "head": head,
    "backbone": backbone,
    "strategy": strategy,
    "blur_kernel_size": blur_kernel_size,
    "optimizer": optimizer,
    "learning_rate": learning_rate,
    "momentum": momentum,
    "size": size,
    "p_color_jitter": p_color_jitter,
    "batch_size": datamodule.batch_size})

# 3. Create the trainer and finetune the model
trainer = Trainer(max_epochs=200,
                  # move_metrics_to_cpu=True,
                  gpus=[2],
                  precision=16,
                  logger=logger,
                  callbacks=[early_stopping, lr_monitor, checkpointing],
                  # callbacks=[early_stopping, lr_monitor],
                  # reload_dataloaders_every_n_epochs=1,
                  enable_checkpointing=True,
                  # log_every_n_steps=10,
                  # limit_train_batches=0.50  # only use 50% of training data in each epoch
                  # auto_lr_find=True
                  # limit_val_batches=0,
                  # auto_scale_batch_size="binsearch",  # TODO not having any effect, i.e. still running OOM
                  # accumulate_grad_batches=accumulate_grad_batches,  # TODO not having any effect
                  # limit_train_batches=2,
                  # limit_val_batches=1,rm
                  )

# Train the model for Optuna to understand the current
# Hyperparameter combination's behaviour.
trainer.finetune(model, datamodule=datamodule, strategy='freeze')
trainer.save_checkpoint("/projects/segment-sideview/sideview_ff.pt")