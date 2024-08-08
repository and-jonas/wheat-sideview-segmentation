
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

# predict_files = glob.glob("/projects/SegEar/data/prediction/*.png")

loss_fn = torch.nn.CrossEntropyLoss()
crop_factor = 0.737
# crop_factor = 0.64


# objective function
def objective(trial: optuna.Trial):

    # This may be required to avoid "OSError: [Errno 24] Too many open files" from multiprocessing
    datamodule = None

    # MODEL OPTIMIZATION -----------------------------------------------------------------------------------------------
    # head = trial.suggest_categorical(name="head", choices=["deeplabv3plus", "unetplusplus", "fpn"])
    head = "unetplusplus"
    # backbone = trial.suggest_categorical(name="backbone",  choices=["resnet18", "resnet34", "resnet50"])
    backbone = "resnet50"
    # strategy = trial.suggest_categorical(name="strategy", choices=["freeze", "no_freeze", "train"])
    strategy = "no_freeze"

    # TRAINING OPTIMIZATION --------------------------------------------------------------------------------------------
    # most important parameter to tune ?
    # https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
    # learning_rate = trial.suggest_loguniform(name="learning_rate", low=10**(-4), high=10**(-1))
    learning_rate = 0.062
    momentum = 0.8
    # momentum = trial.suggest_uniform("momentum", 0.5, 0.99)
    # optimizer = trial.suggest_categorical(name="optimizer", choices=["adam", "sgd"])
    optimizer = "sgd"
    # accumulate_grad_batches = trial.suggest_int("accumulate_grad_batches", low=1, high=1, step=1)
    # batch_size = trial.suggest_int("batch_size", 8, 24, step=16)

    # TRANSFORM OPTIMIZATION -------------------------------------------------------------------------------------------
    # meaningful ranges were defined with help of https://kornia.readthedocs.io/en/latest/enhance.html
    # brightness = trial.suggest_float('brightness', 0.0, 0.5)
    # contrast = trial.suggest_float("contrast", 0.7, 1.3)
    # hue = trial.suggest_float("hue", 0.0, 0.2)
    # saturation = trial.suggest_float("saturation", 0.8, 1.0)
    # blur_kernel_size = trial.suggest_int("kernel_size", low=1, high=7, step=2)
    blur_kernel_size = 1
    size = trial.suggest_int("size", low=300, high=600, step=50)
    # size = 600
    # p_color_jitter = trial.suggest_float("p_color_jitter", low=0.0, high=0.6, step=0.15)
    p_color_jitter = 0
    # rand_rot = trial.suggest_categorical(name="rand_rot", choices=[True, False])
    rand_rot = True
    # scaling = trial.suggest_categorical(name="scaling", choices=[True, False])
    scaling = False

    # T = SemSegInputTransform
    # T = set_input_transform_options(train_size=size, val_size=size, predict_size=size,
    #                                 brightness=brightness, contrast=contrast, hue=hue, saturation=saturation,
    #                                 kernel_size=kernel_size)
    transform = set_input_transform_options(head=head,
                                            size=size,
                                            blur_kernel_size=blur_kernel_size,
                                            p_color_jitter=p_color_jitter,
                                            rand_rot=rand_rot,
                                            scaling=scaling)

    # # get the largest possible batch size depending on image size and crop factor
    # # workaround because auto_scale_batch_size="binsearch" is not working
    # batch_size = find_max_batch_size(
    #     backbone=backbone,
    #     head=head,
    #     dataset_size=153,
    #     transform=transform,
    #     init_batch_size=2,
    #     max_batch_size=76,
    #     strategy=strategy,
    #     loss_fn=loss_fn,
    #     metrics=metrics,
    #     optimizer=optimizer,
    #     learning_rate=learning_rate,
    #     momentum=momentum,
    #     n_iterations=40)
    # batch_size = int(np.floor(3200000 / ((size * crop_factor) ** 2)))  # possible if a single bb is used
    # print(colored("using batch_size = " + str(batch_size), 'red'))
    batch_size = find_max_batch_size_simple(
        backbone=backbone,
        head=head,
        size=size,
        crop_factor=crop_factor,
        max_batch_size=200,
    )
    # batch_size = trial.suggest_int("batch_size", low=2, high=batch_size, step=2)

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
        # pin_memory=True,  # might result in "OSError: [Errno 24] Too many open files" from multiprocessing
        # persistent_workers=True  # might result in "OSError: [Errno 24] Too many open files" from multiprocessing
    )

    pretrained = False if strategy == "train" else True

    # Build the task
    model = SemanticSegmentation(
        pretrained=pretrained,
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
    early_stopping = EarlyStopping(monitor='val_f1score', mode='max', patience=15)
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    # checkpointing = ModelCheckpoint(
    #     dirpath="/projects/segment-sideview",
    #     save_top_k=1,
    #     monitor="val_f1score",
    #     mode="max",
    #     filename="sideview_final-{epoch:02d}-{step:.2f}",
    #     save_weights_only=False,
    #     # save_on_train_epoch_end=False,
    # )

    # must be specified inside the objective()
    # overwrites otherwise
    logger = TensorBoardLogger(save_dir="/projects/segment-sideview",
                               # default_hp_metric=True,
                               name=study_name)

    logger.log_hyperparams({"head": head,
                            "backbone": backbone,
                            "strategy": strategy,
                            "blur_kernel_size": blur_kernel_size,
                            "optimizer": optimizer,
                            "learning_rate": learning_rate,
                            "momentum": momentum,
                            "size": size,
                            "rand_rot": rand_rot,
                            "scaling": scaling,
                            "p_color_jitter": p_color_jitter,
                            "batch_size": datamodule.batch_size})

    # 3. Create the trainer and finetune the model
    trainer = Trainer(max_epochs=150,
                      move_metrics_to_cpu=False,
                      gpus=[2],
                      precision=16,
                      logger=logger,
                      # callbacks=[early_stopping, lr_monitor, checkpointing],
                      callbacks=[early_stopping, lr_monitor],
                      # reload_dataloaders_every_n_epochs=1,
                      enable_checkpointing=False,
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
    if strategy == "train":
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.finetune(model, datamodule=datamodule, strategy=strategy)
    trainer.save_checkpoint("/projects/segment-sideview/sideview_ff.pt")

    # The extra step to tell Optuna which value to base the
    # optimization routine on.
    # But this only gets the metrics of the LAST iteration
    value = trainer.callback_metrics["val_f1score"].item()
    # Get the highest metric from all iterations; based on
    # https://www.programcreek.com/python/example/114903/tensorboard.backend.event_processing.event_accumulator.EventAccumulator
    event_acc = EventAccumulator(path=logger.log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']  # available metrics
    v = event_acc.Scalars('val_f1score')  # use validation f1score
    v = [v[i].value for i in range(len(v))]
    value = max(v)  # get the highest value, since direction='maximize'

    # log this value
    logger.log_metrics({"hp_metric": value})

    # get rid of everything
    del datamodule, model, trainer, transform, logger
    gc.collect()
    torch.cuda.empty_cache()

    return value


if __name__ == "__main__":

    os.chdir('/projects/segment-sideview')

    search_space = {"size": [300, 350, 400, 450, 500, 550, 600]}
    # search_space = {"size": [350, 370, 390, 410, 430, 450, 470, 490, 510, 530, 550, 570, 590]}

    # TPESampler is used for value suggestion, if not specified otherwise
    study = optuna.create_study(study_name=study_name,
                                storage="sqlite:///sideview_final.db",
                                load_if_exists=True,
                                direction='maximize',
                                sampler=optuna.samplers.GridSampler(search_space),
                                # pruner=optuna.pruners.MedianPruner()
                                pruner=optuna.pruners.NopPruner(),
                                )
    # study.optimize(objective, n_trials=36, gc_after_trial=True)
    study.optimize(objective, gc_after_trial=True)

