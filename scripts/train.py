import click
import gin
import pytorch_lightning as pl
import os
import wandb
import torch
from neural_waveshaping_synthesis.data.general import GeneralDataModule
from neural_waveshaping_synthesis.data.urmp import URMPDataModule
from neural_waveshaping_synthesis.models.neural_waveshaping import NeuralWaveshaping


@gin.configurable
def get_model(model, with_wandb):
    return model(log_audio=with_wandb)


@gin.configurable
def trainer_kwargs(**kwargs):
    return kwargs


@click.command()
@click.option("--gin-file", prompt="Gin config file")
@click.option("--dataset-path", prompt="Dataset root")
@click.option("--checkpoint-path", prompt="Dataset root")
@click.option("--checkpoint-file", prompt=".ckpt file")
@click.option("--urmp", is_flag=False)
@click.option("--device", default="0")
@click.option("--instrument", default="vn")
@click.option("--load-data-to-memory", is_flag=True)
@click.option("--with-wandb", is_flag=True, default=True)
@click.option("--restore-checkpoint", is_flag=True, default=False)
def main(
        gin_file,
        dataset_path,
        checkpoint_path,
        checkpoint_file,
        urmp,
        device,
        instrument,
        load_data_to_memory,
        with_wandb,
        restore_checkpoint,
):
    gin.parse_config_file(gin_file)
    torch.autograd.set_detect_anomaly(True)
    print(f"torch: {torch.__version__}")
    print(f"CUDA #devices: {torch.cuda.device_count()}")
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    print(f"Device: {device}")

    model = get_model(with_wandb=with_wandb)

    if restore_checkpoint:
        print(f"Loading model from {checkpoint_path}/{checkpoint_file}")
        model = NeuralWaveshaping.load_from_checkpoint(
            checkpoint_path=os.path.join(checkpoint_path, "checkpoints", checkpoint_file),
            map_location=torch.device(device))
        print(f"Epoch: {model.current_epoch}")
        print(f"Step: {model.global_step}")
        print(f"")
    else:
        print("NOT loading from checkpoint...")

    if urmp:
        print("DataModule is URMP")
        data = URMPDataModule(
            dataset_path,
            instrument,
            load_to_memory=load_data_to_memory,
            num_workers=16,
            shuffle=True,
        )
    else:
        print("DataModule is GeneralDataModule")
        data = GeneralDataModule(
            dataset_path,
            load_to_memory=load_data_to_memory,
            num_workers=16,
            shuffle=True,
        )

    checkpointing = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", save_top_k=3, save_last=True,
        dirpath=os.path.join(checkpoint_path, "checkpoints"),
        # every_n_epochs=20
    )
    callbacks = [checkpointing]
    if with_wandb:
        lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_logger)
        logger = pl.loggers.WandbLogger(project="nws")
        logger.watch(model, log="all")
        wandb.watch(model)

    kwargs = trainer_kwargs()
    trainer = pl.Trainer(
        logger=logger if with_wandb else None,
        callbacks=callbacks,
        # gpus=device,
        # resume_from_checkpoint=restore_checkpoint if restore_checkpoint != "" else None,
        **kwargs
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
