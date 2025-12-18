import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI

# Importing our production-ready components
from model import AdvancedGyroNet
from pipeline import GyroDataModule

def main():
    """
    Main entry point for training.
    Usage:
        python train.py --config config.yaml
        python train.py --trainer.devices 4 --data.batch_size 64
    """
    cli = LightningCLI(
        model_class=AdvancedGyroNet,
        datamodule_class=GyroDataModule,
        save_config_callback=ModelCheckpoint, # Saves the config with the weights
        run=False, # We will manually run to add extra logic if needed
        parser_kwargs={"parser_mode": "omegaconf"},
        trainer_defaults={
            "accelerator": "auto",
            "devices": "auto",
            "strategy": "ddp_find_unused_parameters_true", # For multi-GPU
            "precision": "16-mixed", # Mixed precision for 2x speedup on modern GPUs
            "max_epochs": 100,
            "callbacks": [
                ModelCheckpoint(
                    monitor="val/total_loss",
                    filename="gyronet-{epoch:02d}-{val_loss:.4f}",
                    save_top_k=3,
                    mode="min"
                ),
                EarlyStopping(
                    monitor="val/total_loss",
                    patience=10,
                    mode="min"
                ),
                LearningRateMonitor(logging_interval="step")
            ]
        }
    )

    # Add Weights & Biases or Tensorboard logging
    # cli.trainer.logger = WandbLogger(project="gyro-kinetic-ai")

    # Start training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    main()
