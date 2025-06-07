import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pl_modules.datamodule import SoundDataModule
from pl_modules.model import AlexNetLightning
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.loggers import MLFlowLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)
    dm = SoundDataModule(config)
    model = AlexNetLightning(
        input_shape=tuple(config.model.input_shape),
        num_classes=config.model.num_classes,
    )

    loggers = [
        MLFlowLogger(
            experiment_name="cats-and-dogs",
            run_name="conv-classifier",
            save_dir=".",
            tracking_uri="http://127.0.0.1:8080",
        )
    ]

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
        RichModelSummary(max_depth=2),
        EarlyStopping(monitor="val_loss", patience=10),
    ]

    callbacks.append(
        ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["model"]["save_top_k"],
            every_n_epochs=config["model"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,  # to resolve warnings
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
