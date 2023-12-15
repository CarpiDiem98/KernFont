from kernfont.trainer.train import Trainer


def get_trainer(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    experiment,
):
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        experiment,
    )
    return trainer
