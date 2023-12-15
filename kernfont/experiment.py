import comet_ml
import os
import torch
from dotenv import load_dotenv
from kernfont.data import get_dataloader
from kernfont.model import get_model
from kernfont.parameters import parse_params, load_yaml
from kernfont.logger.logger import logger
from kernfont.model.loss import get_loss_criterion, get_optimizer
from kernfont.trainer.ez_train import run_train


def run_experiment(args):
    load_dotenv()
    logger.info("Running experiment")
    parameters = load_yaml(args.parameters)

    (
        experiment_params,
        train_params,
        dataset_params,
        model_params,
    ) = parse_params(parameters)

    comet = {
        "apykey": os.getenv("COMET_API_KEY"),
        "project_name": experiment_params.get("name"),
    }

    experiment = comet_experiment(comet, parameters, train_params)
    experiment.log_parameters(train_params)

    train_loader, val_loader = get_dataloader(
        experiment_params.get("annotations"),
        **dataset_params,
    )
    model = get_model(model_params.get("name"))
    loss = get_loss_criterion(train_params.get("loss").get("name"))
    optimizer = get_optimizer(
        model, train_params.get("optimizer"), train_params.get("initial_lr")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_train(
        model=model,
        train_loader=train_loader,
        criterizion=loss,
        optimizer=optimizer,
        experiment=experiment,
        epochs=train_params.get("max_epochs"),
        device=device,
    )


# set configuration
def comet_experiment(comet_information, parameters, train_params):
    comet_ml.init(comet_information)
    experiment = comet_ml.Experiment()
    experiment.add_tags(parameters.get("parameters").get("tags"))
    experiment.log_parameters(train_params)
    return experiment
