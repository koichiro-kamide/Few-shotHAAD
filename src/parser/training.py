import os
from argparse import ArgumentParser
from src.parser.base import add_misc_options, add_cuda_options, adding_cuda
from src.parser.tools import save_args
from src.parser.dataset import add_dataset_options
from src.parser.model import add_model_options


def add_training_options(parser):
    group = parser.add_argument_group("Training options")
    group.add_argument("--run_mode", type=str, choices=["train", "test"], default="train",help="Execution mode (train or test)")
    group.add_argument("--normal_action", type=str, default=None, help="One category selected as normal")
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--num_epochs", type=int, required=True, help="Number of epochs of training")
    group.add_argument("--batch_size", type=int, required=True, help="Size of the batches")
    group.add_argument("--lr", type=float, help="AdamW: learning rate")
    group.add_argument("--snapshot", type=int, help="Frequency of saving model checkpoints")
    

def parser():
    parser = ArgumentParser()

    # misc options
    add_misc_options(parser)

    # cuda options
    add_cuda_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    add_model_options(parser)

    # training options
    add_training_options(parser)

    opt = parser.parse_args()
    
    # remove None params, and create a dictionnary
    parameters = {key: val for key, val in vars(opt).items() if val is not None}

    if parameters.get("num_augment", 0) < 1:
        parameters.pop("t_his", None)
        parameters.pop("t_pred", None)
        parameters.pop("n_pred", None)

    if parameters.get("run_mode") == "train":
        parameters.pop("num_support", None)
        os.makedirs(parameters["checkpoints_dir"], exist_ok=True)
        save_args(parameters, folder=parameters["checkpoints_dir"])

    if parameters.get("run_mode") == "test":
        parameters.pop("lr", None)
        parameters.pop("snapshot", None)

    adding_cuda(parameters)
    
    return parameters
