import torch
from datetime import datetime


def add_misc_options(parser):
    group = parser.add_argument_group("Miscellaneous options")
    group.add_argument("--checkpoints_dir", default="checkpoints/" + datetime.now().strftime("%Y%m%d_%H%M%S"), help="General directory to save experiments")
    group.add_argument("--checkpoint_path", default="checkpoints_paper/encoder.p", help="checkpoint path to load encoder parameteres")


def add_cuda_options(parser):
    group = parser.add_argument_group("Cuda options")
    group.add_argument("--cuda", dest="cuda", action="store_true", help="If we want to try to use gpu")
    group.add_argument("--cuda_index", dest="cuda_index", type=int, default=0)
    group.add_argument("--cpu", dest="cuda", action="store_false", help="If we want to use cpu")
    group.set_defaults(cuda=True)

    
def adding_cuda(parameters):
    if parameters["cuda"] and torch.cuda.is_available():
        parameters["device"] = torch.device("cuda", index=parameters["cuda_index"])
    else:
        parameters["device"] = torch.device("cpu")
