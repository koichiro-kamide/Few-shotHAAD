import torch
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn import metrics

from src.utils.torch import *
from src.utils.logger import *
from src.parser.training import parser
from src.datasets.humanact12poses import HumanAct12Poses
from src.models.action_encoder.encoder import Encoder
from src.models.humanmac.motion_generator import motion_generator


# -------------------------------------------------------------
# Build support embeddings with optional data augmentation
# -------------------------------------------------------------
def build_support_embeddings(support_set, encoder):
    for traj in support_set:
        traj = traj.to(parameters["device"])
        with torch.no_grad():
            bs, nframes, njoints, nfeats = traj.shape
            # Augment the support set by generating additional motion samples from each original trajectory
            if parameters["num_augment"] > 0:
                for sample in traj:
                    aug_samples = motion_generator(sample=sample, 
                                                parameters=parameters, 
                                                num_generate=parameters["num_augment"])
                    traj = torch.cat((traj, aug_samples), dim=0)
            # Extract embeddings
            support_embeddings = encoder(traj)
    return support_embeddings


# -------------------------------------------------------------
# Build embeddings for test trajectories
# -------------------------------------------------------------           
def build_test_embeddings(test_dataloader, encoder):
    for traj_test, y_labels in test_dataloader:
        traj_test = traj_test.to(parameters["device"])
        with torch.no_grad():
            bs, nframes, njoints, nfeats = traj_test.shape
            test_embeddings = encoder(traj_test)
    return test_embeddings, y_labels


# -------------------------------------------------------------
# Compute anomaly scores as the average distance to support embeddings
# -------------------------------------------------------------
def calc_ave_dist(support_embeddings, test_embeddings):
    distances = torch.cdist(test_embeddings, support_embeddings, p=2)
    scores = distances.mean(dim=1)
    # Normalize scores for stable evaluation
    scores_normalized = F.normalize(scores, dim=0)
    scores_normalized = scores_normalized.detach().cpu()
    return scores_normalized


def val():
    print(f'action:{parameters["normal_action"]}  '
          f'seed:{parameters["seed"]}  '
          f'num_support:{parameters["num_support"]}  '
          f'num_augment:{parameters["num_augment"]}')

    # Build support embeddings (normal reference set) and test embeddings
    support_embeddings = build_support_embeddings(support_set, encoder)
    test_embeddings, y_labels = build_test_embeddings(test_dataloader, encoder)

    # Compute anomaly scores
    scores = calc_ave_dist(support_embeddings, test_embeddings)
    
    # ROC-AUC evaluation
    fpr, tpr, threshold = metrics.roc_curve(y_labels, scores, drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)
    print('auc: ', auc)
    print('***********************************************')


if __name__ == '__main__':
    # Parse options
    parameters = parser()

    # Init random
    np.random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.set_device(parameters["cuda_index"])

    # Get datasets
    dataset = HumanAct12Poses(normal_action_name=parameters["normal_action"], mode='test', **parameters)
    test_dataloader = dataset.test_dataloader()
    support_set = dataset.support_dataloader()

    # Get encoder
    encoder = Encoder(njoints=parameters["num_joints"],  nfeats=parameters["coord_dim"], nframes=parameters["num_frames"], gpu_index=parameters["cuda_index"])
    cp_path = parameters["checkpoint_path"]
    model_cp = pickle.load(open(cp_path, "rb"))
    encoder.load_state_dict(model_cp['model_dict'], strict=False)
    encoder.to(parameters["device"])
    encoder.eval()

    val()