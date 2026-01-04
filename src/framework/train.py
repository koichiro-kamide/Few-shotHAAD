import os

import torch
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle

from src.utils.torch import *
from src.utils.logger import *

from src.parser.training import parser
from src.datasets.humanact12poses import HumanAct12Poses
from src.datasets.augment_dataset import augment_train_dataset
from src.models.action_encoder.encoder import Encoder
from torch.nn.functional import cosine_similarity


def augmented_loss_function(sims):
    """
    Compute an augmented contrastive loss given a similarity matrix.

    Args:
        sims (Tensor): Pairwise similarity matrix of shape [N, N],
                       where N = num_categories * positive_group_size.
                       sims[i, j] represents the similarity between sample i and j.

    Returns:
        Tensor: Scalar loss value.
    """
    # ---------------------------------------------------------
    # Construct a positive-pair mask
    # mask_positive[i, j] = 1 if (i, j) is a positive pair
    # mask_positive[i, j] = 0 otherwise (negative pairs or self-pairs)
    # ---------------------------------------------------------

    # Number of semantic categories (e.g., action categories)
    num_cat = 9
    # Number of samples per category: for each category, we form positive pairs from the original sample and its num_augment augmentations
    positive_group_size = 2 * (1 + parameters["num_augment"])
    # Initialize all entries as negative (0)
    mask_positive = torch.zeros((len(sims), len(sims)), device=parameters["device"])
    # For each category, mark intra-group pairs as positive
    for i in range(num_cat):
        start = i * positive_group_size
        end = start + positive_group_size
        # Set all pairs within the same category block to 1
        mask_positive[start:end, start:end] = 1
        # Remove self-pairs by subtracting the identity matrix (i.e., diagonal elements are not considered positive pairs)
        mask_positive[start:end, start:end] -= torch.eye(n=positive_group_size, device=parameters["device"])

    # ---------------------------------------------------------
    # Compute softmax-based contrastive loss
    # ---------------------------------------------------------

    # Exponentiate similarities to ensure positivity
    exp_sims = torch.exp(sims)
    # Remove self-similarities from the denominator
    mask = (~torch.eye(n=len(exp_sims), dtype=bool, device=parameters["device"])).float()
    exp_sims = mask * exp_sims
    # Denominator for softmax: sum of similarities to all other samples (positives + negatives)
    denominators = torch.sum(exp_sims, dim=1, keepdim=True)
    # Softmax-normalized similarities for each anchor sample
    softmax_vals = exp_sims / (denominators + 1e-8)  # avoid division by zero
    # Compute negative log-likelihood only for positive pairs
    losses = -torch.log(softmax_vals + 1e-8) * mask_positive
    # Average loss over all positive pairs
    final_loss = torch.sum(losses) / torch.sum(mask_positive)
    return final_loss


def loss_function(sims):
    exp_sims = torch.exp(sims)
    simij = torch.diag(exp_sims, 9)
    simji = torch.diag(exp_sims, -9)
    positives = torch.cat([simij, simji], dim=0)
    mask = (~torch.eye(n=len(exp_sims), dtype=bool, device=parameters["device"])).float()
    pos_and_negatives = mask * exp_sims
    denominator = torch.sum(pos_and_negatives, dim=1)
    losses = -torch.log(positives/denominator)
    final_loss = torch.mean(losses)
    return final_loss


def train(epoch):
    encoder.train()
    t_s = time.time()
    total_loss = 0
    total_num_batch = 0

    train_dataloader = dataset.train_dataloader(num_samples=948, batch_size=parameters["batch_size"])
    if parameters["num_augment"] > 0:
        train_dataloader = dataset.augmented_train_dataloader(num_samples=948, batch_size=parameters["batch_size"], aug_trainset=augmented_trainset)
    
    for traj in train_dataloader:
        traj = traj.to(parameters["device"])  # (bs, nframes, njoints, nfeats)
        z = encoder(traj)
        sims = cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        if parameters["num_augment"] > 0:
            loss = augmented_loss_function(sims)
        else:
            loss = loss_function(sims)
        total_loss = total_loss + loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num_batch = total_num_batch + 1
        del loss, z

    scheduler.step()
    train_loss = total_loss / total_num_batch
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('loss', train_loss, epoch)
    writer.add_scalar('lr', lr, epoch)
    logger.info(f"Epoch {epoch:>3}: loss={train_loss:.4f}, lr={lr:.5f}, Time={time.time() - t_s:.3f} [s]")


if __name__ == '__main__':
    # parse options
    parameters = parser()

    # init random
    np.random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])

    # set gpu
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if torch.cuda.is_available():
        torch.cuda.set_device(parameters["cuda_index"])

    # logging tensorboard
    log_dir = os.path.join(parameters["checkpoints_dir"], 'log_train')
    writer = SummaryWriter(log_dir=log_dir)
    logger = create_logger(os.path.join(log_dir, 'log.txt'))

    # get datasets
    dataset = HumanAct12Poses(mode='train', **parameters)
    if parameters["num_augment"] > 0:
        logger.info(">>> Argumenting:")
        augmented_trainset = augment_train_dataset(parameters=parameters, logger=logger)

    # get model
    encoder = Encoder(njoints=parameters["num_joints"], nfeats=parameters["coord_dim"], nframes=parameters["num_frames"], gpu_index=parameters["cuda_index"])
    encoder.float()

    # optimizer
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=parameters["lr"], weight_decay=1e-3)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=5, nepoch=parameters["num_epochs"])
    
    for k, v in parameters.items():
        logger.info(f"{k}: {v}")
    logger.info(f">>> Total params: {sum(p.numel() for p in encoder.parameters()) / 1_000_000:.2f}M")
    logger.info(">>> Training encoder..")
    print('>>> Training encoder..')

    # start train
    encoder.to(parameters["device"])
    for i in tqdm(range(parameters["num_epochs"])):
        encoder.train()
        train(i)
        if parameters["snapshot"] > 0 and (i + 1) % parameters["snapshot"] == 0:
            with to_cpu(encoder):
                cp_path = os.path.join(parameters["checkpoints_dir"], f"encoder{i+1:03}.p")
                encoder_cp = {'model_dict': encoder.state_dict()}
                pickle.dump(encoder_cp, open(cp_path, 'wb'))

    writer.close()
    print(f'Checkpoints are saved in: {parameters["checkpoints_dir"]}')
    print('To visualize the training loss in TensorBoard, run the following command:')
    print(f'tensorboard --logdir ./{log_dir}')