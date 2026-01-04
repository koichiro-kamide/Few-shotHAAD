import time
from datetime import datetime
import torch
from tqdm import tqdm

import numpy as np
import pickle as pkl

from src.parser.training import parser
from src.utils.torch import *
from src.utils.logger import *
from src.models.humanmac.motion_generator import motion_generator


def augment_train_dataset(parameters, logger):
    """
    Prepare the augmented training dataset.

    - Temporal normalization to fixed length H=60
    - Spatial normalization using pelvis-centered coordinates
    - Category-wise train split (80%)
    - Motion augmentation per sample

    Returns:
        augmented_trainset:
            List of categories, each containing a list of augmented motion tensors 
            with shape (1 + num_augment, H, J, C)
    """
    # === Load dataset ===
    data_file = "./data/humanact12poses.pkl"
    dataset = pkl.load(open(data_file, "rb"))

    # === Preprocess motion data ===
    motion_set = [joint for joint in dataset["joints3D"]]  # (num_samples, H, J, C)
    for index, motion in enumerate(motion_set):
        # Temporal normalization: ensure fixed length of 60 frames
        fn = motion.shape[0]
        if fn <= 60:
            # Pad by repeating the last frame
            frame_ix = np.arange(0, 60)
            frame_ix[fn:] = fn - 1
        else:
            # Randomly sample a continuous 60-frame sequence
            start_idx = np.random.randint(0, fn - 60 + 1)
            frame_ix = np.arange(start_idx, start_idx + 60)
        motion = motion[frame_ix]
        # Spatial normalization: translate motion so that the pelvis (joint 0 at the first frame) is at the origin
        motion = motion - motion[0, 0, :]
        motion_set[index] = torch.tensor(motion).to(parameters["device"]).float()

    # === Split training samples per category (80%) ===
    train_cat_indexes = []
    labels = [label for label in dataset["y"]]
    for cat in range(9):  # ("warm_up" "walk" "run" "jump" "drink" "lift_dumbbell" "sit" "eat" "turn steering wheel")
        cat_indexes = [index for index, label in enumerate(labels) if label == cat]
        split_value = int(len(cat_indexes) * 0.8)
        train_cat_indexes.append(cat_indexes[:split_value])

    # === Data augmentation for training ===
    print(">>> Augmenting train data..")
    augmented_trainset = []
    for cat_indexes in tqdm(train_cat_indexes):
        cat_samples = []
        for index in cat_indexes:
            t_s = time.time()
            # Original motion sample: (H, J, C)
            sample = motion_set[index].to(parameters["device"]).float()
            # Generate augmented samples: (num_augment, H, J, C)
            aug_samples = motion_generator(sample=sample, parameters=parameters, num_generate=parameters["num_augment"])
            # Combine original and augmented samples: (1 + num_augment, H, J, C)
            samples = torch.cat([sample.unsqueeze(0), aug_samples], dim=0)
            cat_samples.append(samples)
            logger.info(f"Sample {index:>4} finished: Time={time.time() - t_s:.3f} [s]")
        # Each category contains a list of augmented samples: (cat_num, num_samples, (1 + num_augment), H, J, C)
        augmented_trainset.append(cat_samples)

    # torch.save({'data': augmented_trainset}, f'augmented_trainset{parameters["num_augment"]}.pt')
    return augmented_trainset


if __name__ == '__main__':
    # parse options
    parameters = parser()
    # set logger
    logger = create_logger(os.path.join('log_augment', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'))
    # init random
    np.random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    # torch.autograd.set_detect_anomaly(True)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if torch.cuda.is_available():
        torch.cuda.set_device(parameters["cuda_index"])

    print(f'num_augment={parameters["num_augment"]}')
    augment_train_dataset(parameters=parameters, logger=logger)
    
    # loaded_data = torch.load(f'augmented_trainset{parameters["num_augment"]}.pt')
    # augmented_trainset = loaded_data['data']
    # cat0_sample0 = augmented_trainset[0][0]  # Tensor of shape (1 + num_augment, H, J, C)
    # print(f'cat0_sample0: {cat0_sample0.shape}')