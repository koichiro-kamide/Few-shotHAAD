import copy
import torch
import numpy as np


def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m


def generate_pad(t_his, t_pred):
    # [0, 1, 2,....,29, 29,....,29]
    idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
    return idx_pad


def padding_traj(traj, idx_pad):
    traj_pad = traj[..., idx_pad, :]
    return traj_pad


def sample_preprocessing(traj, parameters, num_generate=2, dct_m_all=None):
    """
    This function is used to preprocess traj for sample_ddim().

    Args:
        traj (Tensor):
            Trajectory of shape (1, H, 3J),
            where H = number of frames and J = number of joints.
        parameters (dict):
            t_his  : number of observed frames
            t_pred : number of frames to predict
            n_pred  : number of DCT bases used for prediction
        num_generate (int):
            Generation count.
        dct_m_all (Tensor):
            Predefined DCT matrix.

    Returns:
        meta (dict):
            mask       : binary mask (1 = observed, 0 = predicted)
            sample_num : number of samples to generate
        traj_dct (Tensor):
            DCT coefficients of the padded trajectory. 
            shape: (1, M, 3J), where M = number of DCT basis.
        traj_dct_mod (Tensor or None):
            Optional modified DCT coefficients.
            shape: (1, M, 3J)
    """
    traj = traj.repeat(num_generate, 1, 1)

    mask = torch.ones([num_generate, parameters['num_frames'], traj.shape[-1]]).to(parameters['device'])
    for i in range(parameters['t_his'], parameters['t_his']+parameters['t_pred']):
        mask[:, i, :] = 0

    idx_pad = generate_pad(parameters['t_his'], parameters['t_pred'])
    traj_pad = padding_traj(traj, idx_pad)

    traj_dct = torch.matmul(dct_m_all[:parameters['n_pred']], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)

    return {'mask': mask,
            'sample_num': num_generate}, traj_dct, traj_dct_mod


def post_process(pred):
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
    return pred
