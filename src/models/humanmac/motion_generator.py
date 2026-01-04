import torch
import numpy as np
from src.models.humanmac.transformer import MotionTransformer
from src.models.humanmac.diffusion import Diffusion
from src.utils.util import get_dct_matrix, sample_preprocessing, post_process


def get_model_and_diffusion(parameters, dct_m_all, idct_m_all):
    model = MotionTransformer(
        input_feats=3 * parameters["num_joints"],  # 3 means 3d coordinate
        num_frames=parameters["n_pred"],
        num_layers=8,
        num_heads=8,
        latent_dim=512,
        dropout=0.2,
    ).to(parameters["device"])

    diffusion = Diffusion(
        noise_steps=1000,
        motion_size=(parameters["n_pred"], 3 * parameters["num_joints"]),  # 3 means 3d coordinate
        device=parameters["device"],
        EnableComplete=True,
        ddim_timesteps=100,
        scheduler='Cosine',
        mod_test=1.0,
        dct=dct_m_all,
        idct=idct_m_all,
        n_pred=parameters["n_pred"]
    )

    ckpt = torch.load(f'./checkpoints_paper/humanmac.pt', map_location='cpu')
    model.load_state_dict(ckpt)
    model = model.to(parameters["device"])

    return model, diffusion


def motion_generator(sample, parameters, num_generate=2):
    """
    Generate multiple future motion sequences from a single observed pose sequence
    using a diffusion-based model in the DCT domain.

    Args:
        sample (Tensor):
            Input pose sequence of shape (H, J, 3).
        parameters (dict):
            num_frames : total sequence length
            n_pred      : number of DCT bases used for prediction
        num_generate (int):
            Generation count.

    Returns:
        generated_motions (Tensor):
            Generated motion sequences of shape (G, H, J, 3),
            where G = num_generate.
    """
    traj_np = None

    # get DCT and IDCT matrix
    dct_m, idct_m = get_dct_matrix(parameters["num_frames"])
    dct_m_all = dct_m.float().to(parameters["device"])
    idct_m_all = idct_m.float().to(parameters["device"])

    # get diffusion-based generative model
    model, diffusion = get_model_and_diffusion(parameters, dct_m_all, idct_m_all)

    # get DCT coefficient matrix from padded sequences
    sample = sample.unsqueeze(0).to('cpu').detach().numpy()  # (1, H, J, 3)
    gt = sample[0].copy()  # (H, J, 3)
    gt = np.expand_dims(gt, axis=0)  # (1, H, J, 3)
    traj_np = gt.reshape([gt.shape[0], 60, -1])  # (1, H, 3J)
    traj = torch.tensor(traj_np, device=parameters["device"], dtype=torch.float32)
    mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, 
                                                             parameters, 
                                                             num_generate=num_generate, 
                                                             dct_m_all=dct_m_all)  # (1, M, 3J)
    
    # generate motion sequences in frequency space
    generated_motions_dct = diffusion.sample_ddim(model,
                                                traj_dct,
                                                traj_dct_mod,
                                                mode_dict)  # (G, M, 3J)
    
    # transform to temporal sequences by IDCT
    generated_motions = torch.matmul(idct_m_all[:, :parameters["n_pred"]], generated_motions_dct)  # (G, H, 3J)
    generated_motions = post_process(generated_motions)  # (G, H, J, 3)

    return generated_motions