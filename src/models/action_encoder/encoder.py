import torch
import torch.nn as nn
from . import gcn
from src.utils import util


class Encoder(nn.Module):
    def __init__(self, njoints=24, nfeats=3, nframes=60, hidden_dim=128, gpu_index=0):
        """
        Args:
            njoints (int): Number of joints in the graph.
            nfeats (int): Coordinate dimension per joint.
            nframes (int): Number of frames.
            hidden_dim (int): Dimension of hidden features per joint.
        """
        super().__init__()
        # GPU
        device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_index)
     
        # DCT basis matrix
        self.DCT_base = 10
        dct_m, _ = util.get_dct_matrix(nframes)
        self.dct_m_all = dct_m.float().to(device)

        # GCN
        self.gcn = gcn.GCNParts(input_feature=self.DCT_base * nfeats, 
                                hidden_feature=hidden_dim, is_bn=True,
                                num_block=4,
                                node_n=njoints)

    def forward(self, batch_motion):
        """
        Forward pass to obtain motion embeddings.

        Args:
            batch_motion (Tensor): Input skeletal motion sequence
                shape: (B, H, J, 3),
                where B is batch size, H is the number of frames,
                J is the number of joints, and 3 is the 3D coordinate dimension (xyz).

        Returns:
            embeddings (Tensor): Motion embedding
                shape: (B, J)

        Notes:
            M: number of DCT frequency components (DCT bases)
            F: feature dimension of GCN output
        """
        B, H, J, C = batch_motion.shape

        # Apply DCT: temporal motion -> DCT coefficient matrix
        motion_flat = batch_motion.reshape(B, H, -1)           # (B, H, 3J)
        dct_coeff = torch.matmul(
            self.dct_m_all[:self.DCT_base], motion_flat)       # (B, M, 3J)

        # Apply GCN: DCT coefficients -> joint-wise feature matrix
        dct_coeff = dct_coeff.reshape(B, self.DCT_base, J, C)  # (B, M, J, 3)
        dct_coeff = dct_coeff.permute(0, 2, 1, 3)              # (B, J, M, 3)
        dct_coeff = dct_coeff.reshape(B, J, -1)                # (B, J, 3M)
        joint_feat = self.gcn(dct_coeff)                       # (B, J, F)

        # Apply max pooling to obtain motion embedding
        embeddings = torch.max(joint_feat, dim=2)[0]           # (B, J)

        return embeddings


if __name__ == '__main__':
    bs = 32
    node_n = 48
    data_dim = 25
    hidden_dim = 128
    con_dim = 100
    num_flow_layer = 10
    num_ds_layer = 6

    sf = Encoder(data_dim=data_dim)
    # print(torch.mean(sf.prior.log_prob(a).sum([1, 2])))
    sf.double()
    sf.cuda()