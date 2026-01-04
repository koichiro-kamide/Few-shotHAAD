import random
import numpy as np
import torch
from src.utils.misc import to_torch
import src.utils.rotation_conversions as geometry

POSE_REPS = ["xyz", "rotvec", "rotmat", "rotquat", "rot6d"]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=60, sampling_step=1, mode="train",
                 pose_rep="xyz", translation=True, glob=True, **kwargs):
        self.num_frames = num_frames
        self.sampling_step = sampling_step
        self._mode = mode
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self._train = []
        self._test = []

        if self._mode not in ["train", "test"]:
            raise ValueError(f"{self._mode} is not a valid run_mode")
        super().__init__()

    def update_parameters(self, parameters):
        # add specific parameters from the dataset loading
        self.njoints, self.nfeats, _ = self[0][0].shape
        parameters["coord_dim"] = self.nfeats
        parameters["num_joints"] = self.njoints

    def get_action(self, data_index):
        return self._actions[data_index]

    def action_to_label(self, action):
        return self._action_to_label[action]

    def get_label(self, data_index):
        action = self.get_action(data_index)
        return self.action_to_label(action)

    def label_to_action(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            return self._label_to_action[label]
        else:  # if it is one hot vector
            label = np.argmax(label)
            return self._label_to_action[label]

    def action_to_action_name(self, action):
        return self._action_names[action]

    def label_to_action_name(self, label):
        action = self.label_to_action(label)
        return self.action_to_action_name(action)

    def _get_item_data_index(self, data_index):
        nframes = self._num_frames_in_video[data_index]
        if self.num_frames > nframes:
            # padding the last frame until done
            frame_ix = np.arange(0, self.num_frames)
            frame_ix[nframes:] = nframes - 1
        else:
            # extracting a single continuous clip from a long video, starting from a random offset
            step_max = (nframes - 1) // (self.num_frames - 1)
            if self.sampling_step * (self.num_frames - 1) >= nframes:
                step = step_max
            else:
                step = self.sampling_step
            lastone = step * (self.num_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_ix = shift + np.arange(0, lastone + 1, step)

        pose = self.get_pose_data(data_index=data_index, frame_ix=frame_ix)
        label = self.get_label(data_index=data_index)
        return pose, label
    
    def __getitem__(self, index):
        if self._mode == 'train':
            data_index = self._train[index][index]
        else:
            data_index = self._test[index]

        inp, target = self._get_item_data_index(data_index)
        return inp, target

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"

    def get_pose_data(self, data_index, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(data_index, frame_ix)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
        if pose_rep != "xyz":
            if getattr(self, "_load_rotvec", None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_rotvec(data_index, frame_ix)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if pose_rep == "rotvec":
                    ret = pose
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype, device='cpu')
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def seq_sampler(self, data_index):
        pose_data = self._pose[data_index]
        pose_frames = pose_data.shape[0]
        frame_ix = np.arange(0, self.num_frames)
        if pose_frames <= self.num_frames:
            frame_ix[pose_frames:] = pose_frames - 1
        seq = self.get_pose_data(data_index=data_index, frame_ix=frame_ix)
        seq = seq.permute(2, 0, 1).contiguous()
        return seq

    def train_dataloader(self, num_samples=1000, batch_size=18):
        labels = list(range(int(self.num_classes*0.8))) + list(range(int(self.num_classes*0.8)))
        # lebels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(num_samples // batch_size):
            samp = []
            indices = []
            for label in labels:
                idx = np.random.randint(0, len(self._train[label]))
                sample_idx = self._train[label][idx]
                seq = self.seq_sampler(data_index=sample_idx)
                indices.append(sample_idx)
                # check for duplicate pairs
                while len(indices) != len(set(indices)):
                    indices.pop()
                    idx = np.random.randint(0, len(self._train[label]))
                    sample_idx = self._train[label][idx]
                    seq = self.seq_sampler(data_index=sample_idx)
                    indices.append(sample_idx)
                samp.append(seq.unsqueeze(dim=0))
            samp = torch.cat(samp, dim=0)
            yield samp

    def augmented_train_dataloader(self, num_samples=1000, batch_size=18, aug_trainset=None):
        labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
        for i in range(num_samples // batch_size):
            samp = []
            indices = []
            for label in labels:
                sample_idx = np.random.randint(0, len(aug_trainset[label]))
                seq = aug_trainset[label][sample_idx]  # Tensor of shape (1 + num_augment, H, J, C)
                indices.append(sample_idx)
                # check for duplicate pairs
                while len(indices) != len(set(indices)):
                    indices.pop()
                    sample_idx = np.random.randint(0, len(aug_trainset[label]))
                    seq = aug_trainset[label][sample_idx]
                    indices.append(sample_idx)
                samp.append(seq)
            samp = torch.cat(samp, dim=0)
            yield samp

    def test_dataloader(self):
        samp = []
        for i in self._test:
            seq = self.seq_sampler(data_index=i)
            samp.append(seq.unsqueeze(dim=0))
        del i
        samp = torch.cat(samp, dim=0)
        samp = samp.to(torch.float32)
        y_labels = [0 if self.action_to_action_name(self._actions[data_index]) == self._normal_action_name else 1 for data_index in self._test]
        y_labels = torch.tensor(y_labels, dtype=torch.float32)
        yield samp, y_labels

    def support_dataloader(self):
        samp = []
        print(f'suport set:{self._support}')
        for i in self._support:
            seq = self.seq_sampler(data_index=i)
            samp.append(seq.unsqueeze(dim=0))
        samp = torch.cat(samp, dim=0)
        samp = samp.to(torch.float32)
        yield samp