import os
import random
import pickle as pkl
import numpy as np
from src.datasets.dataset import Dataset
from src.datasets.skeleton import Skeleton


class HumanAct12Poses(Dataset):
    dataname = "humanact12"

    def __init__(self, normal_action_name=None, datapath="data/HumanAct12Poses", **kargs):
        '''
        Definition of action-related terminology:
        - label: dataset index of each sample
        - action: integer class ID (e.g., 0, 1, 2, ..., 11)
        - action_name: human-readable action label as a string (e.g., "warm_up", "walk", ..., "throw")
        '''
        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "humanact12poses.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x for x in data["poses"]]
        self._joints = [x for x in data["joints3D"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._actions = [x for x in data["y"]]

        self.skeleton = Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
                                 joints_left=[1, 4, 7, 10, 13, 16, 18, 20, 22],
                                 joints_right=[2, 5, 8, 11, 14, 17, 19, 21, 23])
        self.skeleton._children = [[1, 2, 3], [4], [5], [6], [7], [8], [9], [10], [11], [12, 13, 14], [], [], [15],
                                   [16], [17], [], [18], [19], [20], [21], [22], [23], [], []]
        self.skeleton._has_children = [[True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
                                       [False], [False], [True], [True], [True], [False], [True], [True], [True],
                                       [True], [True], [True], [False], [False]]
        
        self.num_classes = 12
        self._action_names = {
            0: "warm_up",
            1: "walk",
            2: "run",
            3: "jump",
            4: "drink",
            5: "lift_dumbbell",
            6: "sit",
            7: "eat",
            8: "turn steering wheel",
            9: "phone",
            10: "boxing",
            11: "throw",
        }
        self._action_to_label = {x: i for i, x in enumerate(self._actions)}
        self._label_to_action = {i: x for i, x in enumerate(self._actions)}
        
        self._train = []
        self._test = []
        for action_name in self._action_names.values():
            self._labels = [i for i, x in enumerate(self._actions) if self._action_names[x] == action_name]
            split_value = int(len(self._labels) * 0.8)
            self._train.append(self._labels[:split_value])
            self._test.extend(self._labels[split_value:])
        
        if self._mode == 'test':
            random.seed(kargs["seed"])
            self._normal_action_name = normal_action_name
            self._normal_labels = [i for i, x in enumerate(self._actions) if self._action_names[x] == self._normal_action_name]
            split_value = int(len(self._normal_labels) * 0.8)
            self._normal_train = self._normal_labels[:split_value]
            self._support = random.sample(self._normal_train, kargs["num_support"])

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 24, 3)
        return pose