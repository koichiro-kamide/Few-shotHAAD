# [Few-shot human action anomaly detection via a unified contrastive learning framework](https://www.sciencedirect.com/science/article/pii/S0950705125021689)

## Installation :construction_worker:
### 1. Create conda environment
From the repository root, run:
```bash
bash prepare/create_env.sh
conda activate haad
```

### 2. Download the dataset
**Important:** Please follow the original dataset license/terms and cite the dataset accordingly.

Please download **post-processed HumanAct12 file** by following the instructions in the ACTOR project:
- [ACTOR dataset page](https://github.com/Mathux/ACTOR/blob/master/DATASETS.md)

Then, place the downloaded file here:
```
<repository_root>/data/HumanAct12Poses/humanact12poses.pkl
```

### 3. Download the checkpoints
From the repository root, run:
```bash
bash prepare/download_checkpoint.sh
```
This script downloads the checkpoint archive from Google Drive and places the files as:
```text
<repository_root>/
└── checkpoints_paper/
    ├── encoder.p
    └── humanmac.pt
```


## Run Experiments
### Training
From the repository root, run:
```bash
bash run_train.sh
```
### Evaluation
From the repository root, run:
```bash
bash run_test.sh
```


## Acknowledgements


## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.