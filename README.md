# Few-shotHAAD
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18147989.svg)](https://doi.org/10.5281/zenodo.18147989)

Official implementation of [**"Few-shot human action anomaly detection via a unified contrastive learning framework"**](https://www.sciencedirect.com/science/article/pii/S0950705125021689), [Knowledge-Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems).

Overview of the proposed framework:
![Framework](figure/framework.png)


## :construction_worker: Installation
### :wrench: 1. Create conda environment
From the repository root, run:
```bash
bash prepare/create_env.sh
conda activate haad
```
The code was tested on **Python 3.8** and **PyTorch 1.7.1**.

### :package: 2. Download the HumanAct12 dataset
**Important:** Please follow the original dataset license/terms and cite the dataset accordingly.

Please download the post-processed HumanAct12 file by following the instructions in the ACTOR project:
- :link: [ACTOR dataset page](https://github.com/Mathux/ACTOR/blob/master/DATASETS.md)

After downloading, extract the archive and place the extracted file at:
```text
data/HumanAct12Poses/humanact12poses.pkl
```

### :file_folder: 3. Download the checkpoints
From the repository root, run:
```bash
bash prepare/download_checkpoint.sh
```
This script downloads the checkpoint archive from Google Drive and places the files as:
```text
checkpoints_paper/
├── encoder.p
└── humanmac.pt
```


## :rocket: Run Experiments
### :fire: Training
From the repository root, run:
```bash
bash run_train.sh
```
### :clipboard: Evaluation
From the repository root, run:
```bash
bash run_test.sh
```


## :pray: Acknowledgements
This work was supported by JSPS KAKENHI Grant Number 23K10712.

We thank the authors of the datasets and open-source projects used in this repository and our experiments, including:
- **[HumanAct12](https://ericguo5513.github.io/action-to-motion/):** We used HumanAct12 for evaluation and benchmarking.
- **[ACTOR](https://github.com/Mathux/ACTOR):** We refer to ACTOR for dataset preparation instructions and related resources.
- **[HumanMAC](https://github.com/LinghaoChan/HumanMAC):** This repository includes code adapted from HumanMAC (MIT License).
- **[PyTorch3D](https://github.com/facebookresearch/pytorch3d):** This repository includes code adapted from PyTorch3D (BSD 3-Clause).


## :scroll: License
This code is distributed under the [MIT License](LICENSE).

This repository uses third-party libraries and datasets (e.g., PyTorch3D, HumanMAC, HumanAct12), each of which has its own license/terms that must also be followed.

See [`NOTICE.md`](NOTICE.md) for third-party license notices.


## :handshake: Citation
If you find this code helpful in your research, please consider citing our paper:
```text
@article{kamide2026few,
  title   = {Few-shot human action anomaly detection via a unified contrastive learning framework},
  author  = {Kamide, Koichiro and Sakai, Shunsuke and Maeda, Shun and Gu, Chunzhi and Zhang, Chao},
  journal = {Knowledge-Based Systems},
  volume  = {334},
  pages   = {115133},
  year    = {2026},
  doi     = {10.1016/j.knosys.2025.115133}
}
```