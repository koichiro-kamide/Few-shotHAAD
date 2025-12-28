# [Few-shot human action anomaly detection via a unified contrastive learning framework](https://www.sciencedirect.com/science/article/pii/S0950705125021689)

## Installation :construction_worker:
### 1. Create conda environment
```bash
conda create -n haad python=3.8
conda activate haad
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```
The code was tested on **Python 3.8** and **PyTorch 1.7.1**.


### 2. Download the dataset
**Important:** For the HumanAct12 dataset, please read and follow their license agreements and cite them accordingly.

For reproducibility, we use the **post-processed HumanAct12 file** provided by the ACTOR project:
- humanact12poses.pkl  
We do not redistribute the dataset in this repository. Please download it from the ACTOR dataset page and place it as follows:
- ACTOR dataset page: https://github.com/Mathux/ACTOR/blob/master/DATASETS.md
- Place the downloaded file here:
```
data/humanact12poses.pkl
```

### 3. Download the checkpoints
Please download from:
- https://drive.google.com/file/d/1n2iBoivKGJpyDGSs4RDMrLDNU88cGLSA/view?usp=sharing

Steps:
1. Unzip `checkpoints_paper.zip`
2. Move the extracted folder to `checkpoints_paper/`

Expected directory structure:
```text
checkpoints_paper/
├── encoder.p
└── humanmac.pt
```

## Expected project structure
```text
Few-ShotHAAD/
├── checkpoints_paper/
│   ├── encoder.p
│   └── humanmac.pt
└── data/
    └── humanact12poses.pkl
```


## How to use HAAD

### HumanAct12
#### Training
use run.sh file.

### UESTC
#### Training
use run_uestc.sh file.

### Evaluation
```bash
python -m src.evaluate.evaluate_cvae PATH/TO/checkpoint_XXXX.pth.tar --batch_size 64 --niter 20
```
This script will evaluate the trained model, on the epoch ``XXXX``, with 20 different seeds, and put all the results in ``PATH/TO/evaluation_metrics_XXXX_all.yaml``.

If you want to get a table with mean and interval, you can use this script:

```bash
python -m src.evaluate.tables.easy_table PATH/TO/evaluation_metrics_XXXX_all.yaml
```

### Visualization
#### Grid of stick figures
```bash
 python -m src.visualize.visualize_checkpoint PATH/TO/CHECKPOINT.tar --num_actions_to_sample 5  --num_samples_per_action 5
```


### Generating and rendering SMPL meshes
#### Additional dependencies
``` bash
pip install trimesh
pip install pyrender
pip install imageio-ffmpeg
```

#### Generate motions
```bash
python -m src.generate.generate_sequences PATH/TO/CHECKPOINT.tar --num_samples_per_action 10 --cpu
```

It will generate 10 samples per action, and store them in ``PATH/TO/generation.npy``.

#### Render motions
``` bash
python -m src.render.rendermotion PATH/TO/generation.npy
```

It will render the sequences into this folder ``PATH/TO/generation/``.


## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.
