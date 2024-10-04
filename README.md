# CtRNet-X: Camera-to-Robot Pose Estimation in Real-world Conditions Using a Single Camera

Paper https://arxiv.org/pdf/2409.10441

Website https://sites.google.com/ucsd.edu/ctrnet-x/home

### Dependencies
Recommend set up the environment using Anaconda.
Code is developed and tested on Ubuntu 22.04.
- Python(3.8)
- Numpy(1.22.4)
- PyTorch(1.10.0)
- torchvision(0.11.1)
- pytorch3d(0.6.2)
- Kornia(0.6.3)
- Transforms3d(0.3.1)
- pyzed (https://www.stereolabs.com/docs/app-development/python/install)

More details see `environment.yml`.

## Usage
- Run `inference_DROID_raw_file.py` to inference the DROID raw data.
- Run `inference_panda_dataset.py` to inference our panda dataset with ground truth camera info.
- Optional args: `confidence_threshold`

## Dataset

1. [DREAM dataset](https://github.com/NVlabs/DREAM/blob/master/data/DOWNLOAD.sh)
2. [Panda arm dataset with ground truth calibration info](https://drive.google.com/drive/folders/14IyXsYZrTJAa1heVOgPjwc87sXVgNrnQ)
2. [DROID sequences for evaluation](https://drive.google.com/file/d/1cuTelwCWbJwsfa4ByhjIxWSyQN5zKq5X/view?usp=drive_link)

## Weights
[Weights for fine-tuned CLIP model](https://drive.google.com/file/d/1C-Zhih6hLM0ctc0Jz-qA5iHPnITAqqdx/view?usp=drive_link)

[Weights for Camera-to-Robot estimation](https://drive.google.com/file/d/1H6nJ-pXfEG4WzRF-tT74ti4mbsB5SjPU/view?usp=drive_link)







