# [CtRNet-X: Camera-to-Robot Pose Estimation in Real-world Conditions Using a Single Camera](https://sites.google.com/ucsd.edu/ctrnet-x/home)

Jingpei Lu*, Zekai Liang*, Tristin Xie, Florian Ritcher, Shan Lin, Sainan Liu, Michael C. Yip

University of California, San Diego

ICRA 2025

[[arXiv]](https://arxiv.org/pdf/2409.10441) [[Project page]](https://sites.google.com/ucsd.edu/ctrnet-x/home)


## Highlight
CtRNet-X is a novel framework capable of estimating the robot pose with partially visible robot manipulators. Our approach leverages the Vision-Language Models for fine-grained robot components detection, and integrates it into a keypoint-based pose estimation network, which enables more robust performance in varied operational conditions. 
![demo](assets/demo_3.gif)


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


## Dataset

1. [DREAM dataset](https://github.com/NVlabs/DREAM/blob/master/data/DOWNLOAD.sh)
2. [Panda arm dataset with ground truth calibration info](https://drive.google.com/drive/folders/14IyXsYZrTJAa1heVOgPjwc87sXVgNrnQ)
2. [DROID sequences for evaluation](https://drive.google.com/file/d/1cuTelwCWbJwsfa4ByhjIxWSyQN5zKq5X/view?usp=drive_link)

## Weights
[Weights for fine-tuned CLIP model](https://drive.google.com/file/d/1C-Zhih6hLM0ctc0Jz-qA5iHPnITAqqdx/view?usp=drive_link)

[Weights for Camera-to-Robot estimation](https://drive.google.com/file/d/1H6nJ-pXfEG4WzRF-tT74ti4mbsB5SjPU/view?usp=drive_link)


## Quick Start

**Inference DROID raw data:**
```python
python inference_DROID_raw_file.py
```

**Inference panda dataset with ground truth camera info:**
```python
python inference_panda_dataset.py
```

- Optional args: `confidence_threshold`

## Working with RGBD input

A variation of CtRNet-X can integrate depth maps from an RGB-D camera during inference by comparing measured depth to rendered depth from the differentiable renderer. Here we use DROID as example.

 ![raw](assets/raw_depth(1).gif)  ![render](assets/depth_rendering(1).gif)  

 **Use depth input to refine estimation:**
```python
python inference_video_depth.py
``` 

We use Huber loss with delta 0.1, feel free to try your own depth data with different losses!

## Citation
```bibtex
@article{lu2024ctrnet,
  title={CtRNet-X: Camera-to-Robot Pose Estimation in Real-world Conditions Using a Single Camera},
  author={Lu, Jingpei and Liang, Zekai and Xie, Tristin and Ritcher, Florian and Lin, Shan and Liu, Sainan and Yip, Michael C},
  journal={arXiv preprint arXiv:2409.10441},
  year={2024}
}
```



