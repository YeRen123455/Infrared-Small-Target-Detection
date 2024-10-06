# Dense Nested Attention Network for Infrared Small Target Detection

## Good News! Our paper has been accepted by `IEEE Transaction on Image Processing`. Our team will release more interesting works and applications on SIRST soon. Please keep following our repository.

![outline](overall_structure.png)

## Algorithm Introduction

Dense Nested Attention Network for Infrared Small Target Detection, Boyang Li, Chao Xiao, Longguang Wang, and Yingqian Wang, arxiv 2021 [[Paper]](https://arxiv.org/pdf/2106.00487.pdf)

We propose a dense nested attention network (DNANet) to achieve accurate single-frame infrared small target detection and develop an open-sourced infrared small target dataset (namely, NUDT-SIRST) in this paper. Experiments on both public (e.g., NUAA-SIRST, NUST-SIRST) and our self-developed datasets demonstrate the effectiveness of our method. The contribution of this paper are as follows:

1. We propose a dense nested attention network (namely, DNANet) to maintain small targets in deep layers.

2. An open-sourced dataset (i.e., NUDT-SIRST) with rich targets.

3. Performing well on all existing SIRST datasets.

## Dataset Introduction

NUDT-SIRST is a synthesized dataset, which contains 1327 images with resolution of 256x256. The advantage of synthesized dataset compared to real dataset lies in three aspets:

1. Accurate annotations.

2. Massive generation with low cost (i.e., time and money).

3. Numerous categories of target, rich target sizes, diverse clutter backgrounds.

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@inproceedings{li2023monte,
  title={Monte Carlo linear clustering with single-point supervision is enough for infrared small target detection},
  author={Li, Boyang and Wang, Yingqian and Wang, Longguang and Zhang, Fei and Liu, Ting and Lin, Zaiping and An, Wei and Guo, Yulan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1009--1019},
  year={2023}
}
```

## Prerequisite
* Tested on Ubuntu 16.04, with Python 3.7, PyTorch 1.7, Torchvision 0.8.1, CUDA 11.1, and 1x NVIDIA 3090 and also 

* Tested on Windows 10  , with Python 3.6, PyTorch 1.1, Torchvision 0.3.0, CUDA 10.0, and 1x NVIDIA 1080Ti.

* [The NUDT-SIRST download dir](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt) (Extraction Code: nudt)

* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst) [[ACM]](https://arxiv.org/pdf/2009.14530.pdf)

* [The NUST-SIRST download dir](https://github.com/wanghuanphd/MDvsFA_cGAN) [[MDvsFA]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)

## Usage

#### On windows:

```
Click on train.py and run it. 
```

#### On Ubuntu:

#### 1. Preprocessing.

```bash
python generate_single_point_Prior.py

```
#### 2. Pseudo Label Generation.

```bash
python Baseline_NUAA.py
```

#### 3. Retrain SIRST Detection Network (e.g, DNANet) with the Generated Label.





