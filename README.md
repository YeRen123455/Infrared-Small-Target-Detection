# Dense Nested Attention Network for Infrared Small Target Detection

![outline](overall_structure.jpg)
## Algorithm Introduction

The code and trained models of:

Dense Nested Attention Network for Infrared Small Target Detection, Boyang Li, Chao Xiao, Longguang Wang, and Yingqian Wang, arxiv 2021 [[Paper]](https://arxiv.org/pdf/2106.00487.pdf)

We propose a dense nested attention network (DNANet) to achieve accurate single-frame infrared small target detection and develop an open-sourced infrared small target dataset (namely, NUDT-SIRST) in this paper. Experiments on both public (e.g., NUAA-SIRST, NUST-SIRST) and our self-developed datasets demonstrate the effectiveness of our method.

## Dataset Introduction
NUDT-SIRST is a synthesized dataset, which contains 1327 images with resolution of 256x256. The advantages of synthesized dataset compared to real dataset lies in three aspesets
1. Accurate annotations
2. Massive generation with low cost (i.e., time and money)
3. Numerous categories of target, rich target sizes, diverse clutter backgrounds.

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@article{li2021dense,
  title={Dense Nested Attention Network for Infrared Small Target Detection},
  author={Li, Boyang and Xiao, Chao and Wang, Longguang and Wang, Yingqian and Lin, Zaiping and Li, Miao and An, Wei and Guo, Yulan},
  journal={arXiv preprint arXiv:2106.00487},
  year={2021}
}
```

## Prerequisite
* Tested on Ubuntu 16.04, with Python 3.7, PyTorch 1.7, Torchvision 0.8.1, CUDA 11.1, and 1x NVIDIA 3090 and also 
* Tested on Windows 10  , with Python 3.6, PyTorch 1.1, Torchvision 0.3.0, CUDA 10.0, and 1x NVIDIA 1080Ti.
* [The NUDT-SIRST download dir](https://drive.google.com/drive/folders/1YGoYaBi9dLwoTwoeTytEs5m-VeeCDXf7?usp=sharing) (coming soon)
* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst)
* [The NUST-SIRST download dir](https://github.com/wanghuanphd/MDvsFA_cGAN) 

## Usage


#### On windows:
```
Click train.py and run it. (All parameters are s)
```

#### On Ubuntu:
#### 1. Train.

```bash
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset [dataset-name] --split_method 50_50 --model [model name] --backbone resnet_18  --deep_supervision True --train_batch_size 16 --test_batch_size 16 --mode TXT
```

#### 2. Test.

```bash
python test.py --base_size 256 --crop_size 256 --st_model [trained model path] --model_dir [model_dir] --dataset [dataset-name] --split_method 50_50 --model [model name] --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT 
```

#### (Optional) Visulize your predicts.

```bash
python visulization.py --base_size 256 --crop_size 256 --st_model [trained model path] --model_dir [model_dir] --dataset [dataset-name] --split_method 50_50 --model [model name] --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT 
```

#### (Optiona2) Test and visulization.
```bash
python test_and_visulization.py --base_size 256 --crop_size 256 --st_model [trained model path] --model_dir [model_dir] --dataset [dataset-name] --split_method 50_50 --model [model name] --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT 
```

#### (Optiona3) Demo (with your own IR image).
```bash
python demo.py --base_size 256 --crop_size 256 --img_demo_dir [img_demo_dir] --img_demo_index [image_name]  --model [model name] --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT  --suffix [img_suffix]
```




## Results and Trained Models

#### Qualitative Results

![outline](overall_structure.jpg)


#### Class Activation Map

| Model         | Train (mIoU)    | Val (mIoU)    | |
| ------------- |:-------------:|:-----:|:-----:|
| VGG-16        | 48.9 | 46.6 | [[Weights]](https://drive.google.com/file/d/1Dh5EniRN7FSVaYxSmcwvPq_6AIg-P8EH/view?usp=sharing) |
| ResNet-38     | 47.7 | 47.2 | [[Weights]](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view?usp=sharing) |
| ResNet-38     | 48.0 | 46.8 | CVPR submission |

#### Random Walk with AffinityNet

| Model         | alpha | Train (mIoU)    | Val (mIoU)    | |
| ------------- |:-----:|:---------------:|:-------------:|:-----:|
| VGG-16        | 4/16/32 | 59.6 | 54.0 | [[Weights]](https://drive.google.com/file/d/10ue1B20Q51aQ53T93RiaiKETlklzo4jp/view?usp=sharing) |
| ResNet-38     | 4/16/32 | 61.0 | 60.2 | [[Weights]](https://drive.google.com/open?id=1mFvTH3siw0SS0vqPH0o9N3cI_ISQacwt) |
| ResNet-38     | 4/16/24 | 58.1 | 57.0 | CVPR submission |

>*beta=8, gamma=5, t=256 for all settings
