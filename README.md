# Dense Nested Attention Network for Infrared Small Target Detection

![outline](overall_structure.jpg)
## Introduction

The code and trained models of:

Dense Nested Attention Network for Infrared Small Target Detection, Boyang Li, Chao Xiao and Longguang Wang, arxiv 2021 [[Paper]](https://arxiv.org/pdf/2106.00487.pdf)

We propose a dense nested attention network (DNANet) to achieve accurate single-frame infrared small target detection and develop an open-sourced infrared small target dataset (namely, NUDT-SIRST) in this paper. Experiments on both public (e.g., NUAA-SIRST, NUST-SIRST) and our self-developed datasets demonstrate the effectiveness of our method.


>*Our code is implemented in pytorch. The default setting is pytorch==1.1.0, python==3.6, CUDA==10.1. This code also work well on device with pytorch==1.7.0, python==3.7, CUDA==11.1.

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
* [The NUST-SIRST download dir](https://github.com/wanghuanphd/MDvsFA_cGAN) (coming soon)

## Usage
#### 1. Train a classification network to get CAMs.

```bash
python3 train_cls.py --lr 0.1 --batch_size 16 --max_epoches 15 --crop_size 448 --network [network.vgg16_cls | network.resnet38_cls] --voc12_root [your_voc12_root_folder] --weights [your_weights_file] --wt_dec 5e-4
```

#### 2. Generate labels for AffinityNet by applying dCRF on CAMs.

```bash
python3 infer_cls.py --infer_list voc12/train_aug.txt --voc12_root [your_voc12_root_folder] --network [network.vgg16_cls | network.resnet38_cls] --weights [your_weights_file] --out_cam [desired_folder] --out_la_crf [desired_folder] --out_ha_crf [desired_folder]
```


#### (Optional) Check the accuracy of CAMs.
```bash
python3 infer_cls.py --infer_list voc12/val.txt --voc12_root [your_voc12_root_folder] --network network.resnet38_cls --weights res38_cls.pth --out_cam_pred [desired_folder]
```


#### 3. Train AffinityNet with the labels

```bash
python3 train_aff.py --lr 0.1 --batch_size 8 --max_epoches 8 --crop_size 448 --voc12_root [your_voc12_root_folder] --network [network.vgg16_aff | network.resnet38_aff] --weights [your_weights_file] --wt_dec 5e-4 --la_crf_dir [your_output_folder] --ha_crf_dir [your_output_folder]
```

#### 4. Perform Random Walks on CAMs

```bash
python3 infer_aff.py --infer_list [voc12/val.txt | voc12/train.txt] --voc12_root [your_voc12_root_folder] --network [network.vgg16_aff | network.resnet38_aff] --weights [your_weights_file] --cam_dir [your_output_folder] --out_rw [desired_folder]
```

## Results and Trained Models
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
