# FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer

#### [website](https://flatformer.mit.edu/) | [paper](https://arxiv.org/abs/2301.08739)

## Introduction

FlatFormer is an efficient point cloud transformer for outdoor 3D object detection, closing the latency gap by trading spatial proximity for better computational regularity. We first flatten the point cloud with window-based sorting and partition points into **groups of equal sizes** rather than **windows of equal shapes**. This effectively avoids expensive structuring and padding over heads. We then apply self-attention within groups to extract local features, alternate sorting axis to gather features from different directions, and shift windows to exchange features across groups. FlatFormer delivers state-of-the-art accuracy on Waymo Open Dataset with **4.6**$\times$ speed up over (transformer-based) [SST](https://arxiv.org/abs/2112.06375) and **1.4**$\times$ speed up over (sparse convolutional) [CenterPoint](https://arxiv.org/abs/2006.11275).

## Usage

### Prerequisites

The code is built with following libraries:

* Python >= 3.6, <3.8
* [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, <= 1.10.2
* [tqdm](https://github.com/tqdm/tqdm)
* [torchpack](https://github.com/mit-han-lab/torchpack)
* [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
* [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.14.0

After installing these dependencies, please run this command to install the codebase:

```
pip install -v -e .
```

### Dataset Preparation

Please follow the instructions from [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/waymo_det.md)  to download and preprocess the Waymo Open Dataset. After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   │   │   ├── training
│   │   │   ├── testing
│   │   │   ├── waymo_gt_database
│   │   │   ├── waymo_infos_trainval.pkl
│   │   │   ├── waymo_infos_train.pkl
│   │   │   ├── waymo_infos_val.pkl
│   │   │   ├── waymo_infos_test.pkl
│   │   │   ├── waymo_dbinfos_train.pkl
```

### Training

We provide instructions to reproduce our results on Waymo.

```
# multi-gpu training
torchpack dist-run -np 8 python tools/train.py ./configs/flatformer/$CONFIG.py --run-dir ./runs/$CONFIG --cfg-options evaluation.pklfile_prefix=./runs/$CONFIG/results evaluation.metric=waymo
```

## Citation
If FlatFormer is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```
@inproceedings{liu2023flatformer,
  title={FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer},
  author={Liu, Zhijian and Yang, Xinyu and Tang, Haotian and Yang, Shang and Han, Song},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## Acknowledgments
This project is based on the following codebases.  

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [SST](https://github.com/TuSimple/SST)

We would like to thank Tianwei Yin, Lue Fan and Ligeng Mao for providing detailed results of [CenterPoint](https://arxiv.org/abs/2006.11275), [SST](https://arxiv.org/abs/2112.06375)/[FSD](https://arxiv.org/abs/2207.10035) and [VoTr](https://arxiv.org/abs/2109.02497), and Yue Wang and Yukang Chen for their helpful discussions. This work was supported by National Science Foundation, MIT-IBM Watson AI Lab, NVIDIA, Hyundai and Ford. Zhijian Liu was partially supported by the Qualcomm Innovation Fellowship.
