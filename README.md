# FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer

#### [website](https://flatformer.mit.edu/) | [paper](https://arxiv.org/abs/2301.08739)

## Abstract

Transformer, as an alternative to CNN, has been proven effective in many modalities (e.g., texts and images). For 3D point cloud transformers, existing efforts focus primarily on pushing their accuracy to the state-of-the-art level. However, their latency lags behind sparse convolution-based models (**3$\times$ slower**), hindering their usage in resource-constrained, latency-sensitive applications (such as autonomous driving). This inefficiency comes from point clouds' sparse and irregular nature, whereas transformers are designed for dense, regular workloads. This paper presents **FlatFormer** to close this latency gap by trading spatial proximity for better computational regularity. We first flatten the point cloud with window-based sorting and partition points into **groups of equal sizes** rather than **windows of equal shapes**. This effectively avoids expensive structuring and padding overheads. We then apply self-attention within groups to extract local features, alternate sorting axis to gather features from different directions, and shift windows to exchange features across groups. FlatFormer delivers state-of-the-art accuracy on Waymo Open Dataset with **4.6x** speedup over (transformer-based) [SST](https://arxiv.org/abs/2112.06375) and **1.4x** speedup over (sparse convolutional) [CenterPoint](https://arxiv.org/abs/2006.11275). This is the first point cloud transformer that achieves real-time performance on edge GPUs and is faster than sparse convolutional methods while achieving on-par or even superior accuracy on large-scale benchmarks. 

## Results

All the results are reproducible with this repo. Regrettably, we are unable to provide the pre-trained model weights due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/). Discussions are definitely welcome if you could not obtain satisfactory performances with FlatFormer in your projects.

### 3D Object Detection (on Waymo validation)

| Model                                                        | #Sweeps | mAP/H_L1  | mAP/H_L2  | Veh_L1    | Veh_L2    | Ped_L1    | Ped_L2    | Cyc_L1    | Cyc_L2    |
| ------------------------------------------------------------ | ------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| [FlatFormer](https://github.com/mit-han-lab/flatformer-dev/blob/main/configs/flatformer/flatformer_waymo_D1_2x_3class.py) | 1       | 76.1/73.4 | 69.7/67.2 | 77.5/77.1 | 69.0/68.6 | 79.6/73.0 | 71.5/65.3 | 71.3/70.1 | 68.6/67.5 |
| [FlatFormer](https://github.com/mit-han-lab/flatformer-dev/blob/main/configs/flatformer/flatformer_waymo_D1_2x_3class_2f.py) | 2       | 78.9/77.3 | 72.7/71.2 | 79.1/78.6 | 70.8/70.3 | 81.6/78.2 | 73.8/70.5 | 76.1/75.1 | 73.6/72.6 |
| [FlatFormer](https://github.com/mit-han-lab/flatformer-dev/blob/main/configs/flatformer/flatformer_waymo_D1_2x_3class_3f.py) | 3       | 79.6/78.0 | 73.5/72.0 | 79.7/79.2 | 71.4/71.0 | 82.0/78.7 | 74.5/71.3 | 77.2/76.1 | 74.7/73.7 |


## Usage

### Prerequisites

The code is built with following libraries:

* Python >= 3.6, <3.8
* [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, <= 1.10.2
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

```
# multi-gpu training
bash tools/dist_train.sh configs/flatformer/$CONFIG.py 8 --work-dir $CONFIG/ --cfg-options evaluation.pklfile_prefix=./work_dirs/$CONFIG/results evaluation.metric=waymo
```

### Evaluation

```
# multi-gpu testing
bash tools/dist_test.sh configs/flatformer/$CONFIG.py /work_dirs/$CONFIG/latest.pth 8 --eval waymo
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
