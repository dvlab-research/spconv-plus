# spconv-plus

This project is based on the original [spconv](https://github.com/traveller59/spconv). We integrate several new sparse convolution types and operators that might be useful into this library. 


## 1. Operators
### Focals Conv
This is introduced in our [CVPR 2022 (oral) paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.pdf). In this paper, we introduce a new type of sparse convolution that makes feature sparsity learnable with position-wise importance prediction. 

The source code for this operator in this library is [Focals Conv](https://github.com/dvlab-research/spconv-plus/blob/85bc7567b6f867c50580b2291aeb086ad0485489/spconv/pytorch/conv.py#L1010). The example for use this work is shown in its [repo](https://github.com/dvlab-research/FocalsConv/blob/master/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/focal_sparse_conv_cuda.py).


### Spatial Pruned Conv
This is introduced in our [NeurIPS 2022 paper](https://openreview.net/pdf?id=QqWqFLbllZh). In this paper, we propose two new convolution operators, spatial pruned submanifold sparse convolution (SPSS-Conv) and spatial pruned regular sparse convolution (SPRS-Conv), both of which are based on the idea of dynamically determining crucial areas for redundancy reduction.

The source codes for these two operators in this library are shown in [SPSSConv3d](https://github.com/dvlab-research/spconv-plus/blob/85bc7567b6f867c50580b2291aeb086ad0485489/spconv/pytorch/conv.py#L980) and [SPRSConv3d](https://github.com/dvlab-research/spconv-plus/blob/85bc7567b6f867c50580b2291aeb086ad0485489/spconv/pytorch/conv.py#L1104). The example for them can be found in this [file](https://github.com/dvlab-research/spconv-plus/blob/master/examples.py) and its [repo](https://github.com/CVMI-Lab/SPS-Conv).


### Spatial-wise Group Conv
This is introduced in our [Arxiv paper](https://arxiv.org/pdf/2206.10555.pdf). In this paper, we introduce spatial-wise group (partition) convolution, that enables an efficient way to implement 3D large kernels.

The source code for this operators in this library is shown in [SpatialGroupConv3d](https://github.com/dvlab-research/spconv-plus/blob/85bc7567b6f867c50580b2291aeb086ad0485489/spconv/pytorch/conv.py#L1070). The example for it is shown in this [file](https://github.com/dvlab-research/spconv-plus/blob/master/examples.py).


### Channel-wise Group Conv
This is the commonly-used group convolution. We implement this operator into this library. You can directly set "groups" in SparseConvolution.

### Submanifold Sparse Max Pooling
We enable the submanifold version of sparse max pooling in this library. You can directly set "subm=True" when using SparseMaxPool3d. For example,
spconv.SparseMaxPool3d(3, 1, 1, subm=True, algo=ConvAlgo.Native, indice_key='max_pool')



## 2. Installation
This repo should be built from source. Following the readme file in the spconv library,

- install build-essential, install CUDA
- run ```export SPCONV_DISABLE_JIT="1"```
- run ```pip install pccm cumm wheel```
- run ```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

## 3. Citation
Please consider to cite our papers if this repo is helpful.
```
@inproceedings{focalsconv-chen,
  title={Focal Sparse Convolutional Networks for 3D Object Detection},
  author={Chen, Yukang and Li, Yanwei and Zhang, Xiangyu and Sun, Jian and Jia, Jiaya},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

```
@inproceedings{liu2022spatial,
  title={Spatial Pruned Sparse Convolution for Efficient 3D Object Detection},
  author={Liu, Jianhui and Chen, Yukang and Ye, Xiaoqing and Tian, Zhuotao and Tan, Xiao and Qi, Xiaojuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

```
@article{largekernel3d-chen,
  author    = {Chen, Yukang and Liu, Jianhui and Qi, Xiaojuan and Zhang, Xiangyu and Sun, Jian and Jia, Jiaya},
  title     = {Scaling up Kernels in 3D CNNs},
  journal   = {arxiv},
  year      = {2022},
}
```
