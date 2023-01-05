import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from spconv.core import ConvAlgo

import spconv.pytorch as spconv
from spconv.test_utils import TestCase, generate_sparse_data, params_grid
from spconv.constants import FILTER_HWIO
# import sparseconvnet as scn

# we must disable tf32 to increase reference precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



class SPSSConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 algo=spconv.ConvAlgo.Native):
        super().__init__()

        self.net = spconv.SPSSConv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding=padding,
                              dilation=dilation,
                              indice_key="key_"+ str(0),
                              bias=False,
                              algo=algo)

        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size, pruning_ratio=0.5):
        coors = coors.int()  # .cpu()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        mask = torch.ones((features.shape[0],)).bool()

        # assert False
        shuffle_index=torch.randperm(features.shape[0])[0:int(features.shape[0]*pruning_ratio)]
        mask[shuffle_index] = False
        mask = mask.cuda()
        
        return self.net(x, mask)  # .dense()

class SPRSConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 algo=spconv.ConvAlgo.Native):
        super().__init__()

        self.net = spconv.SPRSConv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              pruning_ratio=0.5,
                              padding=padding,
                              dilation=dilation,
                              indice_key="key_"+ str(0),
                              bias=False,
                              algo=algo)

        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size, pruning_ratio=0.5):
        coors = coors.int()  # .cpu()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        '''
        mask = torch.ones((features.shape[0],)).bool()

        shuffle_index=torch.randperm(features.shape[0])[0:int(features.shape[0]*pruning_ratio)]
        mask[shuffle_index] = False
        mask = mask.cuda()
        '''
        mask = abs(torch.rand((features.shape[0],))).cuda()
        
        return self.net(x, mask)  # .dense()


class SpatialGroupConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 spatial_groups,
                 stride,
                 padding,
                 dilation,
                 position_embedding=False,
                 algo=spconv.ConvAlgo.Native):
        super().__init__()
        layers = spconv.SpatialGroupConv3d(in_channels,
                              out_channels,
                              kernel_size, spatial_groups,
                              stride,
                              padding=1,
                              dilation=dilation,
                              indice_key="key_"+ str(0),
                              bias=False,
                              position_embedding=position_embedding,
                              algo=algo)
        self.net = layers
        self.grid = None
        self.shape = shape
        _list = [0, kernel_size//2, kernel_size//2+1, kernel_size]
        self.group_map = torch.zeros((kernel_size**3, spatial_groups**3)) - 1
        _num = 0
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    _pos = a.sum()
                    self.group_map[_num][:_pos] = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    _num += 1
        self.group_map = self.group_map.int().cuda()
        if position_embedding:
            self.net.position_embedding.data.fill_(0)

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size,
                                    self.grid)
        return self.net(x, group_map=self.group_map)


def main_spss(algo=spconv.ConvAlgo.Native, dtype=torch.float32, pruning_ratio=0.5):
    # function for develop.
    np.random.seed(484)
    torch.manual_seed(50051)
    # devices = ["cuda:0"]
    devices = ["cuda"]
    shapes = [[400, 400, 15]]
    batchsizes = [2]

    in_channels = [32]
    out_channels = [32]
    ksizes = [(3, 3, 3)]
    strides = [1]
    paddings = [1]
    dilations = [1]
    for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations):
        if all([s > 1, d > 1]):
            continue
        device = torch.device(dev)
        num_points = [120000] * bs

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)

        indices_t = torch.from_numpy(indices)

        indices_t = torch.from_numpy(indices).int().to(device).to(dtype)
        features_t = torch.from_numpy(features).to(device).to(dtype)


        net = SPSSConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d,
                                  algo=algo).to(device).to(dtype)

        out = net(features_t, indices_t, bs, pruning_ratio=pruning_ratio)

        out = out.dense()
        out_numpy = out.detach().cpu().numpy()

        print(out_numpy.min(), out_numpy.max(), out_numpy.mean(),
              out_numpy.sum())
    return out_numpy


def main_sprs(algo=spconv.ConvAlgo.Native, dtype=torch.float32, pruning_ratio=0.5):
    # function for develop.
    np.random.seed(484)
    torch.manual_seed(50051)
    # devices = ["cuda:0"]
    devices = ["cuda"]
    shapes = [[400, 400, 15]]
    batchsizes = [2]

    in_channels = [32]
    out_channels = [64]
    ksizes = [3]
    strides = [2]
    paddings = [1]
    dilations = [1]
    for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations):
        if all([s > 1, d > 1]):
            continue
        device = torch.device(dev)
        num_points = [120000] * bs

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)

        indices_t = torch.from_numpy(indices)
        # filters = np.random.uniform(0, 1, size=[k[0], 1, 1, IC,
        #                                         OC]).astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device).to(dtype)
        features_t = torch.from_numpy(features).to(device).to(dtype)

        # features_dense_t = torch.from_numpy(features_dense).to(device).to(
        #     dtype)
        net = SPRSConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d,
                                  algo=algo).to(device).to(dtype)

        out = net(features_t, indices_t, bs, pruning_ratio=pruning_ratio)

        out = out.dense()
        out_numpy = out.detach().cpu().numpy()
        print(out_numpy.min(), out_numpy.max(), out_numpy.mean(),
              out_numpy.sum())
    return out_numpy

def main_spatialgroupconv(algo=spconv.ConvAlgo.Native, dtype=torch.float32):
    # function for develop.
    np.random.seed(484)
    torch.manual_seed(50051)
    # devices = ["cuda:0"]
    devices = ["cuda"]
    shapes = [[400, 400, 15]]
    batchsizes = [2]

    spatial_groups = [3]
    in_channels = [16]
    out_channels = [16]
    ksizes = [7]
    strides = [1]
    paddings = [1]
    dilations = [1]
    for dev, shape, bs, IC, OC, k, g, s, p, d in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes,
            spatial_groups, strides, paddings, dilations):
        if all([s > 1, d > 1]):
            continue
        device = torch.device(dev)
        num_points = [120000] * bs

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        indices_t = torch.from_numpy(indices)
        filters = np.random.uniform(0, 1, size=[k, 1, 1, IC, OC]).astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device).to(dtype)
        features_t = torch.from_numpy(features).to(device).to(dtype)

        features_dense_t = torch.from_numpy(features_dense).to(device).to(
            dtype)
        net = SpatialGroupConv3dTestTorch(1, 3, shape, IC, OC, k, g, s, p, d, algo=algo).to(device).to(dtype)
        # filters_t = torch.from_numpy(filters).to(device).to(dtype)
        # net_ref.net[0].weight[:] = filters_t.permute(4, 3, 0, 1,
        #                                              2).contiguous()
        # net.net[0].weight[:] = filters_t
        # out_ref = net_ref(features_dense_t)

        times = []
        for i in range(10):
            t = time.time()
            out = net(features_t, indices_t, bs)
            torch.cuda.synchronize()
            times.append(time.time() - t)

        out = net(features_t, indices_t, bs)
        out = out.dense()
        out_numpy = out.detach().cpu().numpy()
        print(out_numpy.min(), out_numpy.max(), out_numpy.mean(),
              out_numpy.sum())
    return out_numpy
    
if __name__ == '__main__':
    main_spss(algo=spconv.ConvAlgo.Native, dtype=torch.float32, pruning_ratio=0.5)
    main_sprs(algo=spconv.ConvAlgo.Native, dtype=torch.float32, pruning_ratio=0.5)
    main_spatialgroupconv(algo=spconv.ConvAlgo.Native, dtype=torch.float32)
