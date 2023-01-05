# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union
import torch
from cumm import tensorview as tv

from spconv.core_cc.csrc.sparse.all import SpconvOps
from spconv.pytorch.cppcore import torch_tensor_to_tv, get_current_stream


class PointToVoxel(object):
    """WARNING: you MUST construct PointToVoxel AFTER set device.
    """
    def __init__(self,
                 vsize_xyz: List[float],
                 coors_range_xyz: List[float],
                 num_point_features: int,
                 max_num_voxels: int,
                 max_num_points_per_voxel: int,
                 device: torch.device = torch.device("cpu:0")):
        self.ndim = len(vsize_xyz)

        self.device = device
        vsize, grid_size, grid_stride, coors_range = SpconvOps.calc_point2voxel_meta_data(
            vsize_xyz, coors_range_xyz)
        self.num_point_features = num_point_features
        self.max_num_voxels = max_num_voxels
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.vsize = vsize
        self.grid_size = grid_size
        self.grid_stride = grid_stride
        self.coors_range = coors_range

        self.voxels = torch.zeros(
            [max_num_voxels, max_num_points_per_voxel, num_point_features],
            dtype=torch.float32,
            device=device)
        self.indices = torch.zeros([max_num_voxels, self.ndim],
                                   dtype=torch.int32,
                                   device=device)
        self.num_per_voxel = torch.zeros([max_num_voxels],
                                         dtype=torch.int32,
                                         device=device)
        if device.type == "cpu":
            self.hashdata = torch.full(grid_size,
                                       -1,
                                       dtype=torch.int32,
                                       device=device)
            self.point_indice_data = torch.Tensor()
        else:
            self.hashdata = torch.empty([1, 2],
                                        dtype=torch.int64,
                                        device=device)
            self.point_indice_data = torch.empty([1],
                                                 dtype=torch.int64,
                                                 device=device)

    def __call__(self,
                 pc: torch.Tensor,
                 clear_voxels: bool = True,
                 empty_mean: bool = False):
        """generate voxels/indices/num_point_per_voxel/pc_voxel_ids from 
        point cloud.
        This function don't return pc_voxel_id for backward compatility.
        pc_voxel_id will be added in spconv 2.2.
        Args:
            pc: [N, 3+] point cloud.
            clear_voxels: if True, call zero on voxels
            empty_mean: if True, full empty location of voxels with mean.
        Returns:
            voxels: voxels
            indices: quantized coords
            num_per_voxel: number of points in a voxel
        """

        res = self.generate_voxel_with_id(pc, clear_voxels, empty_mean)
        return res[0], res[1], res[2]

    def generate_voxel_with_id(self,
                 pc: torch.Tensor,
                 clear_voxels: bool = True,
                 empty_mean: bool = False):
        """generate voxels/indices/num_point_per_voxel/pc_voxel_ids from 
        point cloud.
        Args:
            pc: [N, 3+] point cloud.
            clear_voxels: if True, call zero on voxels
            empty_mean: if True, full empty location of voxels with mean.
        Returns:
            voxels: voxels
            indices: quantized coords
            num_per_voxel: number of points in a voxel
            pc_voxel_id: voxel id for every point. if not exists, -1.
        """
        assert pc.device.type == self.device.type, "your pc device is wrong"
        expected_hash_data_num = pc.shape[0] * 2
        with torch.no_grad():
            pc_voxel_id = torch.empty([pc.shape[0]],
                                    dtype=torch.int64,
                                    device=self.device)
            pc_voxel_id_tv = torch_tensor_to_tv(pc_voxel_id)

            if self.device.type != "cpu":
                hashdata = torch.empty([expected_hash_data_num, 2],
                                            dtype=torch.int64,
                                            device=pc.device)

                point_indice_data = torch.empty([pc.shape[0]],
                                            dtype=torch.int64,
                                            device=pc.device)

                pc_tv = torch_tensor_to_tv(pc)
                stream = get_current_stream()
                voxels_tv = torch_tensor_to_tv(self.voxels)
                indices_tv = torch_tensor_to_tv(self.indices)
                num_per_voxel_tv = torch_tensor_to_tv(self.num_per_voxel)
                hashdata_tv = torch_tensor_to_tv(
                    hashdata,
                    dtype=tv.custom128,
                    shape=[hashdata.shape[0]])
                point_indice_data_tv = torch_tensor_to_tv(point_indice_data)
                with torch.cuda.device(pc.device):
                    res = SpconvOps.point2voxel_cuda(
                        pc_tv, voxels_tv, indices_tv, num_per_voxel_tv,
                        hashdata_tv, point_indice_data_tv, pc_voxel_id_tv, self.vsize,
                        self.grid_size, self.grid_stride, self.coors_range,
                        empty_mean, clear_voxels, stream)
                num_voxels = res[0].shape[0]
            else:
                pc_tv = torch_tensor_to_tv(pc)
                voxels_tv = torch_tensor_to_tv(self.voxels)
                indices_tv = torch_tensor_to_tv(self.indices)
                num_per_voxel_tv = torch_tensor_to_tv(self.num_per_voxel)
                hashdata_tv = torch_tensor_to_tv(self.hashdata, dtype=tv.int32)
                res = SpconvOps.point2voxel_cpu(pc_tv, voxels_tv, indices_tv,
                                                num_per_voxel_tv, hashdata_tv,
                                                pc_voxel_id_tv,
                                                self.vsize, self.grid_size,
                                                self.grid_stride,
                                                self.coors_range, empty_mean,
                                                clear_voxels)
                num_voxels = res[0].shape[0]

            return (self.voxels[:num_voxels].clone(), self.indices[:num_voxels].clone(),
                    self.num_per_voxel[:num_voxels].clone(), pc_voxel_id)


def gather_features_by_pc_voxel_id(seg_res_features: torch.Tensor, pc_voxel_id: torch.Tensor, invalid_value: Union[int, float] = 0):
    """This function is used to gather segmentation result to match origin pc.
    """
    if seg_res_features.device != pc_voxel_id.device:
        pc_voxel_id = pc_voxel_id.to(seg_res_features.device)
    res_feature_shape = (pc_voxel_id.shape[0], *seg_res_features.shape[1:])
    if invalid_value == 0:
        res = torch.zeros(res_feature_shape, dtype=seg_res_features.dtype, device=seg_res_features.device)
    else:
        res = torch.full(res_feature_shape, invalid_value, dtype=seg_res_features.dtype, device=seg_res_features.device)
    pc_voxel_id_valid = pc_voxel_id != -1
    pc_voxel_id_valid_ids = torch.nonzero(pc_voxel_id_valid).view(-1)
    seg_res_features_valid = seg_res_features[pc_voxel_id[pc_voxel_id_valid_ids]]
    res[pc_voxel_id_valid_ids] = seg_res_features_valid
    return res


def split_voxels(x, b, voxel_importance, kernel_offsets, mask_multi=True, pruning_mode="topk", pruning_ratio=0.5):
    index = x.indices[:, 0]
    batch_index = index == b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    voxel_importance = voxel_importance[batch_index]

    if mask_multi:
        features_ori *= voxel_importance

    # get mask
    # print("pruning_mode-----------------------:", pruning_mode)
    if pruning_mode == "topk":
        if pruning_ratio == 0.5:
            mid_value = voxel_importance.median()
            # print('mid_value', mid_value)
            indices_im = (voxel_importance.view(-1, ) > mid_value)
            indices_nim = (voxel_importance.view(-1, ) <= mid_value)
        else:
            _, indices = voxel_importance.view(-1, ).sort()
            indices_im = indices[int(voxel_importance.shape[0] * pruning_ratio):]
            indices_nim = indices[:int(voxel_importance.shape[0] * pruning_ratio)]
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
        # print("indices_im num:", indices_im.shape, "indices_nim num:",indices_nim.shape, "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
    elif pruning_mode == "thre":
        indices_im = (voxel_importance.view(-1, ) > pruning_ratio)
        indices_nim = (voxel_importance.view(-1, ) <= pruning_ratio)
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)

    features_im = features_ori[indices_im]
    coords_im = indices_ori[indices_im]
    voxel_kerels_offset = kernel_offsets.unsqueeze(0).repeat(features_im.shape[0], 1,
                                                             1)  # [features_im.shape[0], 26, 3]
    indices_im_kernels = coords_im[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0],
                                                              1)  # [coords_im.shape[0], 26, 3]
    # print("kernel_offsets:", kernel_offsets.dtype, "indices_im_kernels:", indices_im_kernels.dtype, "voxel_kerels_offset:", voxel_kerels_offset.dtype)
    indices_with_imp = (indices_im_kernels + voxel_kerels_offset).view(-1, 3)
    spatial_indices = (indices_with_imp[:, 0] > 0) * (indices_with_imp[:, 1] > 0) * (indices_with_imp[:, 2] > 0) * \
                      (indices_with_imp[:, 0] < x.spatial_shape[0]) * (indices_with_imp[:, 1] < x.spatial_shape[1]) * (
                                  indices_with_imp[:, 2] < x.spatial_shape[2])

    selected_indices = indices_with_imp[spatial_indices]
    selected_indices = torch.cat(
        [torch.ones((selected_indices.shape[0], 1), device=features_im.device, dtype=torch.int) * b, selected_indices],
        dim=1)
    selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_im.device)

    features_im = torch.cat([features_im, selected_features], dim=0)  # [N', C]
    coords_im = torch.cat([coords_im, selected_indices], dim=0)  # [N', 3]
    # mask_kernel_im = voxel_importance[indices_im][spatial_indices]
    # mask_kernel_im = mask_kernel_im.unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1)
    # mask_kernel_im = torch.cat([torch.ones(features_im_cat.shape[0], device=features_im.device), mask_kernel_im], dim=0)
    # print("before:", features_im.shape)
    assert features_im.shape[0] == coords_im.shape[0]
    if indices_im.sum() > 0:
        features_im, coords_im, _ = check_repeat(features_im, coords_im)
        # print("after:", features_im.shape)
    # print("coords_im after:", coords_im.dtype)
    features_nim = features_ori[indices_nim]
    coords_nim = indices_ori[indices_nim]

    return features_im, coords_im, features_nim, coords_nim


def check_repeat(x_foreground_features, x_foreground_indices, additional_features=None, sort_first=True,
                 flip_first=True):
    if sort_first:
        x_foreground_features, x_foreground_indices, additional_features = sort_by_indices(x_foreground_features,
                                                                                           x_foreground_indices,
                                                                                           additional_features)

    if flip_first:
        x_foreground_features, x_foreground_indices = x_foreground_features.flip([0]), x_foreground_indices.flip([0])

    if not additional_features is None:
        additional_features = additional_features.flip([0])

    a = x_foreground_indices[:, 1:].int()
    augmented_a = torch.add(torch.add(a.select(1, 0) * a[:, 1].max() * a[:, 2].max(), a.select(1, 1) * a[:, 2].max()),
                            a.select(1, 2))
    _unique, inverse, counts = torch.unique_consecutive(augmented_a, return_inverse=True, return_counts=True, dim=0)

    if _unique.shape[0] < x_foreground_indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        x_foreground_features_new = torch.zeros((_unique.shape[0], x_foreground_features.shape[-1]),
                                                device=x_foreground_features.device)
        x_foreground_features_new.index_add_(0, inverse.long(), x_foreground_features)
        x_foreground_features = x_foreground_features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        x_foreground_indices = x_foreground_indices[perm_].int()

        if not additional_features is None:
            additional_features_new = torch.zeros((_unique.shape[0],), device=additional_features.device)
            additional_features_new.index_add(0, inverse.long(), additional_features)
            additional_features = additional_features_new / counts
    return x_foreground_features, x_foreground_indices, additional_features


def sort_by_indices(features_foreground_cat, indices_foreground_coords, additional_features=None):
    a = indices_foreground_coords[:, 1:]
    # print("a shape:", a.shape)
    augmented_a = a.select(1, 0) * a[:, 1].max() * a[:, 2].max() + a.select(1, 1) * a[:, 2].max() + a.select(1, 2)
    augmented_a_sorted, ind = augmented_a.sort()
    features_foreground_cat = features_foreground_cat[ind]
    indices_foreground_coords = indices_foreground_coords[ind]
    if not additional_features is None:
        additional_features = additional_features[ind]
    return features_foreground_cat, indices_foreground_coords, additional_features
