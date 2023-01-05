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

import sys
import pickle 

import torch
from torch import nn
from torch.autograd import Function
from typing import Optional, TypeVar
from spconv.tools import CUDAKernelTimer
from spconv.pytorch import ops, SparseConvTensor
from spconv.pytorch.constants import PYTORCH_VERSION
from spconv.debug_utils import spconv_save_debug_data
from torch.autograd.function import once_differentiable
import numpy as np
from pathlib import Path
from spconv.pytorch.hash import HashTable
from cumm.gemm.layout import to_stride
from typing import List
from functools import reduce, partial
from spconv.constants import FILTER_HWIO


_MAX_INT32 = 2147483647

_T = TypeVar("_T")

def identity_decorator(func: _T) -> _T:
    return func

if PYTORCH_VERSION >= [1, 6, 0]:
    import torch.cuda.amp as amp
    _TORCH_CUSTOM_FWD = amp.custom_fwd(cast_inputs=torch.float16)
    _TORCH_CUSTOM_BWD = amp.custom_bwd

else:
    _TORCH_CUSTOM_FWD = identity_decorator
    _TORCH_CUSTOM_BWD = identity_decorator

class SparseConvFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx,
                features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                algo,
                groups,
                timer: CUDAKernelTimer = CUDAKernelTimer(False)):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        ctx.algo = algo
        ctx.timer = timer
        try:
            conv_func = ops.indice_conv_groups if groups>1 else ops.indice_conv
            return conv_func(features,
                                filters,
                                indice_pairs,
                                indice_pair_num,
                                num_activate_out,
                                False,
                                groups=groups,
                                algo=algo,
                                timer=timer)
        except Exception as e:
            msg = "[Exception|indice_conv]"
            msg += f"feat={features.shape},w={filters.shape},pair={indice_pairs.shape},"
            msg += f"pairnum={indice_pair_num},act={num_activate_out},algo={algo}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((indice_pairs, indice_pair_num))
            raise e 

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        timer = ctx.timer
        try:
            input_bp, filters_bp = ops.indice_conv_backward(features,
                                                            filters,
                                                            grad_output,
                                                            indice_pairs,
                                                            indice_pair_num,
                                                            False,
                                                            algo=ctx.algo,
                                                            timer=timer)
        except Exception as e:
            msg = "[Exception|indice_conv_backward]"
            msg += f"feat={features.shape},w={filters.shape},pair={indice_pairs.shape},"
            msg += f"pairnum={indice_pair_num},do={grad_output.shape}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((indice_pairs, indice_pair_num))
            raise e 

        return input_bp, filters_bp, None, None, None, None, None, None


class SparseInverseConvFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx,
                features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                algo,
                timer: CUDAKernelTimer = CUDAKernelTimer(False)):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        ctx.algo = algo
        ctx.timer = timer
        try:
            return ops.indice_conv(features,
                                filters,
                                indice_pairs,
                                indice_pair_num,
                                num_activate_out,
                                True,
                                False,
                                algo=algo,
                                timer=timer)
        except Exception as e:
            msg = "[Exception|indice_conv|inverse]"
            msg += f"feat={features.shape},w={filters.shape},pair={indice_pairs.shape},"
            msg += f"pairnum={indice_pair_num},act={num_activate_out},algo={algo}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((indice_pairs, indice_pair_num))
            raise e 

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        timer = ctx.timer
        try:
            input_bp, filters_bp = ops.indice_conv_backward(features,
                                                            filters,
                                                            grad_output,
                                                            indice_pairs,
                                                            indice_pair_num,
                                                            True,
                                                            False,
                                                            algo=ctx.algo,
                                                            timer=timer)
        except Exception as e:
            msg = "[Exception|indice_conv_backward|inverse]"
            msg += f"feat={features.shape},w={filters.shape},pair={indice_pairs.shape},"
            msg += f"pairnum={indice_pair_num},do={grad_output.shape}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((indice_pairs, indice_pair_num))
            raise e 

        return input_bp, filters_bp, None, None, None, None, None


class SparseImplicitGemmFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx,
                features: torch.Tensor,
                filters: torch.Tensor,
                pair_fwd: torch.Tensor,
                pair_bwd: torch.Tensor,
                pair_mask_fwd_splits: List[torch.Tensor],
                pair_mask_bwd_splits: List[torch.Tensor],
                mask_argsort_fwd_splits: List[torch.Tensor],
                mask_argsort_bwd_splits: List[torch.Tensor],
                num_activate_out: int,
                masks: List[np.ndarray],
                is_train: bool,
                is_subm: bool,
                timer: CUDAKernelTimer = CUDAKernelTimer(False),
                fp32_accum: Optional[bool] = None):
        try:
            out, mask_out, mask_width = ops.implicit_gemm(features, filters,
                                                        pair_fwd,
                                                        pair_mask_fwd_splits,
                                                        mask_argsort_fwd_splits,
                                                        num_activate_out, masks,
                                                        is_train, is_subm, timer,
                                                        fp32_accum)
        except Exception as e:
            msg = "[Exception|implicit_gemm]"
            msg += f"feat={features.shape},w={filters.shape},pair={pair_fwd.shape},"
            msg += f"act={num_activate_out},issubm={is_subm},istrain={is_train}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((pair_fwd, pair_bwd, pair_mask_fwd_splits, 
                pair_mask_bwd_splits, mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                masks))
            raise e 
            
        ctx.save_for_backward(features, filters, pair_fwd, pair_bwd)
        ctx.mask_width = mask_width
        ctx.mask_out = mask_out
        ctx.timer = timer
        ctx.pair_mask_fwd_splits = pair_mask_fwd_splits
        ctx.mask_argsort_fwd_splits = mask_argsort_fwd_splits
        ctx.pair_mask_bwd_splits = pair_mask_bwd_splits
        ctx.mask_argsort_bwd_splits = mask_argsort_bwd_splits
        # ctx.num_activate_out = num_activate_out
        ctx.masks = masks
        ctx.is_subm = is_subm
        ctx.fp32_accum = fp32_accum
        return out

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        features, filters, pair_fwd, pair_bwd = ctx.saved_tensors
        mask_width = ctx.mask_width
        mask_out = ctx.mask_out
        pair_mask_fwd_splits = ctx.pair_mask_fwd_splits
        mask_argsort_fwd_splits = ctx.mask_argsort_fwd_splits
        pair_mask_bwd_splits = ctx.pair_mask_bwd_splits
        mask_argsort_bwd_splits = ctx.mask_argsort_bwd_splits
        # num_activate_out = ctx.num_activate_out
        masks = ctx.masks
        is_subm = ctx.is_subm
        timer = ctx.timer
        fp32_accum = ctx.fp32_accum

        try:
            input_bp, filters_bp = ops.implicit_gemm_backward(
                features,
                filters,
                grad_output,
                pair_fwd,
                pair_bwd,
                pair_mask_fwd_splits,
                pair_mask_bwd_splits,
                mask_argsort_fwd_splits,
                mask_argsort_bwd_splits,
                mask_output_fwd=mask_out,
                masks=masks,
                mask_width=mask_width,
                is_subm=is_subm,
                timer=timer,
                fp32_accum=fp32_accum)
        except Exception as e:
            msg = "[Exception|implicit_gemm_backward]"
            msg += f"feat={features.shape},w={filters.shape},pair={pair_fwd.shape},"
            msg += f"issubm={is_subm},do={grad_output.shape}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((pair_fwd, pair_bwd, pair_mask_fwd_splits, 
                pair_mask_bwd_splits, mask_argsort_fwd_splits, mask_argsort_bwd_splits,
                masks))
            raise e 

        None_9 = [None] * 12
        return (input_bp, filters_bp, *None_9)


class SubMConvFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx,
                features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                algo,
                groups,
                spatial_groups=1,
                position_embedding=None,
                timer: CUDAKernelTimer = CUDAKernelTimer(False)):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters, position_embedding)
        ctx.algo = algo
        ctx.timer = timer
        try:
            conv_func = ops.indice_conv_groups if groups>1 else ops.indice_conv
            if spatial_groups > 1:
                conv_func = partial(ops.indice_conv_spatial_groups, position_embedding=position_embedding)
            return conv_func(features,
                                filters,
                                indice_pairs,
                                indice_pair_num,
                                num_activate_out,
                                False,
                                True,
                                groups=groups,
                                algo=algo,
                                timer=timer)
        except Exception as e:
            msg = "[Exception|indice_conv|subm]"
            msg += f"feat={features.shape},w={filters.shape},pair={indice_pairs.shape},"
            msg += f"pairnum={indice_pair_num},act={num_activate_out},algo={algo}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((indice_pairs, indice_pair_num))
            raise e 

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters, position_embedding = ctx.saved_tensors
        timer = ctx.timer
        try:
            backward_func = partial(ops.indice_conv_backward_spatial_groups, position_embedding=position_embedding) if not position_embedding is None else ops.indice_conv_backward
            input_bp, filters_bp, pe_bp = backward_func(features,
                                                            filters,
                                                            grad_output,
                                                            indice_pairs,
                                                            indice_pair_num,
                                                            False,
                                                            True,
                                                            algo=ctx.algo,
                                                            timer=timer)
        except Exception as e:
            msg = "[Exception|indice_conv_backward|subm]"
            msg += f"feat={features.shape},w={filters.shape},pair={indice_pairs.shape},"
            msg += f"pairnum={indice_pair_num},do={grad_output.shape}"
            print(msg, file=sys.stderr)
            spconv_save_debug_data((indice_pairs, indice_pair_num))
            raise e 

        return input_bp, filters_bp, None, None, None, None, None, None, pe_bp, None


class SparseMaxPoolFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx, features, indice_pairs, indice_pair_num,
                num_activate_out, is_subm, return_inverse):
        out, inverse = ops.indice_maxpool(features, indice_pairs, indice_pair_num,
                                 num_activate_out, is_subm, return_inverse)
        subm = torch.Tensor([is_subm]).to(out.device)[0]
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out, subm)
        return out, inverse

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out, subm = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(features, out, grad_output,
                                               indice_pairs, indice_pair_num, subm)
        return input_bp, None, None, None, None, None


class SparseMaxPoolImplicitGemmFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx, features: torch.Tensor, indice_pairs_fwd: torch.Tensor,
                indice_pairs_bwd: torch.Tensor, num_activate_out: int):
        out = ops.indice_maxpool_implicit_gemm(features, indice_pairs_fwd,
                                               num_activate_out)
        ctx.save_for_backward(indice_pairs_bwd, features, out)
        return out

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs_bwd, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_implicit_gemm_backward(
            features, out, grad_output, indice_pairs_bwd)
        return input_bp, None, None, None


class SparseAvgPoolFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx, features, indice_pairs, indice_pair_num,
                num_activate_out, is_subm):
        out, out_nums  = ops.indice_avgpool(features, indice_pairs, indice_pair_num,
                                 num_activate_out, is_subm)
        subm = torch.Tensor([is_subm]).to(out.device)[0]
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out, out_nums, subm)
        return out

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out, out_nums, is_subm = ctx.saved_tensors
        input_bp = ops.indice_avgpool_backward(features, out, grad_output,
                                               indice_pairs, indice_pair_num, out_nums, is_subm)
        return input_bp, None, None, None, None


indice_conv = SparseConvFunction.apply
implicit_gemm = SparseImplicitGemmFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply
indice_maxpool_implicit_gemm = SparseMaxPoolImplicitGemmFunction.apply
indice_avgpool = SparseAvgPoolFunction.apply


def _indice_to_scalar(indices: torch.Tensor, shape: List[int]):
    assert indices.shape[1] == len(shape)
    stride = to_stride(np.array(shape, dtype=np.int64))
    scalar_inds = indices[:, -1].clone()
    for i in range(len(shape) - 1):
        scalar_inds += stride[i] * indices[:, i]
    return scalar_inds.contiguous()

def sparse_add_hash_based(*tens: SparseConvTensor):
    """ sparse add with misaligned indices.
    if you use sparse add, the indice_dict will be dropped and impossible
    to use inverse.
    There is only one situation that keep indices: there is one operand that
    its indices is output indices.
    
    """
    table_size = 0
    max_num_indices = 0
    max_num_indices_idx = 0
    for i, ten in enumerate(tens):
        assert ten.spatial_shape == tens[0].spatial_shape
        assert ten.batch_size == tens[0].batch_size
        assert ten.features.shape[1] == tens[0].features.shape[1]
        table_size += ten.features.shape[0]
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        
    first = tens[0]
    feat = first.features
    shape = [first.batch_size, *first.spatial_shape]
    whole_shape = int(np.prod(shape))
    table_size *= 2
    k_type = torch.int32
    if whole_shape >= _MAX_INT32:
        k_type = torch.int64
    table = HashTable(first.features.device, k_type, torch.int32, table_size)
    scalars: List[torch.Tensor] = []
    for ten in tens:
        indices = ten.indices
        if whole_shape >= _MAX_INT32:
            indices = indices.long()
        scalar = _indice_to_scalar(indices, shape)
        scalars.append(scalar)
        table.insert(scalar)
    # assign arange to values of hash table
    count = table.assign_arange_()
    count_val = count.item()
    out_features = torch.zeros([int(count_val), feat.shape[1]], dtype=feat.dtype, device=feat.device)
    out_indices = torch.zeros([int(count_val), first.indices.shape[1]], dtype=first.indices.dtype, device=first.indices.device)
    for ten, scalar in zip(tens, scalars):
        out_inds, _ = table.query(scalar)
        out_inds = out_inds.long()
        out_features[out_inds] += ten.features
        out_indices[out_inds] = ten.indices
    res = SparseConvTensor(out_features, out_indices, first.spatial_shape, first.batch_size, 
        benchmark=first.benchmark)
    if count_val == max_num_indices:
        res.indice_dict = tens[max_num_indices_idx].indice_dict
    res.benchmark_record = first.benchmark_record
    res._timer = first._timer 
    res.thrust_allocator = first.thrust_allocator
    return res 

def sparse_add(*tens: SparseConvTensor):
    """reuse torch.sparse. the internal is sort + unique 
    """
    max_num_indices = 0
    max_num_indices_idx = 0
    ten_ths: List[torch.Tensor] = []
    first = tens[0]
    res_shape = [first.batch_size, *first.spatial_shape, first.features.shape[1]]

    for i, ten in enumerate(tens):
        assert ten.spatial_shape == tens[0].spatial_shape
        assert ten.batch_size == tens[0].batch_size
        assert ten.features.shape[1] == tens[0].features.shape[1]
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        ten_ths.append(torch.sparse_coo_tensor(ten.indices.T, ten.features, res_shape, requires_grad=True))
    
    c_th = reduce(lambda x, y: x + y, ten_ths).coalesce()
    c_th_inds = c_th.indices().T.contiguous().int()
    c_th_values = c_th.values()
    assert c_th_values.is_contiguous()

    res = SparseConvTensor(c_th_values, c_th_inds, first.spatial_shape, first.batch_size, 
        benchmark=first.benchmark)
    if c_th_values.shape[0] == max_num_indices:
        res.indice_dict = tens[max_num_indices_idx].indice_dict
    res.benchmark_record = first.benchmark_record
    res._timer = first._timer 
    res.thrust_allocator = first.thrust_allocator
    return res 

def group_mult(features, filters, groups, concat=True):
    features_batch = torch.stack(features.chunk(groups, dim=1), dim=0)
    filters_batch = torch.stack(filters.chunk(groups, dim=0 if FILTER_HWIO else 1), dim=0)
    out_features_batch = torch.bmm(features_batch, filters_batch)
    #out_features = [out_f for out_f in out_features_batch]
    if concat:
        out_features = out_features_batch.permute(1, 0, 2).reshape(out_features_batch.shape[1], -1)
    else:
        out_features = [out_f for out_f in out_features_batch]
    return out_features


def indice_conv_groups(features, filters, indice_pairs, indice_pair_num,
                        num_activate_out, algo, groups, spatial_groups=-1,
                        position_embedding=None, timer=CUDAKernelTimer(False), subm=False):
    if not features.is_contiguous():
        features = features.contiguous()
                    
    out_channel = filters.shape[-1] if FILTER_HWIO else filters.shape[-2]

    filters = filters.reshape(-1, *filters.shape[-2:])
    kv = filters.shape[0]
    kv_center = kv // 2
    
    if subm:
        _filters = filters[kv_center] if FILTER_HWIO else filters[kv_center].T
        out_features = group_mult(features, _filters, groups)
    else:
        out_features = torch.zeros((num_activate_out, out_channel),
                                    dtype=features.dtype,
                                    device=features.device)

    indice_pair_num_cpu = indice_pair_num.cpu().tolist()
    pair_in = indice_pairs[0]
    pair_out = indice_pairs[1]

    if all(x == 0 for x in indice_pair_num_cpu):
        return torch.cat(out_features, dim=-1)

    maxnhot = max(indice_pair_num_cpu)
    for i, nhot in enumerate(indice_pair_num_cpu):
        if subm and i == kv_center:
            continue
        if subm and i > kv_center:
            nhot = indice_pair_num_cpu[kv - i - 1]
        if nhot <= 0:
            continue
        inp_indices = pair_in[i][:nhot].long()
        out_indices = pair_out[i][:nhot].long()
        inp_buffer = features[inp_indices]
        filters_cur = filters[i]  if FILTER_HWIO else filters[i].T
        out_buffer = group_mult(inp_buffer[:nhot], filters_cur, groups)
        out_features.index_add_(0, out_indices, out_buffer)

    return out_features

indice_subm_conv_groups = partial(indice_conv_groups, subm=True)
