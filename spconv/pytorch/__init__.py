import platform
from pathlib import Path

import numpy as np
import torch
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch import functional, ops
from spconv.pytorch.conv import (SparseConv1d, SparseConv2d, SparseConv3d,
                                 SparseConv4d, SparseConvTranspose1d,
                                 SparseConvTranspose2d, SparseConvTranspose3d,
                                 SparseConvTranspose4d, SparseInverseConv1d,
                                 SparseInverseConv2d, SparseInverseConv3d,
                                 SparseInverseConv4d, SubMConv1d, SubMConv2d,
                                 SubMConv3d, SubMConv4d, SPSSConv3d, SPRSConv3d, FocalsConv3d,
                                 SpatialGroupConv3d)
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import (SparseModule, SparseSequential,
                                    assign_name_for_sparse_modules)
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import (SparseMaxPool1d, SparseMaxPool2d,
                                 SparseMaxPool3d, SparseMaxPool4d,
                                 SparseAvgPool3d)
from spconv.pytorch.tables import AddTable, ConcatTable, JoinTable


class ToDense(SparseModule):
    """convert SparseConvTensor to NCHW dense tensor.
    """
    def forward(self, x: SparseConvTensor):
        return x.dense()


class RemoveGrid(SparseModule):
    """remove pre-allocated grid buffer.
    """
    def forward(self, x: SparseConvTensor):
        x.grid = None
        return x
