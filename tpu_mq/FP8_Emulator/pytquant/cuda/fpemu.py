#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import math
from torch import nn
from torch.autograd import Function
import torch
import numpy
import fpemu_cuda

from enum import Enum
from torch.onnx import symbolic_helper

torch.manual_seed(42)

"""
    NONE
    E5M2_RTZ
    E5M2_STOCHASTIC
    E5M2_RNE
    E5M2_RNAZ
    E5M2_RNTZ
    E5M2_RPINF
    E5M2_RNINF
    E5M2_DAZ_STOCHASTIC
    E5M2_DAZ_RNE
    E5M2_DAZ_RNAZ
    E5M2_DAZ_RNTZ
    BFLOAT16_STOCHASTIC
    BFLOAT16_RNE
    FLOAT16_RNE
    FLOAT16_STOCHASTIC
    FLOAT16_DAZ_RNE
    E4M3_RNE
    E4M3_STOCHASTIC
"""

class FPEmuOp_cuda_per_tensor(Function):
    @staticmethod
    def forward(ctx, input, mode='NONE', inplace=False, scale=torch.tensor([1.0]), zero_point=torch.tensor([0.0]), quant_min=float(1.5258789E-05), quant_max=float(57344.0), blocknorm=False, blocksize=1):
        scale = float(scale.item())
        if mode == 'NONE' :
            ctx.mark_dirty(input)
            return input
        else :
            if input.is_sparse :
                input = input.coalesce()
                size = input.values().nelement()
                if inplace == True:
                    outputs = fpemu_cuda.forward(input._values().contiguous(), mode, size, inplace, scale, blocknorm, blocksize)
                    output = input
                else :
                    outputs = fpemu_cuda.forward(input._values().contiguous(), mode, size, inplace, scale, blocknorm, blocksize)
                    output = torch.sparse.FloatTensor(input.indices(), outputs[0], input.size())
            else :
                size = input.nelement()
                outputs = fpemu_cuda.forward(input.contiguous(), mode, size, inplace, scale, blocknorm, blocksize)
                output = outputs[0]

            if inplace == True:
                ctx.mark_dirty(input)
            return output

    @staticmethod
    def backward(ctx, output_grad):
        # straight-through estimator
        return output_grad, None, None, None, None

    @staticmethod
    def symbolic(g, input, mode='NONE', inplace=False, scale=torch.tensor([1.0]), zero_point=torch.tensor([0.0]), quant_min=float(1.5258789E-05), quant_max=float(57344.0), blocknorm=False, blocksize=1):

        output = g.op("tpu_custom::FPEmuOp_per_tensor", input, scale, zero_point, quant_min_f=quant_min, quant_max_f=quant_max)

        input_shape = symbolic_helper._get_tensor_sizes(input)
        if input_shape is not None and hasattr(input.type(), 'with_sizes'):
            output_type = input.type().with_sizes(input_shape)
            output.setType(output_type)

        return output

def get_flt_max(mode):
    if mode.lower() == "e5m2":
        return float(57344.0) 
    elif mode.lower() == "e4m3":
        return float(448.0)

def get_flt_min(mode):
    if mode.lower() =="e5m2":
        return float(1.5258789E-05) # Min Subnormal
    elif mode.lower() == "e4m3":
        return float(1.9531250E-03)

def _forward_per_channel(ctx, input, mode='NONE', inplace=False, scale=torch.tensor([1.0]), zero_point=torch.tensor([0.0]), quant_min=float(1.5258789E-05), quant_max=float(57344.0), blocknorm=False, blocksize=1):
    scale = float(scale.item())
    if mode == 'NONE' :
        ctx.mark_dirty(input)
        return input
    else :
        if input.is_sparse :
            input = input.coalesce()
            size = input.values().nelement()
            if inplace == True:
                outputs = fpemu_cuda.forward(input._values().contiguous(), mode, size, inplace, scale, blocknorm, blocksize)
                output = input
            else :
                outputs = fpemu_cuda.forward(input._values().contiguous(), mode, size, inplace, scale, blocknorm, blocksize) 
                output = torch.sparse.FloatTensor(input.indices(), outputs[0], input.size())
        else :
            # input = input.cpu()
            size = input.nelement()
            outputs = fpemu_cuda.forward(input.contiguous(), mode, size, inplace, scale, blocknorm, blocksize)
            output = outputs[0]

        if inplace == True:
            ctx.mark_dirty(input)

        return output


class FPEmuOp_cuda_per_channel(Function):

    @staticmethod
    def forward(ctx, X, mode='NONE', inplace=False, scale=torch.tensor([1.0]), zero_point=torch.tensor([0.0]), quant_min=float(1.5258789E-05), quant_max=float(57344.0), blocknorm=False, blocksize=1):
        tensor_q = torch.zeros_like(X).reshape((-1,X.shape[-1]))
        work_mode = mode
        if mode.startswith('E4M3'):
            max_=get_flt_max('e4m3')
            min_=get_flt_min('e4m3')
        elif mode.startswith('E5M2'):
            max_=get_flt_max('e5m2')
            min_=get_flt_min('e5m2')
        scaling_method = "max"
        channels = tensor_q.shape[0]
        for c in range(channels):
            sub_tensor = X.select(0, c).detach().flatten()  # hack !!!! per_channel for matmul weight, the weight should be in [oc ic], for conv the weight should be oc, ic, kh, kw
            if scaling_method.lower() == "mean":
                mean = torch.mean(abs(sub_tensor))
                mean = abs(mean) if abs(mean) > 1e-5 else min_
                if abs(mean) > 0.0:
                        _scale = min_ / abs(mean)
            elif scaling_method.lower() == "max":
                vmax = torch.max(abs(sub_tensor))
                _scale = max_ / vmax
                _scale = torch.tensor(6.55e+04) if _scale.item() > 3.275e+04 else _scale
            else:
                _scale = torch.tensor(1.0)
            # self.scale.copy_(_scale)
            sub_tensor = _forward_per_channel(ctx, sub_tensor, mode=work_mode, inplace=False, scale=_scale, zero_point=torch.tensor([0.0]), quant_min=quant_min, quant_max=quant_max)
            tensor_q.select(0, c).data.copy_(sub_tensor)

        return tensor_q.reshape(X.shape)

    @staticmethod
    def backward(ctx, output_grad):
        # straight-through estimator
        return output_grad, None, None, None, None

    @staticmethod
    def symbolic(g, input, mode='NONE', inplace=False, scale=torch.tensor([1.0]), zero_point=torch.tensor([0.0]), quant_min=float(1.5258789E-05), quant_max=float(57344.0), blocknorm=False, blocksize=1):
        output = g.op("tpu_custom::FPEmuOp_per_channel", input, scale, zero_point, quant_min_f=quant_min, quant_max_f=quant_max)

        input_shape = symbolic_helper._get_tensor_sizes(input)
        if input_shape is not None and hasattr(input.type(), 'with_sizes'):
            output_type = input.type().with_sizes(input_shape)
            output.setType(output_type)

        return output
