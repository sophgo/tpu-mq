import torch
import math

from tpu_mq.fake_quantize.quantize_base import QuantizeBase
from tpu_mq.utils.hook import PerChannelLoadHook

from tpu_mq.fake_quantize.quantize_base import _version_under_1100
#import ipdb
from scipy.stats import norm
INT4=[-8,
     -7,
     -6,
     -5,
     -4,
     -3,
     -2,
     -1,
     0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,]
NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
AF4=[-1.0, 
     -0.69441008, 
     -0.51243739,
      -0.3736951, 
     -0.25607552, 
     -0.14982478,
     -0.04934812,  
     0.0, 
     0.04273164, 
     0.12934483, 
     0.21961274, 
     0.31675666,
     0.42563882,  
     0.55496234,  
     0.72424863,  
     1.0,
]
FP6E3M2=[
    -28.0000,
    -24.0000,
    -20.0000,
    -16.0000,
    -14.0000,
    -12.0000,
    -10.0000,
    -8.0000,
    -7.0000,
    -6.0000,
    -5.0000,
    -4.0000,
    -3.5000,
    -3.0000,
    -2.5000,
    -2.0000,
    -1.7500,
    -1.5000,
    -1.2500,
    -1.0000,
    -0.8750,
    -0.7500,
    -0.6250,
    -0.5000,
    -0.4375,
    -0.3750,
    -0.3125,
    -0.2500,
    -0.1875,
    -0.1250,
    -0.0625,
    0.0000,
    0.0625,
    0.1250,
    0.1875,
    0.2500,
    0.3125,
    0.3750,
    0.4375,
    0.5000,
    0.6250,
    0.7500,
    0.8750,
    1.0000,
    1.2500,
    1.5000,
    1.7500,
    2.0000,
    2.5000,
    3.0000,
    3.5000,
    4.0000,
    5.0000,
    6.0000,
    7.0000,
    8.0000,
    10.0000,
    12.0000,
    14.0000,
    16.0000,
    20.0000,
    24.0000,
    28.0000
]

FP6E2M3=[
    -7.5,
    -7.0,
    -6.5,
    -6.0,
    -5.5,
    -5.0,
    -4.5,
    -4.0,
    -3.75,
    -3.5,
    -3.25,
    -3.0,
    -2.75,
    -2.5,
    -2.25,
    -2.0,
    -1.875,
    -1.75,
    -1.625,
    -1.5,
    -1.375,
    -1.25,
    -1.125,
    -1.0,
    -0.875,
    -0.75,
    -0.625,
    -0.5,
    -0.375,
    -0.25,
    -0.125,
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
    1.625,
    1.75,
    1.875,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5
]
FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

#NF8 compute
offset=(1-(1/(255*2))+1-(1/(256*2)))*(1/2) #0.9981
v1 = norm.ppf(torch.linspace(offset, 0.5, 129)[:-1]).tolist()
v3 = (-norm.ppf(torch.linspace(offset, 0.5, 128)[:-1])).tolist()
v=v1+v3+[0]
NF8 = torch.Tensor(v)
NF8 = NF8.sort().values
NF8 /= NF8.max()

FLOAT_MAPPING = {"nf4": NF4, "fp4": FP4_BNB, "fp4_e2m1_bnb": FP4_BNB, "fp4_e2m1": FP4_E2M1,"af4":AF4,"nf8":NF8,"int4":INT4, "fp6_e3m2": FP6E3M2, "fp6_e2m3":FP6E2M3}
INT_MAPPING = {"nf4": NF4_BIT, "fp4": FP4_BNB_BIT, "fp4_e2m1_bnb": FP4_BNB_BIT, "fp4_e2m1": FP4_E2M1_BIT,"af4":AF4,"nf8":NF8,"int4":INT4, "fp6_e3m2": FP6E3M2, "fp6_e2m3":FP6E2M3}

# 写一个获取FP8量化所能表示的最大值函数：
def get_flt_max(mode):
    if mode.lower() == "e5m2":
        return float(57344.0) # E5M2所能表示的最大值
    elif mode.lower() == "e4m3":
        return float(448.0) # E4M3所能表示的最大值
    
# 写一个获取FP8量化所能表示的最小值函数：
def get_flt_min(mode):
    if mode.lower() =="e5m2":
        return float(1.5258789E-05) # E5M2所能表示的最小值
    elif mode.lower() == "e4m3":
        return float(1.9531250E-03) #E4M3所能表示的最小值

# 写一个Int量化的转化函数（以支持INT8/INT4量化）：
def quantize_to_integer(tensor, mode, inplace=False):
    # compute tensor min and max values
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    # int8 quantization range 

    nbits = int(mode.split("INT")[1])-1
    q_min = -1*2**nbits
    q_max = (2**nbits)-1

    """
    q_min = -128
    q_max = 127
    if mode == "INT4":
        q_min = -8
        q_max = 7
    """
    # compute scale and zero_point 
    scale = (max_val - min_val) / (q_max - q_min)
    zero_point = q_min - (min_val / scale)
    # Quantize the input tensor using int8 representation
    qtensor = torch.round((tensor / scale) + zero_point)
    # Clamp the values to the int8 range
    qtensor = torch.clamp(qtensor, q_min, q_max)
    # Dequantize the tensor
    dqtensor = scale * (qtensor - zero_point)

    if inplace is True:
        tensor.data.copy_(dqtensor)
        return tensor
    
    return dqtensor

#调用emulator函数计算量化后的权重：
def fpemu_device_fn(tensor, mode, inplace=True, scale=1.0):
    #if "INT8" in mode or "INT4" in mode:
    if "INT" in mode: # 如果输入的mode是INT类型，走这个循环进行整数的量化
        return quantize_to_integer(tensor, mode.split("_")[0], inplace=inplace)

    if tensor.is_cuda : # 如果使用CUDA走这个循环，调用了pytquant中的CUDA函数
        from tpu_mq.FP8_Emulator.pytquant.cuda import fpemu_cuda
        X = fpemu_cuda.FPEmuOp_cuda_per_tensor.apply(tensor, mode, inplace, scale)

    else : # 如果使用CPU走这个循环，调用了pytquant中的CPP函数
        from tpu_mq.FP8_Emulator.pytquant.cpp import fpemu_cpp
        X = fpemu_cpp.FPEmuOp_cpp_per_tensor.apply(tensor, mode, inplace, scale)

    return X


class FPXGROUPFakeQuantize(QuantizeBase):
    """This is fp4 Quantization Emulator..
    """
    def __init__(self, observer, **observer_kwargs):
        super(FPXGROUPFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.load_state_dict_hook = PerChannelLoadHook(self)
        import os
        bench_dtype = os.getenv("BENCH_DTYPE", default="fp6_e3m2")
        bench_groupsize = os.getenv("BENCH_GROUPSIZE", default=128)
        per_chan = os.getenv("BENCH_PERCHAN", default=0)
        #self.data_type="fp6_e3m2"
        self.data_type=bench_dtype
        self.quantile=1.0
        self.return_int=False
        #self.group_size=bench_groupsize
        self.group_size=int(bench_groupsize)
        self.per_chan=int(per_chan)
        self.double_quant=False
        self.double_quant_dtype="E4M3_RNE"
        self.printed=False

    def forward(self, X):
        assert self.data_type in FLOAT_MAPPING, "unexpected data type."
        allow_data = FLOAT_MAPPING[self.data_type]                 #float类型
        allow_data_bit = INT_MAPPING[self.data_type]               #int类型
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            if self.group_size > 0:
                _scale = X.abs().max(1)[0] * self.quantile/ max(allow_data)
            else:
                _scale = X.abs().max() * self.quantile/ max(allow_data)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            Xs=X.shape[1]
            if not self.printed:
                print(f'Group size is {self.group_size} datatype is {self.data_type} double quant is {self.double_quant}')
                self.printed=True
            if self.group_size > 0:
                X= X.reshape((-1, self.group_size))
                if self.per_chan == 1:
                    X= X.reshape((-1, Xs))
                _scale = X.abs().max(1)[0] * self.quantile/ max(allow_data)
            else:
                X= X.flatten()
                if self.per_chan == 1:
                    X= X.reshape((-1, Xs))
                _scale = X.abs().max() * self.quantile/ max(allow_data)
            #scale quantize
            if self.double_quant:
                scalemax = torch.max(abs(torch.flatten(_scale.detach()))) #求出_scale的max
                _scale1 = get_flt_max("e4m3") / scalemax
                _scale1 = torch.tensor(6.55e+04) if _scale1.item() > 3.275e+04 else _scale1
                _scale1 = _scale1.to(self.scale.device)
                _scale = fpemu_device_fn(_scale, mode=self.double_quant_dtype, inplace=False, scale=_scale1) #返回per tensor方式计算的量化权重

            if self.group_size > 0:
                _scale.unsqueeze_(dim=-1)
            X = X/_scale
            # if self.data_type.lower=="nf4":
            #     cdf_values = [norm.cdf(x) for x in allow_data]
            #     intermediate_cdf_values = [(cdf_values[i] + cdf_values[i+1]) / 2 for i in range(len(allow_data) - 1)]
            #     mid_data = norm.ppf(intermediate_cdf_values)
            # else:
            #     mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
            mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
            q_X= torch.zeros_like(X)
            for i in range(len(allow_data)):
                data = allow_data_bit[i] if self.return_int else allow_data[i]
                if i == 0:
                    q_X += torch.where(X <= mid_data[i], data, 0)
                elif i == len(allow_data) - 1:
                    q_X += torch.where(X > mid_data[i - 1], data, 0)
                else:
                    q_X += torch.where((mid_data[i - 1] < X) & (X <= mid_data[i]), data, 0)
            # if self.return_int:
            #     return q_X.type(torch.int8), _scale.type(torch.float), None
            X=q_X * _scale
            X=X.reshape((-1,Xs))
        return X

    @torch.jit.export
    def extra_repr(self):
        allow_data = FLOAT_MAPPING[self.data_type] 
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   min(allow_data), max(allow_data),
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List', 
                   self.zero_point if self.ch_axis == -1 else 'List')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FPXGROUPFakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super(FPXGROUPFakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)
