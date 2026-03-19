import torch
import os
import yaml
from easydict import EasyDict
from .quantize_base import QuantizeBase
from .lsq import LearnableFakeQuantize
from ..utils.hook import PerChannelLoadHook

class Fp16FakeQuantize(LearnableFakeQuantize):
    """This is FP16 Quantization Emulator"""

    def __init__(self, observer, **observer_kwargs):
        super(Fp16FakeQuantize, self).__init__(observer, **observer_kwargs)
        self.dtype = 'FP16'

    def forward(self, X):

        ori_x = X
        ori_dtype = X.dtype
        ori_device = X.device
        if not torch.jit.is_tracing():
            if ori_dtype != torch.float16:
                ori_x = X.to(torch.float16).to(torch.float)
        ori_x = super(Fp16FakeQuantize, self).forward(X)
        if not torch.jit.is_tracing():
            if ori_x.dtype != ori_dtype:
                ori_x = ori_x.to(ori_dtype).to(ori_device)
        return ori_x


    @torch.jit.export
    def extra_repr(self):
        return 'F16 fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List', 
                   self.zero_point if self.ch_axis == -1 else 'List')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(Fp16FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(Fp16FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)
        
class BF16FakeQuantize(LearnableFakeQuantize):
    """This is BF16 Quantization Emulator."""
    def __init__(self, observer, **observer_kwargs):
        super(BF16FakeQuantize, self).__init__(observer, **observer_kwargs)
        self.dtype = 'BF16'

    def forward(self, X):
        ori_x = X
        if isinstance(X, torch.Tensor):
            ori_dtype = X.dtype
            ori_device = X.device
            if not torch.jit.is_tracing():
                if ori_dtype != torch.float16:
                    ori_x = X.to(torch.float16).to(torch.float)
            ori_x = super().forward(X)
            if not torch.jit.is_tracing():
                if ori_x.dtype != ori_dtype:
                    ori_x = ori_x.to(ori_dtype).to(ori_device)
        return ori_x
    
    @torch.jit.export
    def extra_repr(self):
        return 'BF16 fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List[%s]' % str(self.scale.shape),
                   self.zero_point if self.ch_axis == -1 else 'List')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(BF16FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(BF16FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)
