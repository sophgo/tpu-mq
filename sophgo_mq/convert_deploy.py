import io
import json
import os.path as osp
import os
import pdb
import copy
import time
import subprocess
import numpy as np
import torch
from torch.fx import GraphModule
import onnx
from onnxsim import simplify
import sophgo_mq.custom_symbolic_opset  # noqa: F401
import sophgo_mq.fusion_method          # noqa: F401
from sophgo_mq.utils import deepcopy_graphmodule
from .OnnxOpt import onnx_opt
from sophgo_mq.utils.logger import logger
from sophgo_mq.utils.registry import (
    NET_DEPLOY_FUNCTION,
    FUSED_MODULE_CONVERT_FUNCTION,
    register_deploy_function
)
from sophgo_mq.deploy import (
    remove_fakequantize_and_collect_params,
    remove_fakequantize_and_collect_params_tf,
    remove_fakequantize_and_collect_params_flt,
    remove_fakequantize_and_collect_params_sophgo,
)

from sophgo_mq.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize,
    TqtFakeQuantize,
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
    E4M3FakeQuantize,
    E5M2FakeQuantize,
    GPTQFakeQuantize,
    FP4FakeQuantize,
    GPTQFP4FakeQuantize,
    FP4GROUPFakeQuantize,
    FP4GROUPFakeQuantize1
)
# from utils.mlir_shell import mlir_opt_for_top, mlir_lowering, mlir_to_model, f32_blobs_compare
# from tools.model_runner import mlir_inference, free_mlir_module
import tpu_mlir
from tools.model_transform import model_transform_func
import torchvision.transforms as TorchTransforms
from utils.mlir_parser import MlirParser

import subprocess
import shlex 

FP8_FAKEQUANTIZER = [E4M3FakeQuantize, E5M2FakeQuantize]
FP4_FAKEQUANTIZER = [FP4FakeQuantize, GPTQFP4FakeQuantize, FP4GROUPFakeQuantize, FP4GROUPFakeQuantize1]
INT_FAKEQUANTIZER = [LearnableFakeQuantize, NNIEFakeQuantize, FixedFakeQuantize, DoReFaFakeQuantize, DSQFakeQuantize, PACTFakeQuantize,\
                     TqtFakeQuantize, AdaRoundFakeQuantize, QDropFakeQuantize, GPTQFakeQuantize]

__all__ = ['convert_deploy']

param_to_idx_dict = {}

model_onnx_mem = None
@register_deploy_function("CNN")
def convert_merge_bn(model: GraphModule, **kwargs):
    # print('wlog before convert_merge_bn, model.named_modules:', dict(model.named_modules())[''])
    # print('wlog before convert_merge_bn, model.graph:', model.graph)
    # logger.info("Merge BN for deploy.")
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    for node in nodes:
        if node.op == 'call_module':
            if node.target in modules and type(modules[node.target]) in FUSED_MODULE_CONVERT_FUNCTION:
                FUSED_MODULE_CONVERT_FUNCTION[type(modules[node.target])](model, node)
    # print('wlog after convert_merge_bn, model.named_modules:', dict(model.named_modules())[''])
    # print('wlog after convert_merge_bn, model.graph:', model.graph)


@register_deploy_function("CNN")
def convert_onnx(model: GraphModule, input_shape_dict, dummy_input, onnx_model_path, **kwargs):
    # pt_file_name = onnx_model_path.split('.')
    # pt_file_name[-1] = 'pt'
    #torch.save(model, '.'.join(pt_file_name))
    export_to_mem = 'not_gen_bmodel' in kwargs and kwargs['not_gen_bmodel']
    if not export_to_mem:
        logger.info("Export to onnx, onnx_model_path:{}".format(onnx_model_path))
    model = model.cpu()
    output_names = kwargs.get('output_names', [])
    dynamic_axes = kwargs.get('dynamic_axes', {})
    input_names = kwargs.get('input_names', [])
    if dummy_input is None:
        device = next(model.parameters()).device
        dummy_input = {name: torch.rand(shape).to(device) for name, shape in input_shape_dict.items()}
        input_names = list(dummy_input.keys())
        dummy_input = tuple(dummy_input.values())
    # Per-channel QuantizeLinear and DequantizeLinear is supported since opset 13
    opset_version = 13 if kwargs.get('deploy_to_qlinear', False) else 11
    # opset_version = 18

    # open all fake quant node to export
    if isinstance(model, torch.fx.graph_module.GraphModule):
        if not export_to_mem:
            print(">>>>> print graphmodule before export", model)
        # print(">>>>> print graph before export")
        # model.graph.print_tabular()
        for name, submodule in model.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantizeBase):
                if submodule.only_enable_observer == True:
                    submodule.only_enable_observer = False
                submodule.disable_observer()
                submodule.enable_fake_quant()
            class_of_submodule = submodule.__class__
            if class_of_submodule in FP8_FAKEQUANTIZER:
                quant_mode = "FP8"
    if export_to_mem:
        onnx_model_path = io.BytesIO()

    simplified_model = None
    with torch.no_grad():
        try:
            from torch.onnx.utils import ONNXCheckerError
            try:
                torch.onnx.export(model, dummy_input, onnx_model_path,
                                input_names=input_names,
                                output_names=output_names,
                                opset_version=opset_version,
                                dynamic_axes=dynamic_axes,
                                do_constant_folding=True,
                                custom_opsets={'Sophgo_custom' : opset_version})
            except ONNXCheckerError:
                pass
        except ImportError:
            ### torch 1.13 and 2.0.1
            s0 = time.time()
            torch.onnx.export(model, dummy_input, onnx_model_path,
                                input_names=input_names,
                                output_names=output_names,
                                opset_version=opset_version,
                                do_constant_folding=True,
                                custom_opsets={'Sophgo_custom' : opset_version})
            print(f'torch.onnx.export time:{time.time() - s0}')
            if export_to_mem:
                tmp_model = onnx.load_model_from_string(onnx_model_path.getvalue())
            else:
                tmp_model = onnx.load(onnx_model_path)
            simplified_model, check = simplify(tmp_model)
            if not export_to_mem:
                onnx.save_model(simplified_model, onnx_model_path)

    if export_to_mem:
        model_onnx = simplified_model if simplified_model is not None else onnx_model_path
    else:
        model_onnx = onnx.load(onnx_model_path)
    onnx.checker.check_model(model_onnx)
    model_onnx = onnx.shape_inference.infer_shapes(model_onnx)
    if export_to_mem:
        global model_onnx_mem
        model_onnx_mem = model_onnx
    else:
        os.system(f"rm -f {onnx_model_path}")
        onnx.save(model_onnx, onnx_model_path)


@register_deploy_function("Transformer")
def convert_onnx(model: GraphModule, input_shape_dict, dummy_input, onnx_model_path, **kwargs):
    pt_file_name = onnx_model_path.split('.')
    pt_file_name[-1] = 'pt'
    #torch.save(model, '.'.join(pt_file_name))
    logger.info("Export to onnx, onnx_model_path:{}".format(onnx_model_path))
    model = model.cpu()
    output_names = kwargs.get('output_names', [])
    dynamic_axes = kwargs.get('dynamic_axes', {})
    input_names = kwargs.get('input_names', [])
    if dummy_input is None:
        device = next(model.parameters()).device
        dummy_input = {name: torch.rand(shape).to(device) for name, shape in input_shape_dict.items()}
        input_names = list(dummy_input.keys())
        dummy_input = tuple(dummy_input.values())
    # Per-channel QuantizeLinear and DequantizeLinear is supported since opset 13
    # opset_version = 11 if kwargs.get('deploy_to_qlinear', False) else 13
    opset_version = 18

    quant_mode = 'INT8'
    # open all fake quant node to export
    if isinstance(model, torch.fx.graph_module.GraphModule):
        # print(">>>>> print graphmodule before export", model)
        # print(">>>>> print graph before export")
        # model.graph.print_tabular()
        for name, submodule in model.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantizeBase):
                submodule.disable_observer()
                submodule.enable_fake_quant()
            class_of_submodule = submodule.__class__
            if class_of_submodule in FP8_FAKEQUANTIZER:
                quant_mode = "FP8"
    # skip FP8 fakequant onnx because of the export problem
    if quant_mode != "FP8":
        with torch.no_grad():
            try:
                from torch.onnx.utils import ONNXCheckerError
                try:
                    torch.onnx.export(model, dummy_input, onnx_model_path,
                                    input_names=input_names,
                                    output_names=output_names,
                                    opset_version=opset_version,
                                    dynamic_axes=dynamic_axes,
                                    do_constant_folding=True,
                                    custom_opsets={'Sophgo_custom' : opset_version})
                except ONNXCheckerError:
                    pass
            except ImportError:
                ### torch 1.13 and 2.0.1
                torch.onnx.export(model, dummy_input, onnx_model_path,
                                    input_names=input_names,
                                    output_names=output_names,
                                    opset_version=opset_version,
                                    do_constant_folding=True,
                                    custom_opsets={'Sophgo_custom' : opset_version})
                tmp_model = onnx.load(onnx_model_path)
                simplified_model, check = simplify(tmp_model)
                onnx.save_model(simplified_model, onnx_model_path)
    # if it is F8 fakequant, load the existed onnx with INT8 fakequant
    model_onnx = onnx.load(onnx_model_path)
    onnx.checker.check_model(model_onnx)
    model_onnx = onnx.shape_inference.infer_shapes(model_onnx)
    os.system(f"rm -f {onnx_model_path}")
    onnx.save(model_onnx, onnx_model_path)


def export_qtable_tf(context_filename, model_name, output_path, quant_mode):
        print("导出qtable")
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[quant_mode]
        file_h.close()
        q_table = osp.join(output_path, '{}_q_table_from_sophgo_mq_{}'.format(model_name, quant_mode))
        with open(q_table, 'w') as f:
            f.write("# qtable from sophgo_mq\n")
            f.write("# op_name  quantize_mode\n") # match the colnames of qtable in tpu-mlir
            for name,value in blob_range.items():
                if 'quant_type' in value:
                    quant_type = value['quant_type']
                    if 'threshold' in value:
                        f.write("{} {}\n".format(name, dtype))
                        # f.write("{} {}\n".format(name[:-2], quant_type))
                    else:
                        f.write("{} {}\n".format(name, quant_type))
                    continue
                if value['only_observer']==0 or value['only_observer']==1:
                    dtype = 'F32'
                    if 'threshold' in value:
                    # f.write("{} {}\n".format(name[:-2], dtype))
                        f.write("{} {}\n".format(name, dtype))
                    else:
                        f.write("{} {}\n".format(name, dtype))
                    continue
                dtype ='INT8'
                if value['bit'] == 4:
                    dtype = 'INT4'
                elif value['type'] == 'uint':
                    dtype = 'UINT8'
                if 'threshold' in value:
                    # f.write("{} {}\n".format(name[:-2], dtype))
                    f.write("{} {}\n".format(name, dtype))
                else:
                    f.write("{} {}\n".format(name, dtype))

def export_qtable(context_filename, model_name, output_path, quant_mode):
        print("导出qtable")
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[quant_mode]
        file_h.close()
        q_table = osp.join(output_path, '{}_q_table_from_sophgo_mq_{}'.format(model_name, quant_mode))
        with open(q_table, 'w') as f:
            f.write("# qtable from sophgo_mq\n")
            f.write("# op_name  quantize_mode\n") # match the colnames of qtable in tpu-mlir
            for name,value in blob_range.items():
                if 'quant_type' in value:
                    quant_type = value['quant_type']
                    if quant_type in ['None', 'BF16_to_INT8']:
                        continue
                    if 'threshold' in value and quant_type != 'BF16':
                        f.write("{} {}\n".format(name[:-2], quant_type))
                    else:
                        f.write("{} {}\n".format(name, quant_type))
                    continue
                dtype = 'INT8'
                if value['bit'] == 4:
                    dtype = 'INT4'
                elif value['type'] == 'uint':
                    dtype = 'UINT8'
                if 'threshold' in value:
                    f.write("{} {}\n".format(name[:-2], dtype))
                else:
                    f.write("{} {}\n".format(name, dtype))


@register_deploy_function("Transformer")
def deploy_qparams_Academic_NLP(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Academic_NLP.")
    node_name_only_observer=[]
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if submodule.only_enable_observer==True:
                node_name_only_observer.append(name)
    quant_mode = "INT8"
    for name, submodule in model.named_modules():
        class_of_submodule = submodule.__class__
        if class_of_submodule in FP8_FAKEQUANTIZER:
            quant_mode = "FP8"
    cali_mode = "Academic_NLP"
    if quant_mode == "FP8":
        remove_fakequantize_and_collect_params_flt(onnx_model_path, model_name, backend='Academic_NLP')
        print("导出calitable")
        output_path = osp.dirname(onnx_model_path)
        context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[cali_mode+"_Float"]
        file_h.close()
        cali_table = osp.join(output_path, '{}_float_cali_table_from_sophgo_mq_Academic_NLP'.format(model_name))
        fp8_header = "sophgo_mq-fp8"
        with open(cali_table, 'w') as f:
            f.write(f"# work_mode:{fp8_header} #Automatically generated, do not modify, work_mode choice:[E4M3_RNE, E5M2_RNE]\n")
            f.write("#       op_name        threshold        min        max\n")
            weight_scale_fp8 = []
            for name,value in blob_range.items():
                if 'threshold' in value:
                    tmpstr = "{}     {:.7f}     {:.7f}     {:.7f}\n".format(name[:-2], value['threshold'], value['min'], value['max'])
                    if name.endswith('_fp8'):
                        f.write(tmpstr)
                    else:
                        f.write("{}     {:.7f}     {:.7f}     {:.7f}\n".format(name, value['threshold'], value['min'], value['max']))
                else:
                    tmpstr = "{} {} {} {} {}\n".format(name, len(value['step']), ' '.join(['{:.7f}'.format(i) for i in value['step']]),
                            len(value['zero_point']), ' '.join([str(i) for i in value['zero_point']]))
                    if name.endswith('_weight_fp8') or name.endswith('_bias_fp8'):
                        weight_scale_fp8.append(tmpstr)
                    else:
                        f.write(tmpstr)
            f.write('#weight_scale_fp8\n')
            for i in weight_scale_fp8:
                f.write(i)
        cali_mode_new = cali_mode + '_Float'
        export_qtable_tf(context_filename, model_name, output_path, cali_mode_new)
    else:
        remove_fakequantize_and_collect_params_tf(onnx_model_path, model_name, backend='Academic_NLP',node_name_only_observer=node_name_only_observer)
        remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='Academic_NLP')
        print("导出calitable")
        output_path = osp.dirname(onnx_model_path)
        context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())["Academic_NLP"]
        file_h.close()
        cali_table = osp.join(output_path, '{}_cali_table_from_sophgo_mq_Academic_NLP'.format(model_name))
        work_mode = kwargs.get('work_mode', 'QAT_all_int8')
        if work_mode not in  ['QAT_all_int8', 'int4_and_int8_mix', 'int4_and_int8_mix_no_fc']:
            print('QAT_all_int8 not in [QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc],set to QAT_all_int8')
            work_mode = 'QAT_all_int8'
        with open(cali_table, 'w') as f:
            f.write(f"# work_mode:{work_mode} #Automatically generated, do not modify, work_mode choice:[QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc]\n")
            f.write("# op_name    threshold    min    max\n")
            weight_scale = []
            int4_th = []
            for name,value in blob_range.items():
                if 'threshold' in value:
                    tmpstr = "{} {:.7f} {:.7f} {:.7f}\n".format(name[:-2], value['threshold'], value['min'], value['max'])
                    if name.endswith('_4'):
                        int4_th.append(tmpstr)
                    elif name.endswith('_8'):
                        f.write(tmpstr)
                    else:
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(name, value['threshold'], value['min'], value['max']))
                else:
                    tmpstr = "{} {} {} {} {}\n".format(name, len(value['step']), ' '.join(['{:.7f}'.format(i) for i in value['step']]),
                            len(value['zero_point']), ' '.join([str(i) for i in value['zero_point']]))
                    if name.endswith('_weight') or name.endswith('_bias'):
                        weight_scale.append(tmpstr)
                    else:
                        f.write(tmpstr)
            f.write('#int4_th\n')
            for i in int4_th:
                f.write(i)
            f.write('#weight_scale\n')
            for i in weight_scale:
                f.write(i)
        export_qtable_tf(context_filename, model_name, output_path, cali_mode)

@register_deploy_function("CNN")
def deploy_qparams_sophgo_tpu(model: GraphModule, onnx_model_path, model_name, quant_type_dict, **kwargs):
    logger.info("Extract qparams for sophgo_tpu.")
    cali_mode = "sophgo_tpu"
    global model_onnx_mem
    export_to_mem = 'not_gen_bmodel' in kwargs and kwargs['not_gen_bmodel']
    if not export_to_mem:
        model_onnx_mem = None
    model_onnx_mem = remove_fakequantize_and_collect_params_sophgo(onnx_model_path, model_name, quant_type_dict, model_onnx_mem)
    # print("导出calitable")
    output_path = osp.dirname(onnx_model_path)
    context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
    file_h = open(context_filename, "r")
    blob_range = json.loads(file_h.read())["sophgo_tpu"]
    file_h.close()
    cali_table = osp.join(output_path, '{}_cali_table_from_sophgo_mq_sophgo_tpu'.format(model_name))
    work_mode = kwargs.get('work_mode', 'QAT_all_int8')
    if work_mode not in  ['QAT_all_int8', 'int4_and_int8_mix', 'int4_and_int8_mix_no_fc']:
        print('QAT_all_int8 not in [QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc],set to QAT_all_int8')
        work_mode = 'QAT_all_int8'
    with open(cali_table, 'w') as f:
        f.write(f"# work_mode:{work_mode} #Automatically generated, do not modify, work_mode choice:[QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc]\n")
        f.write("# op_name    threshold    min    max\n")
        weight_scale = []
        int4_th = []
        for name,value in blob_range.items():
            if 'threshold' in value:
                tmpstr = "{} {:.7f} {:.7f} {:.7f}\n".format(name[:-2], value['threshold'], value['min'], value['max'])
                if name.endswith('_4'):
                    int4_th.append(tmpstr)
                elif name.endswith('_8'):
                    f.write(tmpstr)
                else:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(name, value['threshold'], value['min'], value['max']))
            else:
                tmpstr = "{} {} {} {} {}\n".format(name, len(value['step']), ' '.join(['{:.7f}'.format(i) for i in value['step']]),
                        len(value['zero_point']), ' '.join([str(i) for i in value['zero_point']]))
                if name.endswith('_weight') or name.endswith('_bias'):
                    weight_scale.append(tmpstr)
                else:
                    f.write(tmpstr)
        f.write('#int4_th\n')
        for i in int4_th:
            f.write(i)
        f.write('#weight_scale\n')
        for i in weight_scale:
            f.write(i)
    export_qtable(context_filename, model_name, output_path, cali_mode)

def get_quant_type_from_fakequant_type(model: GraphModule):
    r"""
    Given GraphModule, Traverse each fakequant node within it,
    obtain the quantization type based on the class of the fakequant node.
    """
    quant_type_dict = {}
    for name, submodule in model.named_modules(remove_duplicate=False):
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if submodule.only_enable_observer:
                quant_type_dict[name] = 'F32'
            else:
                class_of_submodule = submodule.__class__
                if class_of_submodule in INT_FAKEQUANTIZER:
                    qmax = submodule.quant_max
                    qmin = submodule.quant_min
                    bit = int(np.log2(qmax-qmin+1))
                    if (bit == 8 and qmin < 0):
                        quant_type_dict[name] = 'INT8'
                    elif (bit == 8 and qmin ==0):
                        quant_type_dict[name] = 'UINT8'
                    elif (bit == 4 and qmin < 0):
                        quant_type_dict[name] = 'INT4'
                elif class_of_submodule in FP4_FAKEQUANTIZER:
                    quant_type_dict[name] = 'FP4'
                elif class_of_submodule in FP8_FAKEQUANTIZER:
                    quant_type_dict[name] = 'FP8'
                else:
                    quant_type_dict[name] = 'None'

    return quant_type_dict


def convert_deploy(model: GraphModule, net_type='CNN',
                   input_shape_dict=None, dummy_input=None, output_path='./',
                   model_name='sophgo_mq_qmodel', deploy_to_qlinear=False, mlir_deploy=False, chip="BM1690", val_loader=None, **extra_kwargs):
    r"""Convert model to onnx model and quantization params depends on backend.

    Args:
        model (GraphModule): GraphModule prepared qat module.
        backend_type (BackendType): specific which backend should be converted to.
        input_shape_dict (dict): keys are model input name(should be forward function
                                 params name, values are list of tensor dims)
        output_path (str, optional): path to save convert results. Defaults to './'.
        model_name (str, optional): name of converted onnx model. Defaults to 'sophgo_mq_qmodel'.

    >>> note on input_shape_dict:
        example: {'input_0': [1, 3, 224, 224]
                'input_1': [1, 3, 112, 112]
                }
        while forward function signature is like:
                def forward(self, input_0, input_1):
                    pass
    """
    batch_size = next(iter(input_shape_dict.values()))[0]
    quant_type_dict = get_quant_type_from_fakequant_type(model)
    kwargs = {
        'input_shape_dict': input_shape_dict,
        'dummy_input': dummy_input,
        'output_path': output_path,
        'model_name': model_name,
        'onnx_model_path': osp.join(output_path, '{}.onnx'.format(model_name)),
        'deploy_to_qlinear': deploy_to_qlinear,
        'quant_type_dict': quant_type_dict
    }
    kwargs.update(extra_kwargs)
    deploy_model = deepcopy_graphmodule(model)
    for convert_function in NET_DEPLOY_FUNCTION[net_type]:
        s0 = time.time()
        convert_function(deploy_model, **kwargs)
        print(f'convert_function:{convert_function.__name__}, time:{time.time() - s0}')

    if mlir_deploy and chip != 'academic':
        if val_loader == None:
            print(f'VAL Loader not set for deploy!')
            return
        export_to_mm = 'not_gen_bmodel' in kwargs and kwargs['not_gen_bmodel']
        mlir_scale = '1,1,1'
        mlir_mean = '0,0,0'
        transforms = val_loader.dataset.transform.transforms
        for i, transform in enumerate(transforms):
            if isinstance(transform, TorchTransforms.Normalize) and isinstance(transforms[i - 1], TorchTransforms.ToTensor):
                mean_np = np.array(transform.mean)
                std_np = np.array(transform.std)
                mlir_scale=1/(255*std_np)
                mlir_mean=mean_np/(std_np*mlir_scale)
                mlir_scale = ','.join([str(round(i,4)) for i in mlir_scale.tolist()])
                mlir_mean = ','.join([str(round(i,4)) for i in mlir_mean.tolist()])
                print('mlir_mean:', mlir_mean, 'mlir_scale:', mlir_scale)

        if not export_to_mm:
            onnx_filename = os.path.join(output_path, '{}_deploy_model.onnx'.format(model_name))
        else:
            global model_onnx_mem
            onnx_filename = model_onnx_mem
        shape_str_list = []
        for name in input_shape_dict:
            shape_str = ','.join([str(i) for i in input_shape_dict[name]])
            shape_str_list.append(f'[{shape_str}]')
        shape_str_list = ','.join(shape_str_list)
        
        if export_to_mm:
            s0 = time.time()
            model_transform_func(f'{model_name}_qat', 
                                onnx_filename, list(input_shape_dict.values()), 
                                mlir_scale, mlir_mean,
                                f'{model_name}_qat.mlir')
        else:
            cmd_str = f"model_transform.py \
            --model_name {model_name}_qat \
            --model_def {onnx_filename} \
            --input_shapes [{shape_str_list}] \
            --mean {mlir_scale} \
            --scale {mlir_mean} \
            --keep_aspect_ratio \
            --pixel_format rgb --debug \
            --mlir {model_name}_qat.mlir"
            print('model_transform cmd_str:', cmd_str)
            os.system(cmd_str)
        work_path = output_path
        if output_path.startswith('/dev/shm/'):
            cmd_str = f'cp {model_name}_qat* /dev/shm/'
            os.system(cmd_str)
            cmd_str = 'cp input_data_0.npz /dev/shm/'
            os.system(cmd_str)
            cmd_str = 'cp layer_outputs_0.npz /dev/shm/'
            os.system(cmd_str)
            cmd_str = f'cp {model_name}_cali_table_from_sophgo_mq_sophgo_tpu /dev/shm/'
            os.system(cmd_str)
            cmd_str = f'cp {model_name}_q_table_from_sophgo_mq_sophgo_tpu /dev/shm/'
            os.system(cmd_str)
            print(f'copy files to {output_path}')
            os.system(f'ls {output_path}')
            work_path='/dev/shm/'

        print(f'model_transform_func time:{time.time() - s0}')

        if 'mlir_deploy_debug_onlytransform' not in extra_kwargs or extra_kwargs['mlir_deploy_debug_onlytransform'] == False:
            calibration_table = os.path.join(work_path, '{}_cali_table_from_sophgo_mq_sophgo_tpu'.format(model_name))
            test_input_file = os.path.join(work_path, 'input_data_0.npz')            
            test_reference_file = os.path.join(work_path, 'layer_outputs_0.npz')
            quantize_table = ''
            not_gen_bmodel = ''
            if export_to_mm:
                test_input, test_reference = '', ''
                not_gen_bmodel = '--not_gen_bmodel'
            else:
                test_input = f'--test_input {test_input_file}'
                test_reference = f'--test_reference {test_reference_file}'
            if batch_size == 1 or not os.path.exists(test_input_file) or not os.path.exists(test_reference_file):
                test_input, test_reference = '', ''

            quantize_mode = 'INT8'
            quantize_str = 'int8_sym'
            if 'bf16_mix_prec' in kwargs and kwargs['bf16_mix_prec']:
                quantize_table = os.path.join(work_path, '{}_q_table_from_sophgo_mq_sophgo_tpu'.format(model_name))
                quantize_table = f'--quantize_table {quantize_table}'
                quantize_mode = 'BF16'
                quantize_str = 'bf16'
            bmodel_ext = 'cvimodel' if chip in ['MARS3', 'CV183X', 'CV182X', 'CV181X', 'CV180X', 'CV186X'] else 'bmodel'
            cmd_str = f"model_deploy.py \
            --mlir {model_name}_qat.mlir \
            --quantize {quantize_mode} \
            --calibration_table {calibration_table} {quantize_table} \
            --chip {chip} {test_input} {test_reference}\
            --fazzy_match \
            --tolerance 0.99,0.90 --debug {not_gen_bmodel}\
            --model {model_name}_qat_{chip.lower()}_{quantize_str}.{bmodel_ext}"
            print('model_deploy cmd_str:', cmd_str)
            s0 = time.time()
            if output_path.startswith('/dev/shm/'):
                cmd_args = shlex.split(cmd_str)
                p = subprocess.Popen(cmd_args, cwd=output_path)
                p.wait()
                print(f'model_deploy time:{time.time() - s0}')
                return f'{output_path}{model_name}_qat_{chip.lower()}_{quantize_str}_tpu.mlir'
            else:
                os.system(cmd_str)
                print(f'model_deploy time:{time.time() - s0}')
                return f'{model_name}_qat_{chip.lower()}_{quantize_str}_tpu.mlir'

def lower_net(model_name, chip, output_path, log_out = False):
    cali_table = os.path.join(output_path, '{}_cali_table_from_sophgo_mq_sophgo_tpu'.format(model_name))
    cmd_str = f"tpuc-opt {model_name}_qat_origin.mlir --shape-infer --canonicalize --extra-optimize -o {model_name}_qat.mlir"
    if log_out:
        print(f'cmd_str:{cmd_str}')
    s0 = time.time()
    os.system(cmd_str)
    if log_out:
        print(f'convert origin mlir, time:{time.time() - s0}')
    
    cmd_str = f"tpuc-opt {model_name}_qat.mlir --processor-assign=\"chip={chip.lower()} num_device=1 num_core=1\"  \
                         --import-calibration-table=\"file={cali_table} asymmetric=false\" --processor-top-optimize \
                         --convert-top-to-tpu=\"mode=INT8 asymmetric=false doWinograd=false ignore_f16_overflow=true\" \
                         --canonicalize --weight-fold -o  {model_name}_qat_{chip.lower()}_int8_sym_tpu.mlir"
    if log_out:
        print(f'cmd_str:{cmd_str}')
    s0 = time.time()
    os.system(cmd_str)
    if log_out:
        print(f'convert top mlir, time:{time.time() - s0}')

def find_new_param(model, unique_id, log_out = False):
    global param_to_idx_dict
    for name, param in model.named_parameters():
        if name in param_to_idx_dict and unique_id == param_to_idx_dict[name][1]:
            tmp = param.cpu().detach().numpy()
            if log_out:
                print(f'find torch name:{name}, shape:{tmp.shape}, old_data0:{tmp.reshape(-1)[0]}')
            return tmp
        
def clip_weight(data, scale, ConvTranspose):
    clip_range_min = ((-127 - 0) * scale).astype(data.dtype)
    clip_range_max = ((127 - 0) * scale).astype(data.dtype)
    if len(scale.shape) > 0 and scale.shape[0] > 1:
        new_data = []
        if ConvTranspose:
            data = data.transpose(1, 0, 2, 3)
        for c in range(data.shape[0]):
            new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
        new_data = np.array(new_data)
        if ConvTranspose:
            new_data = new_data.transpose(1, 0, 2, 3)
    else:
        new_data = np.clip(data, clip_range_min, clip_range_max)
    return torch.from_numpy(new_data)

def update_model_param(model, model_name='sophgo_mq_qmodel', chip= 'CV181X', output_path='./', log_out = False, idx = -1, save_all_iter_weight=False):
    parser=MlirParser(f'{model_name}_qat_origin.mlir')
    weight_file = parser.module_weight_file
    if not weight_file.startswith('/dev/shm/'):
        cmd_str = f'cp {weight_file} /dev/shm/'
        os.system(cmd_str)
        cmd_str = f'sed -i \'s/{weight_file}/\/dev\/shm\/{weight_file}/\' {model_name}_qat_origin.mlir'
        print(f'copy to shm before update: {cmd_str}')
        os.system(cmd_str)
        weight_file = f'/dev/shm/{weight_file}'
    w_ = np.load(weight_file)
    w = {}
    for k in w_:
        w[k] = w_[k]
    file_h = open('/dev/shm/{}_weight_name_to_unique_id.json'.format(model_name), "r")
    weight_name_to_unique_id = json.loads(file_h.read())
    file_h.close()
    module_tmp = deepcopy_graphmodule(model)
    convert_merge_bn(module_tmp.eval())
    for item in w:
        item2 = item
        if item not in weight_name_to_unique_id:
            item2 = item[:-4] #strip '_fix'
            if item2 not in weight_name_to_unique_id:
                if log_out:
                    print(f'warning, {item} not in weight_name_to_unique_id')
                continue
        unique_id = weight_name_to_unique_id[item2][0]
        if log_out:
            print(f'update {item}, unique_id:{unique_id}')
        tmp = find_new_param(module_tmp, unique_id)
        if tmp is None:
            if log_out:
                print(f'warning, {item}, find_new_param fail')
            continue
        if tmp.shape != w[item].shape:
            tmp = tmp.T
            if log_out:
                print(f'transpose {item}')
        w[item] = tmp

    file_h = open('/dev/shm/{}_clip_ranges.json'.format(model_name), "r")
    blob_range = json.loads(file_h.read())["sophgo_tpu"]
    file_h.close()
    cali_table = osp.join(output_path, '{}_cali_table_from_sophgo_mq_sophgo_tpu'.format(model_name))
    with open(cali_table, 'w') as f:
        f.write(f"# work_mode:QAT_all_int8 #Automatically generated, do not modify, work_mode choice:[QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc]\n")
        f.write("# op_name    threshold    min    max\n")
        weight_scale = []
        int4_th = []
        from sophgo_mq.prepare_by_platform import ParamsTable
        a_quant_min = ParamsTable[chip]['a_qscheme'].to_observer_params()['quant_min'] 
        a_quant_max = ParamsTable[chip]['a_qscheme'].to_observer_params()['quant_max'] 
        for name, value in blob_range.items():
            unique_id = value['param_id']['step']
            if log_out:
                print(f'blob_range, update {name}, unique_id:{unique_id}')
            step =  find_new_param(module_tmp, unique_id)
            for weight_name in weight_name_to_unique_id:
                if weight_name_to_unique_id[weight_name][1] == unique_id:
                    weight_data = find_new_param(module_tmp, weight_name_to_unique_id[weight_name][0])
                    w[weight_name] = clip_weight(weight_data, step, weight_name_to_unique_id[weight_name][2])
            if 'threshold' in value:
                assert len(step) == 1
                threshold = float(step[0]*max(-a_quant_min, a_quant_max))
                v_min = float(step[0]*a_quant_min)
                v_max = float(step[0]*a_quant_max)
                tmpstr = "{} {:.7f} {:.7f} {:.7f}\n".format(name[:-2], threshold, v_min, v_max)
                if name.endswith('_4'):
                    int4_th.append(tmpstr)
                elif name.endswith('_8'):
                    f.write(tmpstr)
                else:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(name, threshold, v_min, v_max))
            else:
                tmpstr = "{} {} {} {} {}\n".format(name, len(value['step']), ' '.join(['{:.7f}'.format(i) for i in step]),
                        len(value['zero_point']), ' '.join([str(i) for i in value['zero_point']]))
                if name.endswith('_weight') or name.endswith('_bias'):
                    weight_scale.append(tmpstr)
                else:
                    f.write(tmpstr)
        f.write('#int4_th\n')
        for i in int4_th:
            f.write(i)
        f.write('#weight_scale\n')
        for i in weight_scale:
            f.write(i)
        os.fsync(f.fileno())

    with open(weight_file, 'wb') as f:
        np.savez(f, **w)
        f.flush()
    lower_net(model_name, chip, output_path, log_out)

    if save_all_iter_weight:
        weight_file_copyed = f'idx{idx}_{weight_file}'
        if idx >= 0:
            os.system(f'cp -f {weight_file} {weight_file_copyed}')
        return weight_file_copyed
    else:
        return None

def export_onnx_with_fakequant_node(model: GraphModule, net_type='CNN',
                   input_shape_dict=None, dummy_input=None, output_path='./',
                   model_name='sophgo_mq_qmodel', deploy_to_qlinear=False, **extra_kwargs):
    r"""Convert GraphModule with fakequant node to onnx

    Args:
        model (GraphModule): GraphModule prepared qat module.
        backend_type (BackendType): specific which backend should be converted to.
        input_shape_dict (dict): keys are model input name(should be forward function
                                 params name, values are list of tensor dims)
        output_path (str, optional): path to save convert results. Defaults to './'.
        model_name (str, optional): name of converted onnx model. Defaults to 'sophgo_mq_qmodel'.

    >>> note on input_shape_dict:
        example: {'input_0': [1, 3, 224, 224]
                'input_1': [1, 3, 112, 112]
                }
        while forward function signature is like:
                def forward(self, input_0, input_1):
                    pass
    """
    kwargs = {
        'input_shape_dict': input_shape_dict,
        'dummy_input': dummy_input,
        'output_path': output_path,
        'model_name': model_name,
        'onnx_model_path': osp.join(output_path, '{}.onnx'.format(model_name)),
        'deploy_to_qlinear': deploy_to_qlinear
    }
    kwargs.update(extra_kwargs)
    deploy_model = deepcopy_graphmodule(model)
    convert_merge_bn(deploy_model, **kwargs)
    convert_onnx(deploy_model, **kwargs)
