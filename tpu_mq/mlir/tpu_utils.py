import os
import time
import copy
import hashlib
import subprocess
import shlex
import json
import os.path as osp

import numpy as np
try:
    import tpu_mlir
    from tools.model_transform import model_transform_func
    from utils.mlir_parser import MlirParser
    from mlir.ir import *
    from mlir.dialects import quant
    import mlir.ir
    import pymlir
    pymlir.set_mem_mode("value_mem")
    from utils.mlir_parser import MlirParser
    from utils.mlir_shell import mlir_lowering
    from utils.mlir_shell import _os_system
except ModuleNotFoundError:
    print("tpu_mlir not found, use gpu and cpu")
    pass
except ImportError:
    print("tpu_mlir import error, check its installation")
    sys.exit(1)
except Exception as e:
    print(f"tpu_mlir import error {e}")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as TorchTransforms

from torch.fx import GraphModule

from tpu_mq.utils import deepcopy_graphmodule
from tpu_mq.utils import generate_random_string


param_to_idx_dict = {}
__all__ = ['deploy_to_mlir', 'update_model_param', 'param_to_idx_dict']


def deploy_to_mlir(chip: str = "BM1690", val_loader: torch.utils.data.DataLoader = None, **kwargs):
    input_shape_dict = kwargs['input_shape_dict']
    output_path = kwargs['output_path']
    model_name = kwargs['model_name']
    batch_size = next(iter(input_shape_dict.values()))[0]
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
        from tpu_mq.convert_deploy import model_onnx_mem
        global model_onnx_mem
        onnx_filename = model_onnx_mem
    shape_str_list = []
    for name in input_shape_dict:
        shape_str = ','.join([str(i) for i in input_shape_dict[name]])
        shape_str_list.append(f'[{shape_str}]')
    shape_str_list = ','.join(shape_str_list)
    
    s0 = time.time()
    if export_to_mm:
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
        cmd_str = f'cp {model_name}_cali_table_from_tpu_mq /dev/shm/'
        os.system(cmd_str)
        cmd_str = f'cp {model_name}_q_table_from_tpu_mq /dev/shm/'
        os.system(cmd_str)
        print(f'copy files to {output_path}')
        os.system(f'ls {output_path}')
        work_path='/dev/shm/'

    print(f'model_transform_func time:{time.time() - s0}')

    if 'mlir_deploy_debug_onlytransform' not in kwargs or kwargs['mlir_deploy_debug_onlytransform'] == False:
        calibration_table = os.path.join(work_path, '{}_cali_table_from_tpu_mq'.format(model_name))
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
            quantize_table = os.path.join(work_path, '{}_q_table_from_tpu_mq'.format(model_name))
            quantize_table = f'--quantize_table {quantize_table}'
            quantize_mode = 'BF16'
            quantize_str = 'bf16'
        bmodel_ext = 'cvimodel' if chip in ['MARS3', 'CV183X', 'CV182X', 'CV181X', 'CV180X', 'CV186X'] else 'bmodel'
        if 'fuse_preprocess' in kwargs and kwargs['fuse_preprocess']:
            if 'customization_format' in kwargs and kwargs['customization_format']:
                customization_format=f"--customization_format={kwargs['customization_format']}"
            else:
                customization_format=''
            if 'aligned_input' in kwargs and kwargs['aligned_input']:
                aligned_input=f"--aligned_input={kwargs['aligned_input']}"
            else:
                aligned_input=''

            align=kwargs['aligned_input']
            cmd_str = f"model_deploy.py \
            --mlir {model_name}_qat.mlir \
            --quantize {quantize_mode} \
            --calibration_table {calibration_table} {quantize_table} \
            --chip {chip} {test_input} {test_reference}\
            --fazzy_match \
            --tolerance 0.99,0.90 --debug {not_gen_bmodel}\
            --model {model_name}_qat_{chip.lower()}_{quantize_str}.{bmodel_ext}\
            --fuse_preprocess {customization_format} {aligned_input}"
        else:
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
    cali_table = os.path.join(output_path, '{}_cali_table_from_tpu_mq'.format(model_name))
    cmd_str = f"tpuc-opt {model_name}_qat_origin.mlir --shape-infer --canonicalize --extra-optimize -o {model_name}_qat.mlir"
    if log_out:
        print(f'cmd_str:{cmd_str}')
    s0 = time.time()
    os.system(cmd_str)
    if log_out:
        print(f'convert origin mlir, time:{time.time() - s0}')

    cmd_str = f"tpuc-opt {model_name}_qat.mlir --processor-assign=\"chip={chip.lower()} mode=INT8 num_device=1 num_core=1 addr_mode=auto\"  \
                         --import-calibration-table=\"file={cali_table} asymmetric=false\" --processor-top-optimize \
                         --convert-top-to-tpu=\"asymmetric=false doWinograd=false ignore_f16_overflow=False q_group_size=0 matmul_perchannel=False gelu_mode=normal\" \
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

def update_model_param(model, model_name='tpu_mq_qmodel', chip= 'CV181X', output_path='./', log_out = False, idx = -1, save_all_iter_weight=False):
    parser=MlirParser(f'{model_name}_qat_origin.mlir')
    weight_file = parser.module_weight_file
    '''
    if not weight_file.startswith('/dev/shm/'):
        cmd_str = f'cp {weight_file} /dev/shm/'
        os.system(cmd_str)
        cmd_str = f'sed -i \'s/{weight_file}/\/dev\/shm\/{weight_file}/\' {model_name}_qat_origin.mlir'
        print(f'copy to shm before update: {cmd_str}')
        os.system(cmd_str)
        weight_file = f'/dev/shm/{weight_file}'
    '''
    w_ = np.load(weight_file)
    w = {}
    for k in w_:
        w[k] = w_[k]
    file_h = open('/tmp/{}_weight_name_to_unique_id.json'.format(model_name), "r")
    #file_h = open('/dev/shm/{}_weight_name_to_unique_id.json'.format(model_name), "r")
    weight_name_to_unique_id = json.loads(file_h.read())
    file_h.close()
    module_tmp = deepcopy_graphmodule(model)
    from tpu_mq.convert_deploy import convert_merge_bn
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

    file_h = open('/tmp/{}_clip_ranges.json'.format(model_name), "r")
    #file_h = open('/dev/shm/{}_clip_ranges.json'.format(model_name), "r")
    blob_range = json.loads(file_h.read())["tpu"]
    file_h.close()
    cali_table = osp.join(output_path, '{}_cali_table_from_mq'.format(model_name))
    with open(cali_table, 'w') as f:
        f.write(f"# work_mode:QAT_all_int8 #Automatically generated, do not modify, work_mode choice:[QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc]\n")
        f.write("# op_name    threshold    min    max\n")
        weight_scale = []
        int4_th = []
        from tpu_mq.prepare_by_platform import ParamsTable
        a_quant_min = ParamsTable[chip]['a_qscheme'].to_observer_params()['quant_min'] 
        a_quant_max = ParamsTable[chip]['a_qscheme'].to_observer_params()['quant_max'] 
        for name, value in blob_range.items():
            unique_id = value['param_id']['step']
            if log_out:
                print(f'blob_range, update {name}, unique_id:{unique_id}')
            step =  find_new_param(module_tmp, unique_id)
            step = np.abs(step)
            step[step<1e-6] += 1e-6
            for weight_name in weight_name_to_unique_id:
                if weight_name_to_unique_id[weight_name][1] == unique_id:
                    weight_data = find_new_param(module_tmp, weight_name_to_unique_id[weight_name][0])
                    w[weight_name] = clip_weight(weight_data, step, weight_name_to_unique_id[weight_name][2])
            if 'threshold' in value:
                assert len(step) == 1
                threshold = float(step[0]*max(-a_quant_min, a_quant_max))
                v_min = float(step[0]*a_quant_min)
                v_max = float(step[0]*a_quant_max)
                tmpstr = "{} {:.15f} {:.15f} {:.15f}\n".format(name[:-2], threshold, v_min, v_max)
                if name.endswith('_4'):
                    int4_th.append(tmpstr)
                elif name.endswith('_8'):
                    f.write(tmpstr)
                else:
                    f.write("{} {:.15f} {:.15f} {:.15f}\n".format(name, threshold, v_min, v_max))
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
