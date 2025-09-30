import os
import tpu_mlir
from mlir.ir import *
from mlir.dialects import quant
import mlir.ir

import numpy as np
import pymlir
pymlir.set_mem_mode("value_mem")
from utils.mlir_parser import MlirParser
from utils.mlir_shell import mlir_lowering
from utils.mlir_shell import _os_system

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from tpu_mq.convert_deploy import convert_deploy
from tpu_mq.convert_deploy import convert_merge_bn
from tpu_mq.utils import deepcopy_graphmodule
from tpu_mq.utils import generate_random_string

import hashlib
import copy

def hash_(data_array):
    m = hashlib.md5(repr(data_array).encode())
    return m.hexdigest()

def hash_mlir(mlir_file):
    #and move weight file to /dev/shm
    hash_mlir = {}
    weights_mlir = {}
    parser=MlirParser(mlir_file)
    weight_file_name = parser.module_weight_file
    module=pymlir.module()
    module.load(mlir_file)
    for op in module.all_weight_names:
        w = module.get_tensor(op).copy()
        weights_mlir[op] = w
        h = hash_(w)
        hash_mlir[op] = h
    cmd = f"sed -i 's/{weight_file_name}/\/dev\/shm\/{weight_file_name}/' {mlir_file}"
    print(f'SED cmd {cmd}')
    os.system(cmd)
    '''
    with open(mlir_file,'w') as f:
        f.write(str(parser.module))
    '''
    print(f'load {weight_file_name} and save to dev/shm')
    w = np.load(weight_file_name, allow_pickle=True)
    weight_file_name = f'/dev/shm/{weight_file_name}'
    np.savez(weight_file_name, **w)
    os.system('ls -al /dev/shm/')
    return hash_mlir, weights_mlir, weight_file_name

def hash_torch(model):
    from tpu_mq.convert_deploy import convert_merge_bn
    module_tmp2 = copy.deepcopy(model)
    convert_merge_bn(module_tmp2.eval())
    hash_torch = {}
    param_torch = {}
    for name, param in module_tmp2.named_parameters():
        if name.rsplit(".scale",1)[0] == name and ("post_act_fake" in name or "weight_fake_quant" in name):
            continue
        if name.rsplit(".zero_point",1)[0] == name and ("post_act_fake" in name or "weight_fake_quant" in name):
            continue
        param_torch[name] = param.detach().cpu().numpy().copy()
        h = hash_(param_torch[name])
        hash_torch[name] = h
        #print(f'param name/shape {name} {param.shape} {h}')
    return hash_torch, param_torch

def check_match(hash_torch, hash_mlir, param_torch, weights_mlir):
    found_torch = {}
    found_mlir = {}
    name_dict = {}
    need_transpose = []
    for p in hash_torch:
        h = hash_torch[p]
        found = False
        for p_ in hash_mlir:
            h_ = hash_mlir[p_]
            if h == h_:
                print(f'found {p} match {p_} {h}')
                found_torch[p] = True
                found_mlir[p_] = True
                name_dict[p] = p_
                found = True
                break
        if not found:
            found_ = False
            for w_ in weights_mlir:
                if w_ in found_mlir:
                    continue
                if (param_torch[p].shape == weights_mlir[w_].shape) and np.max(np.abs(param_torch[p]-weights_mlir[w_])) < 1e-6:
                    found_torch[p] = True
                    found_mlir[w_] = True
                    name_dict[p] = w_
                    found_ = True
                    break
                elif (len(param_torch[p].shape) == 2 and len(weights_mlir[w_].shape) == 2 and param_torch[p].size == weights_mlir[w_].size) and np.max(np.abs(param_torch[p]-weights_mlir[w_].T)) < 1e-6:
                    found_torch[p] = True
                    found_mlir[w_] = True
                    name_dict[p] = w_
                    need_transpose.append(p)
                    found_ = True
                    break
            '''
            if found_:
                print(f'found {p} match {w_} in retry')
            else:
                print(f'weight not found {p} {h} in retry')
            '''
        '''
        else:
            print(f'found {p} match {p_} {h}')
        '''
    for p in hash_torch:
        if p in found_torch:
            continue
        else:
            print(f'{p} not found in mlir!!!!!')
            return False, None
    return True, name_dict, need_transpose

def tpu_train_prepare(model, args, val_loader):

    torch_hash, torch_param = hash_torch(model)
    
    convert_deploy(model.eval(), input_shape_dict={'data': [args.batch_size, 3, 224, 224]},
        model_name='{}'.format(args.arch), output_path=args.output_path, mlir_deploy=True, chip=args.chip, val_loader=val_loader)

    mlir_hash, mlir_weights, mlir_weight_file = hash_mlir(f'{args.arch}_qat.mlir')

    suc, name_dict, need_transpose = check_match(torch_hash, mlir_hash, torch_param, mlir_weights)
    if suc:
        return True, name_dict, need_transpose, mlir_weight_file
    else:
        return False, None, None, None
    
def update_mlir_weight(torch_module, need_transpose, mlir_weight_file, name_dict):
    print(f'weight file {mlir_weight_file}')
    w_ = np.load(mlir_weight_file, allow_pickle=True)
    w = {}
    for k in w_:
        w[k] = w_[k]
    from tpu_mq.convert_deploy import convert_merge_bn
    module_tmp = copy.deepcopy(torch_module)
    convert_merge_bn(module_tmp.eval())

    for name, param in module_tmp.named_parameters():
        if name.rsplit(".scale",1)[0] == name and ("post_act_fake" in name or "weight_fake_quant" in name):
            continue
        if name.rsplit(".zero_point",1)[0] == name and ("post_act_fake" in name or "weight_fake_quant" in name):
            continue
        param = param.detach().cpu().numpy().copy()
        if name in need_transpose:
            w[name_dict[name]] = param.T
        else:
            w[name_dict[name]] = param
    np.savez(mlir_weight_file, **w)

def lower_net(arch, chip):
    top_mlir=f'{arch}_qat.mlir'
    tpu_mlir=f'/dev/shm/{arch}_int8_sym_tpu.mlir'
    cali_table=f'{arch}_cali_table_from_tpu_mq'
    mode = 'INT8'
    int8_weight_file=f'/dev/shm/{arch}_int8_weight.npz'

    cmd = [
        "tpuc-opt", top_mlir, "--processor-assign=\"chip={} num_device={} num_core={}\"".format(
            chip.lower(), 1, 1)
    ]
    mode = mode.upper()
    cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
        cali_table, False)
    cmd.extend([cali_param])
    #do extra conversion for differnet chips
    cmd.extend(["--processor-top-optimize"])
    lower_param = "--convert-top-to-tpu=\"mode={} asymmetric={} doWinograd={} ignore_f16_overflow={} weightFileName={}\"".format(
        mode, False, False, True, int8_weight_file)
    cmd.extend([
        lower_param,
        "--canonicalize",
        "--weight-fold",
        "-o",
        tpu_mlir,
    ])
    log_file = ""
    _os_system(cmd)

    module=pymlir.module()
    module.load(tpu_mlir)
    return module
