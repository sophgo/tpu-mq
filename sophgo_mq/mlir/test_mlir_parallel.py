import os
import sys
import tpu_mlir

from mlir.ir import *
from mlir.dialects import quant
import mlir.ir

from tpu_mlir.python.utils.mlir_parser import MlirParser


import numpy as np
import pymlir
pymlir.set_mem_mode("value_mem")


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from multiprocessing import Pool

num = 10

parser=MlirParser("resnet18_qat.mlir")
inputs=parser.inputs[0].name
outputs=parser.get_output_op_names_n_shapes()
for n in outputs:
    o = n
    break
input_shapes=parser.get_input_shapes()[0]
print(f'input {inputs} {o} {input_shapes}')

test_inputs=[]
test_golden=[]
module=pymlir.module()
module.load("resnet18_qat.mlir")
for i in range(num):
    test_input=np.random.rand(16,3,224,224)
    test_inputs.append(test_input)
    module.set_tensor(inputs, test_input)
    module.invoke()
    golden=module.get_tensor(o)
    test_golden.append(golden)
param=[]
for i in range(num):
    param.append((test_inputs[i],test_golden[i]))
def inference(x):
    module=pymlir.module()
    module.load("resnet18_qat.mlir")
    ok=True
    for i in range(10000):
        module.set_tensor(inputs, x[0])
        module.invoke()
        results=module.get_tensor(o)
        if np.max(np.abs(results-x[1]))>1e-6:
            ok = False
    if not ok:
        print('result not match!!!!')
    else:
        print(f"{x} {i} compare OK!")

    return ok

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        results = pool.map(inference, param)

