from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import sys

def const(shape):
    w = np.random.rand(*shape).astype(np.float32)
    return op.Constant(w)

def value(*shape):
    return np.random.rand(*shape).astype(np.float32)

def model(weight_value, bias_value = None):
    # [N, K]
    N, K = weight_value.shape

    # input is [B,M,K]
    input = opset.parameter([-1, -1, K], Type.f32, name = 'in')

    weight = op.Constant(weight_value)
    mm = opset.matmul(input, weight, transpose_a = False, transpose_b = True)
    if bias_value:
        mm = opset.add(mm, op.Constant(bias_value))

    Result = opset.result(mm, name='mm')
    return Model([Result], [input], 'Model15')


def model_parallel(weight_value, bias_value = None):
    # [N, K]
    N, K = weight_value.shape

    # input is [B,M,K]
    input = opset.parameter([-1, -1, K], Type.f32, name = 'in')

    weight = op.Constant(weight_value)

    weights = opset.split(weight, 0, 2)

    mm0 = opset.matmul(input, weights.output(0), transpose_a = False, transpose_b = True, name="mm0")
    mm1 = opset.matmul(input, weights.output(1), transpose_a = False, transpose_b = True, name="mm1")

    mm0.rt_info["paralellDomain"] = "mm"
    mm1.rt_info["paralellDomain"] = "mm"
    mm = opset.concat([mm0, mm1], 2)

    Result = opset.result(mm, name='mm')
    return Model([Result], [input], 'Model15')

core = Core()

B=1
M=1
N=32*8*16
K=4096

input_value = np.random.rand(B, M, K).astype(np.float32)
weight_value = np.random.rand(N, K).astype(np.float32)
ref_out_value = np.matmul(input_value.reshape([B*M, K]), weight_value.transpose(1,0)).reshape([B,M,N])

#m = model(weight_value)
m = model_parallel(weight_value)

config={
    "NUM_STREAMS":1,
    "INFERENCE_PRECISION_HINT":"bf16"
}
cm = core.compile_model(m, "CPU", config)

act_out_value = np.array(cm([input_value])[cm.output(0)].data)

allcose = np.allclose(ref_out_value, act_out_value)
if not allcose:
    print(ref_out_value)
    print(act_out_value)
    print(f"allclose={allcose}")
    sys.exit(1)
print(f"allclose={allcose}")

#for i in range(1000):
#    a = cm([input_value])
