from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import sys
import argparse


def const(shape):
    w = np.random.rand(*shape).astype(np.float32)
    return op.Constant(w)

def model_parallel(weight_value, split_N = 2, num_layers = 1):
    # [N, K]
    N, K = weight_value.shape

    # input is [B,M,K]
    input = opset.parameter([-1, -1, K], Type.f32, name = 'in')

    cur_input = input
    for layer in range(num_layers):
        weight = op.Constant(weight_value.copy())
        layername = f"fc{layer}"

        if split_N < 2:
            output = opset.matmul(cur_input, weight, transpose_a = False, transpose_b = True, name=layername)
            output.rt_info["paralellDomain"] = layername
        else:
            weights = opset.split(weight, 0, split_N)
            mm_out = []
            for i in range(split_N):
                mmi = opset.matmul(cur_input, weights.output(i), transpose_a = False, transpose_b = True, name=f"{layername}_split{i}")
                mmi.rt_info["paralellDomain"] = layername
                mm_out.append(mmi)
            output = opset.concat(mm_out, 2)            # concat along dim2
        
        output = opset.normalize_l2(output, 2, eps_mode = "add", eps=1e-6)
        cur_input = output

    Result = opset.result(output, name='mm')
    return Model([Result], [input], 'Model15')


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-n', '--num_layers', type=int, default=10)
    parser.add_argument('-p', '--perf_test_cnt', type=int, default=0)
    parser.add_argument('-bf16', action="store_true")
    parser.add_argument('prompt', type=str, nargs='?')
    
    args = parser.parse_args()
    
    core = Core()

    B=1
    M=1
    N=4096
    K=4096

    np.random.seed(0)
    input_value = np.random.rand(B, M, K).astype(np.float32) - 0.5
    weight_value = np.random.rand(N, K).astype(np.float32) - 0.5

    cur_input = input_value.reshape([B*M, K])
    for layer in range(args.num_layers):
        cur_input = np.matmul(cur_input, weight_value.transpose(1,0))
    ref = cur_input.reshape([B,M,N])

    #m = model(weight_value)
    m0 = model_parallel(weight_value, 0, args.num_layers)
    m1 = model_parallel(weight_value, 2, args.num_layers)

    config={
        "NUM_STREAMS":1,
        "INFERENCE_PRECISION_HINT":"bf16" if args.bf16 else "f32"
    }
    def infer_async(req, inputs):
        req.start_async(inputs)
        req.wait()
        return req.get_output_tensor()

    cm0 = core.compile_model(m0, "CPU", config)
    request0 = cm0.create_infer_request()
    act0 = np.array(infer_async(request0, [input_value]).data)

    cm1 = core.compile_model(m1, "CPU", config)
    request1 = cm1.create_infer_request()
    act1 = np.array(infer_async(request1, [input_value]).data)

    np.set_printoptions(suppress=True)
    #np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    def diff(a, b):
        adiff = a-b
        return f"{np.mean(adiff):.1f}, {np.std(adiff):.3f}"

    #print(f"ref : {diff(ref, ref)} {ref.shape} {ref.dtype} {ref}")
    #print(f"act0: {diff(ref, act0)} {act0.shape} {act0.dtype} {act0}")
    print(f"act1: {diff(act0, act1)} {act1.shape} {act1.dtype} {act1}")

    if args.perf_test_cnt > 0:
        print("performance test: ")
        import time
        t0 = time.time()
        for i in range(args.perf_test_cnt):
            a = infer_async(request0, [input_value])
        t0 = time.time() - t0
        print(f"t0={t0}")

        time.sleep(0.1)

        t1 = time.time()
        for i in range(args.perf_test_cnt):
            a = infer_async(request1, [input_value])
        t1 = time.time() - t1
        print(f"t1={t1}")

if __name__ == '__main__':
    main()