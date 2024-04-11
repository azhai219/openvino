import openvino.runtime as ov
from transformers import AutoModelForCausalLM, AutoConfig
from optimum.intel.openvino import OVModelForCausalLM
import time

import types
import inspect
import argparse

import sys, os

'''
mpi4py has issue working with intel MPI, so we can only perform MPI API calls inside CPU plugin code.
https://github.com/mpi4py/mpi4py/issues/418

    export MPICC=`which mpicc`
    python -m pip install git+https://github.com/mpi4py/mpi4py

OV_CPU_PROFILE=1 mpirun   \
  -n 1 env RANK_INFO=0of2 numactl --all -C 0-7 -m 0 python ./testLLM.py /mnt/disk2/tingqian/models/Mistral-7B-Instruct-v0.1-OV/FP16 : \
  -n 1 env RANK_INFO=1of2 numactl --all -C 48-55 -m 1 python ./testLLM.py /mnt/disk2/tingqian/models/Mistral-7B-Instruct-v0.1-OV/FP16
'''


rank_info = os.environ["RANK_INFO"]
print(rank_info)

'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
rank_info = f"[rank: {rank}/{world_size}] "
isMaster = (rank == 0)

print(rank_info)
sys.exit(0)
'''


def hook_forward(model):
    model._org_forward = model.forward
    model._latencies = []
    def new_forward(self, *args, **kwargs):
        # Call the original method
        # print(args, kwargs)
        t0 = time.time()
        ret = self._org_forward(*args, **kwargs)
        t1 = time.time()
        self._latencies.append(t1 - t0)
        return ret
    # https://stackoverflow.com/questions/1409295/set-function-signature-in-python
    # package transformers uses inspect.signature to detect exact signature of
    # model's forward method and behave differently based on that, for example
    # only auto-generate attention-mask when signature contain such named parameter
    new_forward.__signature__ = inspect.signature(model.forward)
    model.forward = types.MethodType(new_forward, model)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--new-tokens', type=int, default=32)
parser.add_argument('-b', '--batch-sizes', type=int, nargs='*')
parser.add_argument('--bf16', action="store_true")
parser.add_argument('-p', '--prompt', type=str, default="What's oxygen?")
parser.add_argument('model_path')

args = parser.parse_args()
prompt = args.prompt
if prompt.isdigit():
    prompt = "Hi"*(int(prompt) - 1)

if args.batch_sizes is None:
    args.batch_sizes = [1]

print(rank_info, " OV VERSION=", ov.get_version())

# model list
OV_IR = []
OV_IR.append('/home/xiping/openvino.genai/llm_bench/python/my_test/gemma/models/ov-share-05/models/gemma-7b-it/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/')
OV_IR.append('/mnt/disk2/models_tmp/gemma/my_converted/gemma-2b-it/FP16/')
OV_IR.append('/mnt/disk2/models_tmp/gemma/my_converted/gemma-2b-it/INT4_compressed_weights/')
OV_IR.append("/mnt/disk2/models_tmp/gemma/ov-share-05/models/gemma-7b-it/pytorch/dldt/FP16/")
OV_IR.append("/home/tingqian/models/llama-2-7b-chat")
OV_IR.append("/mnt/disk2/tingqian/models/Mistral-7B-Instruct-v0.1-OV/FP16")

for i, model_ir in enumerate(OV_IR):
    print(f"[{i}] : {model_ir}")

model_path = args.model_path
if model_path.isdigit():
    model_path = OV_IR[int(model_path)]

core = ov.Core()
device = "CPU"

from transformers import AutoTokenizer
 
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
if args.bf16:
    ov_config["INFERENCE_PRECISION_HINT"] = "bf16"
else:
    ov_config["INFERENCE_PRECISION_HINT"] = "f32"

print(rank_info, "--> load tokenizer.")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.add_special_tokens({'pad_token': '[PAD]'})
    #tok.pad_token = tok.eos_token_id
 
cfg=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(rank_info, f"--> config {cfg}.")

t1=time.time()
ov_model = OVModelForCausalLM.from_pretrained(
    model_path,
    device=device,
    ov_config=ov_config,
    config=cfg,
    trust_remote_code=True,
)
print(rank_info, f" Model compilation took {time.time()-t1:.2f} seconds.")




hook_forward(ov_model)

print(rank_info, "prompt:", prompt[:32], "...")

print(f"============== {model_path} ==============")
for batch_size in args.batch_sizes:
    prompt_batch = [prompt] * batch_size
    inputs = tok(prompt_batch, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    print(rank_info, f" generating for prompt shape: {input_ids.shape[0]}x{input_ids.shape[1]}...", end="")

    # synchronize (can be skipped)
    # data = (rank+1)**2
    # print(rank_info, f"data={data}")
    # all_data = comm.allgather(data)
    # print(rank_info, f"all_data={all_data}")

    ov_model._latencies = []
    answer = ov_model.generate(**inputs, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False)

    print("\r", " " * 80, "\r", end="")
    for out in tok.batch_decode(answer[:8], skip_special_tokens=True):
        print("\t Output=", out.encode("utf-8")[:200], "...")

    l = ov_model._latencies
    second_tok_latency = sum(l[1:])/(len(l)-1)
    print(rank_info, f" prompt:{input_ids.shape[0]}x{input_ids.shape[1]}  {l[0]*1e3:.1f} ms + {second_tok_latency*1e3:.1f} ms x {len(l)-1}")

