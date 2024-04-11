# OpenVINO integration with oneCCL for LLM inferencing on NUMA

```bash
cd openvino
chmod +x prepare_oneccl.sh

# clong oneccl, build & install to local folder
./prepare_oneccl.sh

# initialize oneccl env
source oneccl/build/_install/env/setvars.sh

# build openvino as usuall（with -DENABLE_CPPLINT=OFF）
```

## run LLM and profile:
- `OV_CPU_PROFILE`: turn on the `profiler` tool at runtime
- `ENABLE_CCL`: turn on the `ccl` at runtime.

```bash
OV_CPU_PROFILE=1 ENABLE_CCL=1 mpirun \
    -n 1 env RANK_INFO=0of2 numactl --all -C 0-7 -m 0 python ./testLLM.py --bf16 /mnt/disk2/tingqian/models/Mistral-7B-Instruct-v0.1-OV/FP16 -b 1 1 4 1 4 : \
    -n 1 env RANK_INFO=1of2 numactl --all -C 48-55 -m 1 python ./testLLM.py --bf16 /mnt/disk2/tingqian/models/Mistral-7B-Instruct-v0.1-OV/FP16 -b 1 1 4 1 4
```

 - -b 1 1 4 1 4  will run promprt 5 times with different batch size
 -  --bf16 should be used on SPR
 - number of output tokens are specified with `-n 512`
 - prompt tokens can be specified with:
    -  `-p "What's oxygen?"`
    -  `-p 1024` : generate 1024 tokens (EOS + token `Hi` repeated 1023 times)

latency will printed as follows:

```bash
# first batch is warm-up, must ignored
0of2  prompt:1x6  3971.1 ms + 158.7 ms x 31
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
0of2  prompt:1x6  167.9 ms + 158.0 ms x 31
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
0of2  prompt:4x6  193.7 ms + 162.6 ms x 31
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
0of2  prompt:1x6  169.5 ms + 158.1 ms x 31
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
         Output= b'What\'s oxygen?\nA: The gas that makes you go "Huh?"\n\nWhat\'s carbon?\nA: The stuff that makes diamonds. And black' ...
0of2  prompt:4x6  188.3 ms + 160.9 ms x 31
```

open generated `ov_profile_mpi.json` in `chrome://tracing/`
