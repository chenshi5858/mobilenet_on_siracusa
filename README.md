# Manual MiniMobileNet for Siracusa

This project deploys a small MobileNet-like integer network manually on the
Siracusa N-EUREKA accelerator and executes it with GVSoC. It does not use
Deeploy, ONNX, DORY or an inference runtime.

The model classifies three synthetic 10x10 images: a vertical line, a
horizontal line and a diagonal line. It is a didactic model with deliberately
designed weights, not a model trained on ImageNet.

## Network

| Layer | Input | Operation | Output | Requantization |
| --- | --- | --- | --- | --- |
| Stem | 10x10x1 | 3x3 convolution | 8x8x3 | ReLU, shift 0 |
| Spatial | 8x8x3 | 3x3 depthwise | 6x6x3 | ReLU, shift 2 |
| Expansion | 6x6x3 | 1x1 pointwise | 6x6x6 | ReLU, shift 0 |
| Head | 6x6x6 | 1x1 pointwise | 6x6x3 | ReLU, shift 1 |
| Classifier | 6x6x3 | Global sum + argmax on RISC-V | 3 scores | None |

The depthwise plus pointwise pair is the characteristic MobileNet block.

## Build and run

Open a normal WSL shell, not the Deeploy container:

```bash
cd /home/chen/master/pulp-sdk
unset CFLAGS CPPFLAGS CXXFLAGS
export PULP_RISCV_GCC_TOOLCHAIN=/opt/riscv
export PATH="/opt/riscv/bin:$PATH"
source configs/siracusa.sh

cd /home/chen/master/manual-mobilenet-siracusa
python3 tools/generate_model.py
make clean
make all platform=gvsoc
make run platform=gvsoc
```

The final line must report `RESULT: PASS`. Every intermediate output produced
by N-EUREKA is compared against the NumPy integer reference.

## What is manual here

`tools/generate_model.py` defines all weights, runs the integer reference and
implements the N-EUREKA bit-plane weight packing. `src/main.c` constructs each
accelerator task, selects convolution modes, provides addresses and strides,
sets normalization and quantization, launches N-EUREKA, waits for completion,
and performs the final global reduction on the RISC-V cluster.

The project vendors the low-level PULP-NNX hardware abstraction under
`vendor/pulp-nnx`. That driver writes the N-EUREKA registers; it does not
generate, schedule or transform the network.
