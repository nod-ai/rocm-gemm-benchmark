# GEMM/Attention Benchmarking Suite

This repository provides a comprehensive benchmarking framework to evaluate the performance of GEMM (General Matrix Multiply) and Attention operations across multiple ROCm backends, including IREE, rocBLAS, hipBLASLt, Triton, and PyTorch. The goal is to gain a deep understanding of how IREE compares to other AMD backends and to continuously track and visualize performance metrics.

## Problem Statement

We currently lack a comprehensive understanding of how IREE's performance stacks up against other backends like rocBLAS and hipBLASLt, particularly within AMD hardware. This repository aims to fill that gap by providing a robust and extensible benchmarking suite.

## Design Solution

To address this problem, we provide an extensible benchmarking framework with the following features:

- **Efficient GEMM Benchmarking**: Instead of full model inference, which can be slow, this suite benchmarks thousands of simple GEMM kernels to gather performance data rapidly.
- **Extensible Harness**: The framework supports multiple backends, allowing for easy comparison and integration of new backends as needed.
- **GPU Strain Simulation**: Techniques are implemented to simulate the GPU workload of full model execution using isolated GEMMs.
- **Nightly Performance Benchmarking**: A nightly runner automatically benchmarks the latest performance results.
- **Frontend Interface**: An online interface that is constantly updated to show current performance metrics.

## Components

### C++ Benchmarking Library

The C++ benchmarking library forms the core of this suite, providing:

- **Configurable Benchmark Runner**: Customizable settings for various benchmarking scenarios.
- **Data Initialization & Memory Management**: Efficient handling of data and memory for accurate benchmarking.
- **De-optimization Techniques**: Methods to simulate realistic GPU workload scenarios.
- **Backends**:
  - IREE (using the IREE C API)
  - rocBLAS (C++ API)
  - hipBLASLt (C++ API)

### Python Data Collection

A Python-based data collection system integrates seamlessly with the C++ library:

- **ctypes + ZMQ Integration**: Interfaces with the C++ library to dispatch and manage benchmarking tasks.
- **GEMM Problem Generation**: Automatically generates various GEMM problems, including different shapes and data types.
- **Parallel Dispatching**: Distributes benchmarking tasks across multiple GPUs, pooling results efficiently.
- **Performance Visualization**: Generates comparison graphs and roofline plots to visualize performance data.

### Python Autotuner

The autotuner is designed to optimize performance:

- **MLIR Generation**: Produces candidate MLIRs with varying workgroup and tiling sizes.
- **Performance Tuning**: Selects the best-performing configurations for GEMM operations.

### Flash Attention Benchmarking

In addition to GEMM, the framework also supports benchmarking Flash Attention operations:

- **Shared Pipeline**: Utilizes the same pipeline as GEMM benchmarking for consistency and efficiency.
- **Python Codegen + Compilation**: Specialized scripts for code generation and compilation of Flash Attention operations.
- **Tuning with Spec MLIR Files**: Implements tuning with MLIR files for FP8 and FP16 data types.
- **Backends**:
  - Native torch (compiled with ROCm)
  - Triton Flash Attention

Here are the updated setup and execution instructions for your README based on the GitHub workflow:

---

## Setup & Execution

### Prerequisites

- **Hardware**: Currently, this repository is tested only on AMD MI300 GPUs. There are future plans to support heterogeneous AMD hardware benchmarks.
- **Software**:
  - ROCm v6.1+ and associated libraries (rocBLAS, hipBLASLt)
  - CMake, Ninja
  - Python 3.10+
  - Meson build system

### Installation & Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/suryajasper/rocm-gemm-benchmark.git
   cd rocm-gemm-benchmark
   ```

2. **Install Required Dependencies**:

   ```bash
   sudo apt install -y libzmq3-dev libboost-numpy-dev python3-zmq python3-h5py python3-tqdm ninja-build meson
   ```

3. **Update IREE Submodule**:

   ```bash
   cd third_party/iree
   git submodule update --init
   cd ../..
   ```

4. **Build LLVM Project**:

   ```bash
   cd third_party/llvm-project
   cmake -G Ninja -B build/ -S llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;clang"
   cmake --build build/
   cd ../..
   ```

5. **Build IREE Kernels**:

   ```bash
   cd src/ireekernels
   cmake -G Ninja -B ../ireekernelsbuild
   cmake --build ../ireekernelsbuild
   cd ../..
   ```

6. **Build the Main Project**:

   ```bash
   CXX=hipcc meson setup build
   cd build
   sudo ninja
   cd ..
   mkdir -p results
   ```

7. **Set Up Python Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r gemmbench/requirements.txt
   deactivate
   ```

### Running Benchmarks

Benchmarks can be run in parallel across multiple GPUs. If the `--backends` argument is unspecified, all backends will be initialized and can be benchmarked simultaneously.

1. **RocBLAS Benchmarks**:

   ```bash
   sudo pkill -f gemm-bench
   source venv/bin/activate
   for device in $(seq 0 2); do (sudo build/gemm-bench --device=$device &); done
   ./gb run --backends=rocblas --repeat=1 --output=results/rocblas.csv
   sudo pkill -f gemm-bench
   deactivate
   ```

2. **HipBLASLt Benchmarks**:

   ```bash
   sudo pkill -f gemm-bench
   source venv/bin/activate
   for device in $(seq 0 2); do (sudo build/gemm-bench --device=$device &); done
   ./gb run --backends=hipblaslt --repeat=1 --output=results/hipblaslt.csv
   sudo pkill -f gemm-bench
   deactivate
   ```

3. **IREE Benchmarks**:

   ```bash
   sudo pkill -f gemm-bench
   source venv/bin/activate
   for device in $(seq 0 2); do (sudo build/gemm-bench --device=$device &); done
   ./gb run --backends=iree --repeat=1 --output=results/iree.csv
   sudo pkill -f gemm-bench
   deactivate
   ```

4. **AMD-SHARK Attention Benchmarks**:
   ```bash
   sudo pkill -f gemm-bench
   source venv/bin/activate
   for device in $(seq 0 2); do (sudo build/gemm-bench --device=$device &); done
   ./gb run --backends=amdsharkfa --suite=flash_attention --repeat=1 --output=results/amd-sharkfa_llama_sdxl_attention.csv
   sudo pkill -f gemm-bench
   deactivate
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.
