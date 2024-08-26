import os
import iree.compiler as ireec
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import subprocess
from pathlib import Path
import csv
from typing import Sequence
from collections import namedtuple
import argparse

def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logging.getLogger().info(command)
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    (
        stdout_v,
        stderr_v,
    ) = (
        proc.stdout,
        proc.stderr,
    )
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout
    logging.getLogger().error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stderr

def generate_mlir_content(image, conv_filter, stride, output, inputs_dtype, output_dtype):
    image_shape = ""
    for dim in image:
        image_shape += f"{dim}x"
    image_shape = image_shape[:-1]

    filter_shape = ""
    for dim in conv_filter:
        filter_shape += f"{dim}x"
    filter_shape = filter_shape[:-1]

    output_shape = ""
    for dim in output:
        output_shape += f"{dim}x"
    output_shape = output_shape[:-1]

    mlir_template = f"""
util.func public @main(%arg0: tensor<{image_shape}x{inputs_dtype}>, %arg1: tensor<{filter_shape}x{inputs_dtype}>) -> tensor<{output_shape}x{output_dtype}> {{
    %cst = arith.constant 0.0 : {output_dtype}
    %9 = tensor.empty() : tensor<{output_shape}x{output_dtype}>
    %10 = linalg.fill ins(%cst : {output_dtype}) outs(%9 : tensor<{output_shape}x{output_dtype}>) -> tensor<{output_shape}x{output_dtype}>
    %11 = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : vector<2xi64>, strides = dense<{stride}> : vector<2xi64>}} ins(%arg0, %arg1 : tensor<{image_shape}x{inputs_dtype}>, tensor<{filter_shape}x{inputs_dtype}>) outs(%10 : tensor<{output_shape}x{output_dtype}>) -> tensor<{output_shape}x{output_dtype}>
    util.return %11 : tensor<{output_shape}x{output_dtype}>
}}
"""
    return mlir_template

def compile_shape(image, conv_filter, stride, output, inputs_dtype, output_dtype, vmfb_dict):
    
    # Generate MLIR content
    mlir_content = generate_mlir_content(image, conv_filter, stride, output, inputs_dtype, output_dtype)
    
    # Generate filenames
    mlir_filename = f"conv/mlir/conv_2d_nchw_fchw_{output[0]}x{output[2]}x{output[3]}x{conv_filter[1]}x{conv_filter[2]}x{conv_filter[3]}x{conv_filter[0]}_{inputs_dtype}x{inputs_dtype}x{output_dtype}_stride{stride}.mlir"
    vmfb_filename = f"conv/vmfb/conv_2d_nchw_fchw_{output[0]}x{output[2]}x{output[3]}x{conv_filter[1]}x{conv_filter[2]}x{conv_filter[3]}x{conv_filter[0]}_{inputs_dtype}x{inputs_dtype}x{output_dtype}_stride{stride}.vmfb"
    
    # Write MLIR content to file
    with open(mlir_filename, 'w') as f:
        f.write(mlir_content)
    
    # Compile MLIR to VMFB
    exec_args = [
        "iree-compile",
        f"{mlir_filename}",
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx942",
        "-o",
        f"{vmfb_filename}",
    ]
    ret_value, stdout = run_iree_command(exec_args)
    
    vmfb_dict[vmfb_filename] = [image, conv_filter, inputs_dtype, output, output_dtype]
    if ret_value == 0:
        return f"Successfully compiled {mlir_filename} to {vmfb_filename}"

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)

def decode_output(bench_lines):
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )
    return benchmark_results

def bench_summary_process(ret_value, output):
    if ret_value == 1:
        # Output should have already been logged earlier.
        logging.getLogger().error("Running convolution benchmark failed. Exiting.")
        return

    bench_lines = output.decode().split("\n")[3:]
    benchmark_results = decode_output(bench_lines)
    logging.getLogger().info(benchmark_results)
    benchmark_mean_time = float(benchmark_results[10].time.split()[0])
    
    return benchmark_mean_time

def write_results_to_csv(results : list[tuple] | list[list] | list[dict], output_filename: str):
    if len(results) == 0:
        print('No valid results')
        return
    
    fieldnames = [
        'index', 
        'tag',
        'name',
        'image', 
        'conv_filter', 
        'output', 
        'input_dtype', 
        'output_dtype',
        'mean_microseconds',
        'arithmetic_intensity',
        'tflops',
        'ok'
    ]

    with open(output_filename, 'w', newline='') as f:
        if isinstance(results[0], list) or isinstance(results[0], tuple):
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        elif isinstance(results[0], dict):
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
        else:
            print('Invalid result format')
            return
        
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file updater.")
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set the logging level",
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    
    shapes = []
    print(f"Generated {len(shapes)} conv shapes.")
    
    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()
    INITIAL_CONVS = [
        ([1,64,230,230], [3,64,7,7], 2, [1,3,112,112], "f32", "f32", vmfb_dict),
    ]
    shapes.extend(INITIAL_CONVS)

    with Pool(num_cpus) as pool:
        results = list(tqdm(pool.starmap(compile_shape, shapes)))
    
    error_count = 0
    for result in results:
        if 'error' in result.lower():
            # print(result)
            error_count += 1
    print(f'{len(shapes) - error_count} Success, {error_count} Failed out of {len(shapes)} shapes')

    print("Compilation process completed.")

    repo_root = Path(__file__).parent.parent

    vmfb_dir = repo_root / Path('conv/vmfb')

    results = []
    tag = "conv"
    index = 0
    output_csv = "results/iree_conv.csv"

    for vmfb_filename, input_list in vmfb_dict.items():
        vmfb_filename = vmfb_filename.split("/")[-1]
        name = vmfb_filename.split(".")[0]
        image = input_list[0]
        conv_filter = input_list[1]
        input_dtype = input_list[2]
        output = input_list[3]
        output_dtype = input_list[4]
        
        image_shape = ""
        for dim in image:
            image_shape += f"{dim}x"
        image_shape = image_shape[:-1]
        image_shape += f"x{input_dtype}"

        filter_shape = ""
        for dim in conv_filter:
            filter_shape += f"{dim}x"
        filter_shape = filter_shape[:-1]
        filter_shape += f"x{input_dtype}"

        exec_args = [
            "iree-benchmark-module",
            f"--device=hip",
            "--device_allocator=caching",
            f"--module={vmfb_dir}/{vmfb_filename}",
            "--function=main",
            f"--input={image_shape}",
            f"--input={filter_shape}",
            "--benchmark_repetitions=10",
            "--benchmark_min_warmup_time=3.0",
        ]

        # iree benchmark command for full sdxl pipeline
        ret_value, cmd_out = run_iree_command(exec_args)
        ok = ret_value == 0
        benchmark_conv_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_conv_mean_time_us = benchmark_conv_mean_time_ms * 1000

        bytes_per_input = int(input_dtype[1:]) / 8
        batch = image[0]
        input_channels = image[1]
        width = image[2]
        height = image[3]
        k_width = conv_filter[2]
        k_height = conv_filter[3]
        output_channels = conv_filter[0]
        output_width = output[2]
        output_height = output[3]

        operation_per_pixel = k_width * k_height * input_channels * 2
        output_pixels_per_batch = output_width * output_height * output_channels
        
        flops = operation_per_pixel * output_pixels_per_batch * batch
        byte_count = batch * input_channels * width * height * bytes_per_input + batch * output_channels * output_width * output_height * bytes_per_input + k_width * k_height * input_channels * output_channels * bytes_per_input
        arithmetic_intensity = flops / byte_count
        tflops_per_second = (flops / 1e12) / (benchmark_conv_mean_time_us / 1e6)

        results.append((
            index, tag, name, str(image), str(conv_filter), str(output), input_dtype, output_dtype,
            round(benchmark_conv_mean_time_us, 4),
            round(arithmetic_intensity, 4),
            round(tflops_per_second, 4),
            ok
        ))
        index += 1

    write_results_to_csv(results, output_csv)
    print(f"Results written to {output_csv}")
        
        
