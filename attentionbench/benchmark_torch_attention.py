import torch
import time
import csv
from typing import List, Tuple
from tqdm import tqdm
from attention_utils import *

device = "cuda:0"

class TestModule(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

arg_to_torch_dtype = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32
}

def run_benchmark(shape: tuple[int, int, int, int, int, str]) -> tuple[float, float, float]:
    B, H, S_Q, S_KV, DH, dtype = shape

    torch_dtype = arg_to_torch_dtype[dtype]
    
    q = torch.rand([B, H, S_Q, DH], dtype=torch_dtype, device=device)
    k = torch.rand([B, H, S_KV, DH], dtype=torch_dtype, device=device)
    v = torch.rand([B, H, S_KV, DH], dtype=torch_dtype, device=device)
    
    test_module = torch.compile(TestModule(), dynamic=True)
    test_module.to(device=device, dtype=torch_dtype)
    
    # Warmup
    for _ in range(10):
        test_module(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    eval_steps = 100
    for _ in range(eval_steps):
        test_module(q, k, v)
    torch.cuda.synchronize()
    
    mean_microseconds = (time.time() - start_time) * 1e6 / eval_steps
    
    flops = 4 * S_Q * S_KV * DH * B * H
    bytes = B * H * 2 * (2 * S_KV * DH + 2 * S_Q * DH + S_Q * S_KV)
    arithmetic_intensity = flops / bytes
    tflops_per_second = (flops / 1e12) / (mean_microseconds / 1e6)
    
    return mean_microseconds, arithmetic_intensity, tflops_per_second

def main():
    input_csv = "results/sharkfa.csv"
    output_csv = "torch_llama_sdxl_attention.csv"
    
    shapes = read_shapes_from_csv(input_csv)
    results = []
    
    for index, shape in enumerate(tqdm(shapes)):
        B, H, S_Q, S_KV, DH, dtype = shape
        if dtype not in arg_to_torch_dtype:
            continue
        mean_microseconds, arithmetic_intensity, tflops = run_benchmark(shape)
        
        results.append((
            index, B, H, S_Q, S_KV, DH, dtype,
            round(mean_microseconds, 4),
            round(arithmetic_intensity, 4),
            round(tflops, 4)
        ))
    
    write_results_to_csv(results, output_csv)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    main()
