import torch
import time
import csv
from typing import List, Tuple
from tqdm import tqdm

device = "cuda:0"
dtype = torch.float16  # Using fp16 as specified

class TestModule(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def read_shapes_from_csv(filename: str) -> List[Tuple[int, int, int, int, int, int]]:
    shapes = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append((
                int(row['batch_size']),
                int(row['num_heads']),
                int(row['seqlen_query']),
                int(row['seqlen_key_value']),
                int(row['head_dim']),
                int(row['serial'])
            ))
    return shapes

def run_benchmark(shape: Tuple[int, int, int, int, int, int]) -> Tuple[float, float, float]:
    B, H, S_Q, S_KV, DH, serial = shape
    
    q = torch.rand([B, H, S_Q, DH], dtype=dtype, device=device)
    k = torch.rand([B, H, S_KV, DH], dtype=dtype, device=device)
    v = torch.rand([B, H, S_KV, DH], dtype=dtype, device=device)
    
    test_module = torch.compile(TestModule(), dynamic=True)
    test_module.to(device=device, dtype=dtype)
    
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

def write_results_to_csv(results: List[Tuple], output_filename: str):
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['serial', 'batch_size', 'num_heads', 'seqlen_query', 'seqlen_key_value', 'head_dim', 'dtype', 'mean_microseconds', 'arithmetic_intensity', 'tflops'])
        for result in results:
            writer.writerow(result)

def main():
    input_csv = "sharkfa_large_mul_raw_data.csv"
    output_csv = "torch_benchmark_results.csv"
    
    shapes = read_shapes_from_csv(input_csv)
    results = []
    
    for shape in tqdm(shapes):
        B, H, S_Q, S_KV, DH, serial = shape
        mean_microseconds, arithmetic_intensity, tflops = run_benchmark(shape)
        
        results.append((
            serial, B, H, S_Q, S_KV, DH, 'f16',
            round(mean_microseconds, 4),
            round(arithmetic_intensity, 4),
            round(tflops, 4)
        ))
    
    write_results_to_csv(results, output_csv)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    main()
