from enum import Enum

class DeviceDtype(Enum):
    FP8 = 1
    FP16 = 2
    BF16 = 3

def device_sizeof(dtype : DeviceDtype | str) -> int:
    return 1

def get_problem_compute(algorithm : str, *args: float) -> tuple[float, float]:
    flops = 0
    bytes = 0
    
    if algorithm == 'gemm':
        M, N, K, dtype = args
        bpe = device_sizeof(dtype)
        flops = 2 * M * N * K
        bytes = bpe * (M * K + N * K + M * N)
    elif algorithm == 'attention':
        B, H, S_Q, S_KV, DH, dtype = args
        bpe = device_sizeof(dtype)
        flops = 4 * S_Q * S_KV * DH * B * H
        bytes = bpe * B * H * (2 * S_KV * DH + 2 * S_Q * DH + S_Q * S_KV)
    
    return flops, bytes
