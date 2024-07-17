from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    FP16 = 1
    BF16 = 2
    FP8 = 3


# @dataclass(frozen=True)
# class GEMM:
#     tag: str
#     M: int
#     N: int
#     K: int
#     transposeA: str
#     transposeB: str
#     dtype: DataType
#     alpha: float = 1.0
#     beta: float = 0.0
#     group: str = "NA"

#     @property
#     def compute_bound(self):
#         """Is this GEMM compute (or memory) bound?"""
#         bytes_per_element = {DataType.FP16: 2, DataType.BF16: 2, DataType.FP8: 1}[
#             self.dtype
#         ]
#         return is_compute_bound(self.M, self.N, self.K, bytes_per_element)


# def is_compute_bound(M, N, K, bpe):
#     """Is this GEMM compute (or memory) bound?"""
#     magic_ratio = 64
#     flops = 2 * M * N * K
#     bytes = bpe * (M * K + K * N + M * N)
#     return flops > magic_ratio * bytes
