import ctypes
from ctypes import c_int, c_char, c_double, c_float, POINTER
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libgemm_bench.so')  # Adjust the path as needed

class Problem(ctypes.Structure):
    _fields_ = [
        ("M", c_int),
        ("N", c_int),
        ("K", c_int),
        ("tag", c_char * 64),
        ("dtype", c_char * 8),
        ("A", c_char),
        ("B", c_char)
    ]

class Solution(ctypes.Structure):
    _fields_ = [("name", c_char * 32)]

class Configuration(ctypes.Structure):
    _fields_ = [
        ("fclk", c_int),
        ("mclk", c_int),
        ("gfxclk", c_int)
    ]

class Result(ctypes.Structure):
    _fields_ = [
        ("ok", c_int),
        ("warm_iterations", c_int),
        ("mean_microseconds", c_double),
        ("min_sclk", c_double),
        ("mean_sclk", c_double),
        ("max_sclk", c_double),
        ("min_mclk", c_double),
        ("mean_mclk", c_double),
        ("max_mclk", c_double),
        ("min_fclk", c_double),
        ("mean_fclk", c_double),
        ("max_fclk", c_double),
        ("device", c_int)
    ]

# Define function prototypes
lib.initialize_gemm_pipeline.argtypes = []
lib.initialize_gemm_pipeline.restype = None

lib.destroy_gemm_pipeline.argtypes = []
lib.destroy_gemm_pipeline.restype = None

lib.set_device.argtypes = [c_int]
lib.set_device.restype = None

lib.run_gemm.argtypes = [POINTER(Problem), POINTER(Solution), POINTER(Configuration)]
lib.run_gemm.restype = Result

lib.get_sclk_data.argtypes = [POINTER(c_int)]
lib.get_sclk_data.restype = POINTER(c_float)

lib.get_mclk_data.argtypes = [POINTER(c_int)]
lib.get_mclk_data.restype = POINTER(c_float)

lib.get_temperature_data.argtypes = [POINTER(c_int)]
lib.get_temperature_data.restype = POINTER(c_float)

lib.get_power_data.argtypes = [POINTER(c_int)]
lib.get_power_data.restype = POINTER(c_float)

def initialize():
    lib.initialize_gemm_pipeline()

def destroy():
    lib.destroy_gemm_pipeline()

def set_device(device_id):
    lib.set_device(device_id)

def run_gemm(problem, solution, configuration):
    result = lib.run_gemm(ctypes.byref(problem), ctypes.byref(solution), ctypes.byref(configuration))
    return result

def get_float_array(func):
    size = c_int()
    data_ptr = func(ctypes.byref(size))
    return np.ctypeslib.as_array(data_ptr, shape=(size.value,))

def get_sclk_data():
    return get_float_array(lib.get_sclk_data)

def get_mclk_data():
    return get_float_array(lib.get_mclk_data)

def get_temperature_data():
    return get_float_array(lib.get_temperature_data)

def get_power_data():
    return get_float_array(lib.get_power_data)