import os
import iree.compiler as ireec
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import itertools

def generate_attention_shapes(
    batch_sizes : list[int], 
    head_counts : list[int], 
    head_dims : list[int], 
    seq_lengths : list[int], 
    datatypes : list[int]):

    # batch_sizes = [1, 2, 4, 8, 16]
    # head_counts = [12, 24, 32, 36, 40, 42, 48, 64]
    # head_dims = [32, 64, 128]
    # seq_lengths = [64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64320]
    # datatypes = ["f8", "f16"]

    shapes = []
    for B, H, S_Q, S_KV, DH, datatype in itertools.product(batch_sizes, head_counts, seq_lengths, seq_lengths, head_dims, datatypes):
        shapes.append((B, H, S_Q, S_KV, DH, datatype))

    return shapes

def generate_mlir_content(B, H, S_Q, S_KV, DH, datatype):
    key_shape = f"[{B},{H},{S_KV},{DH}]"
    query_shape = f"[{B},{H},{S_Q},{DH}]"
    value_shape = f"[{B},{H},{S_KV},{DH}]"
    output_shape = f"[{B},{H},{S_Q},{DH}]"
    mlir_dtype = 'f16'
    mlir_template = f"""
module {{
    func.func @main_0(%295 : !torch.vtensor<{query_shape},{mlir_dtype}>, %298 : !torch.vtensor<{key_shape},{mlir_dtype}>, %301 : !torch.vtensor<{value_shape},{mlir_dtype}>) -> !torch.vtensor<{output_shape},{mlir_dtype}> {{
        %false_371 = torch.constant.bool false
        %float0.000000e00 = torch.constant.float 0.000000e+00
        %none_372 = torch.constant.none
        %none_373 = torch.constant.none
        %282:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%295, %298, %301, %float0.000000e00, %false_371, %none_372, %none_373) : (!torch.vtensor<{query_shape},{mlir_dtype}>, !torch.vtensor<{key_shape},{mlir_dtype}>, !torch.vtensor<{value_shape},{mlir_dtype}>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<{output_shape},{mlir_dtype}>, !torch.vtensor<[{B},{H},{S_Q}], f32>)
        return %282#0 : !torch.vtensor<{output_shape},{mlir_dtype}>
    }}
}} 
"""
    return mlir_template

def compile_shape(shape):
    B, H, S_Q, S_KV, DH, datatype = shape
    
    # Generate MLIR content
    mlir_content = generate_mlir_content(B, H, S_Q, S_KV, DH, datatype)
    
    # Generate filenames
    mlir_filename = f"attention/mlir/attention_B{B}_H{H}_SQ{S_Q}_SKV{S_KV}_DH{DH}_{datatype}.mlir"
    vmfb_filename = f"attention/vmfb/attention_B{B}_H{H}_SQ{S_Q}_SKV{S_KV}_DH{DH}_{datatype}.vmfb"
    
    # Write MLIR content to file
    with open(mlir_filename, 'w') as f:
        f.write(mlir_content)
    
    # Compile MLIR to VMFB
    compile_options = ireec.CompilerOptions()
    compile_options.hal_target_backends = ["rocm"]
    compile_options.rocm_target_chip = "gfx942"
    
    try:
        compiled_binary = ireec.compile_str(
            mlir_content,
            target_backends=compile_options.hal_target_backends,
            input_type="torch",
            output_format="FLATBUFFER_BINARY",
            extra_args=[
                f"--iree-rocm-target-chip={compile_options.rocm_target_chip}",
                "--iree-global-opt-propagate-transposes=true",
                f"--iree-codegen-transform-dialect-library=attention_and_matmul_spec_{datatype}.mlir",
                "--iree-opt-outer-dim-concat=true",
                "--iree-opt-const-eval=false",
                "--iree-opt-data-tiling=false",
                "--iree-rocm-waves-per-eu=2",
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-llvmgpu-use-vector-distribution",
                "--iree-codegen-gpu-native-math-precision=true",
                "--iree-flow-enable-aggressive-fusion",
                # f"--dump-compilation-phases-to=compile_phases_{B}_{H}_{S_Q}_{S_KV}_{DH}_{datatype}",
            ]
        )
        
        # Write the compiled binary to the VMFB file
        with open(vmfb_filename, 'wb') as f:
            f.write(compiled_binary)
        
        return f"Successfully compiled {mlir_filename} to {vmfb_filename}"
    except Exception as e:
        return f"Error compiling {mlir_filename}: {str(e)}"

def llama70battention():
    return generate_attention_shapes(
        batch_sizes=[1, 2, 4],
        head_counts=[32, 40, 64],
        head_dims=[128],
        seq_lengths=[1024, 2048, 4096],
        datatypes=["f8", "f16"],
    )

SDXL_ATTN = [
    (2, 10, 4096, 4096, 64, "f16"),
    (2, 10, 4096, 64, 64, "f16"),
    (2, 10, 1024, 1024, 64, "f16"),
    (2, 20, 1024, 64, 64, "f16"),
    (2, 10, 4096, 4096, 64, "f8"),
    (2, 10, 4096, 64, 64, "f8"),
    (2, 10, 1024, 1024, 64, "f8"),
    (2, 20, 1024, 64, 64, "f8"),
]

if __name__ == "__main__":
    shapes = []
    shapes.extend(llama70battention())
    shapes.extend(SDXL_ATTN)
    print(f"Generated {len(shapes)} attention shapes.")
    
    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")
    
    with Pool(num_cpus) as pool:
        results = list(tqdm(pool.imap(compile_shape, shapes), total=len(shapes)))
    
    error_count = 0
    for result in results:
        if 'error' in result.lower():
            # print(result)
            error_count += 1
    print(f'{len(shapes) - error_count} Success, {error_count} Failed out of {len(shapes)} shapes')

    print("Compilation process completed.")
