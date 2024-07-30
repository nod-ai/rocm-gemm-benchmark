import os
import iree.compiler as ireec
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import itertools

def generate_attention_shapes():
    batch_sizes = [1, 2, 4]
    head_counts = [12, 16, 24, 32, 40, 48, 64]
    seq_lengths = [64, 128, 256, 384, 512, 768, 1024, 2048, 4096, 8192]
    head_dims = [16, 32, 64, 128, 256]
    datatypes = ["f16"]

    shapes = []
    for B, H, S, DH, datatype in itertools.product(batch_sizes, head_counts, seq_lengths, head_dims, datatypes):
        S_Q = S
        S_KV = S

        shapes.append((B, H, S_Q, S_KV, DH, datatype))
        
        if S_KV > 64:
            shapes.append((B, H, S_KV // 2, S_KV, DH, datatype))
        if S_Q > 64:
            shapes.append((B, H, S_Q, S_Q // 2, DH, datatype))

    return shapes

def generate_mlir_content(B, H, S_Q, S_KV, DH, datatype):
    key_shape = f"[{B},{H},{S_KV},{DH}]"
    query_shape = f"[{B},{H},{S_Q},{DH}]"
    value_shape = f"[{B},{H},{S_KV},{DH}]"
    output_shape = f"[{B},{H},{S_Q},{DH}]"

    mlir_template = f"""
module {{
    func.func @main_0(%295 : !torch.vtensor<{query_shape},{datatype}>, %298 : !torch.vtensor<{key_shape},{datatype}>, %301 : !torch.vtensor<{value_shape},{datatype}>) -> !torch.vtensor<{output_shape},{datatype}> {{
        %false_371 = torch.constant.bool false
        %float0.000000e00 = torch.constant.float 0.000000e+00
        %none_372 = torch.constant.none
        %none_373 = torch.constant.none
        %282:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%295, %298, %301, %float0.000000e00, %false_371, %none_372, %none_373) : (!torch.vtensor<{query_shape},{datatype}>, !torch.vtensor<{key_shape},{datatype}>, !torch.vtensor<{value_shape},{datatype}>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<{output_shape},{datatype}>, !torch.vtensor<[{B},{H},{S_Q}], f32>)
        return %282#0 : !torch.vtensor<{output_shape},{datatype}>
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
                "--iree-opt-outer-dim-concat=true",
                "--iree-opt-const-eval=false",
                "--iree-opt-data-tiling=false",
                "--iree-rocm-waves-per-eu=2",
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-llvmgpu-use-vector-distribution",
                "--iree-codegen-gpu-native-math-precision=true",
                "--iree-flow-enable-aggressive-fusion",
                f"--dump-compilation-phases-to=compile_phases_{B}_{H}_{S_Q}_{S_KV}_{DH}_{datatype}",
            ]
        )
        
        # Write the compiled binary to the VMFB file
        with open(vmfb_filename, 'wb') as f:
            f.write(compiled_binary)
        
        return f"Successfully compiled {mlir_filename} to {vmfb_filename}"
    except Exception as e:
        return f"Error compiling {mlir_filename}: {str(e)}"

if __name__ == "__main__":
    shapes = generate_attention_shapes()
    print(f"Generated {len(shapes)} attention shapes.")
    
    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")
    
    with Pool(num_cpus) as pool:
        results = list(tqdm(pool.imap(compile_shape, shapes), total=len(shapes)))
    
    for result in results:
        if 'error' in result.lower():
            print(result)

    print("Compilation process completed.")
