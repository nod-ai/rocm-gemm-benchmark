import os

# Function to generate MLIR content and write to a file
def generate_mlir_file(B, H, S_Q, S_KV, DH, datatype):
    key_shape = f"[{B},{H},{S_KV},{DH}]"
    query_shape = f"[{B},{H},{S_Q},{DH}]"
    value_shape = f"[{B},{H},{S_KV},{DH}]"
    output_shape = f"[{B},{H},{S_Q},{DH}]"

    mlir_template = f"""
func.func @main(%295 : !torch.vtensor<{key_shape},{datatype}>, %298 : !torch.vtensor<{query_shape},{datatype}>, %301 : !torch.vtensor<{value_shape},{datatype}>) -> !torch.vtensor<{output_shape},{datatype}> {{
    %false_371 = torch.constant.bool false
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %none_372 = torch.constant.none
    %none_373 = torch.constant.none
    %282:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%295, %298, %301, %float0.000000e00, %false_371, %none_372, %none_373) : (!torch.vtensor<{query_shape},{datatype}>, !torch.vtensor<{key_shape},{datatype}>, !torch.vtensor<{value_shape},{datatype}>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<{output_shape},{datatype}>, !torch.vtensor<[{B},{H},{S_Q}], f32>)
    return %282#0 : !torch.vtensor<{output_shape},{datatype}>
}} 
"""

    # Generate unique filename based on parameters
    filename = f"attention_B{B}_H{H}_SQ{S_Q}_SKV{S_KV}_DH{DH}_{datatype}.mlir"
    with open(filename, 'w') as f:
        f.write(mlir_template)

    return filename

# List of known attention shapes in popular LLM architectures
known_shapes = [
    (1, 42, 384, 64320, 64, "f16")
    # (1, 12, 512, 512, 64, "f16"),   # Example shape for BERT base
    # (1, 16, 1024, 1024, 64, "f16"), # Example shape for GPT-3 small
    # (1, 12, 384, 384, 64, "f16"),   # Example shape for some other model
]

# Function to add more shapes iteratively
def add_more_shapes(shape_list):
    # Example of adding more shapes
    shape_list.append((1, 8, 256, 256, 32, "f16"))    # Custom shape 1
    shape_list.append((1, 32, 2048, 2048, 128, "f32")) # Custom shape 2
    # Add more shapes as needed
    return shape_list

# Main script
if __name__ == "__main__":
    # Add more shapes to the known list
    # known_shapes = add_more_shapes(known_shapes)
    
    # Generate MLIR files for each shape in the list
    for shape in known_shapes:
        B, H, S_Q, S_KV, DH, datatype = shape
        filename = generate_mlir_file(B, H, S_Q, S_KV, DH, datatype)
        print(f"MLIR file '{filename}' generated successfully.")
