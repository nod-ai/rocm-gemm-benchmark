// Transform dialect specification for attention on MI300 with MFMA.
// This script only supports variants of attention with a sequence
// length that is a multiple of 64. There are two near duplicate
// because we need different tile sizes when the head dimension is 512.
// TODO: Figure out how to parameterize the tile sizes without duplicating
// the attention function.

#layout_f8 = #iree_gpu.mma_layout<MFMA_F8E4M3FNUZ_16x16x32_F32>
#layout_16 = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
#layout = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>

module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Attention
//===----------------------------------------------------------------------===//

  // Utility matching for finding all undistributed fills.
  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["linalg.fill"] : !transform.any_op
    %0 = transform.get_parent_op %arg0 {allow_empty_results, nth_parent = 2 : i64, op_name = "scf.forall"} : (!transform.any_op) -> !transform.any_op
    transform.match.operation_empty %0 : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @get_undistributed_fills(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield %0 : !transform.any_op
  }

  // Script for FA2 transform pipeline when head_dim % 64 = 0.
  transform.named_sequence @__attention_main(%variant_op: !transform.any_op {transform.readonly}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %elementwise = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================

    %blocked_el, %wg_forall =
    transform.structured.tile_using_forall %elementwise tile_sizes [1, 1, 128, 0]
      ( mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %blocked_att, %wg_forall2 = transform.structured.fuse_into_containing_op %attention into %wg_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.iree.populate_workgroup_count_region_using_num_threads_slice %wg_forall2 : (!transform.any_op) -> ()

    // Cleanup
    %attfunc = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %attfunc {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %attfunc : !transform.any_op

    // Convert to online attention
    // ==========================================
    transform.iree.convert_to_online_attention %blocked_att : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %attfunc {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %attfunc : !transform.any_op

    %func = transform.apply_registered_pass "linalg-fuse-elementwise-ops" to %attfunc : (!transform.any_op) -> (!transform.any_op)

    // Tile along K2
    %online_att = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %k2_for = transform.structured.tile_using_for %online_att tile_sizes [0, 0, 0, 0, 64, 0]: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Promote key and value operands
    // ==========================================
    %attt = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %promoted_att, %alloc0, %alloc1 = transform.iree.promote_operands %attt [1, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // This is a hack... We distribute the loop and the merging seperately and just assume they are fused.
    %warp_attention = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %warp_merge = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %warp_attention tile_sizes[0, 0, 32, 0, 0, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %warp_merge tile_sizes[0, 0, 32, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill = transform.split_handle %fills : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %acc_fill tile_sizes[0, 0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %max_fill tile_sizes[0, 0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %sum_fill tile_sizes[0, 0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Decompose attention
    %func2 = transform.apply_registered_pass "iree-linalg-ext-decompose-attention" to %func : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func2 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func2 : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func_3 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, 4, 1] subgroup_size = 64 sync_after_distribution = false : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    %func_9 = transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    %intrinsic = transform.param.constant #layout -> !transform.any_param

    %mma = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %mma1, %mma2 = transform.split_handle %mma : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %mma "iree.amdgpu.mma" = %intrinsic : !transform.any_op, !transform.any_param

    transform.apply_registered_pass "iree-llvmgpu-cast-type-to-fit-mma" to %func_9 : (!transform.any_op) -> (!transform.any_op)

    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.set_contraction_layout_attributes %contract1, %intrinsic { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %intrinsic : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    %distribute_func_2 = transform.iree.amdgpu_distribute_vectors %distribute_func : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op
    %distribute_func_3 = transform.apply_registered_pass "iree-codegen-optimize-tensor-insert-extract-slices" to %distribute_func_2 : (!transform.any_op) -> (!transform.any_op)
    transform.print %distribute_func_3 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    transform.iree.gpu_distribute_shared_memory_copy %distribute_func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %distribute_func_3 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_3 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %distribute_func_3 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_3 : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.reduce_shared_memory_bank_conflicts %func_11 : (!transform.any_op) -> ()
    transform.yield
  }

  // Script for FA2 transform pipeline for FP8 attention.
  transform.named_sequence @__attention_main_f8(%variant_op: !transform.any_op {transform.readonly}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %blocked_att, %wg_forall =
    transform.structured.tile_using_forall %attention tile_sizes [1, 128, 0, 0, 0]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %wg_forall : (!transform.any_op) -> ()

    // Cleanup
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Convert to online attention
    // ==========================================
    transform.iree.convert_to_online_attention %blocked_att : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Tile along K2
    %online_att = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %k2_for = transform.structured.tile_using_for %online_att tile_sizes [0, 0, 0, 64, 0]: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Promote key and value operands
    // ==========================================
    %attt = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %promoted_att, %alloc0, %alloc1 = transform.iree.promote_operands %attt [1, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // This is a hack... We distribute the loop and the merging seperately and just assume they are fused.
    %warp_attention = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %warp_merge = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %warp_attention tile_sizes[0, 32, 0, 0, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %warp_merge tile_sizes[0, 32, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill = transform.split_handle %fills : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %acc_fill tile_sizes[0, 32, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %max_fill tile_sizes[0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %sum_fill tile_sizes[0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Decompose attention
    // TODO: Fix erroring out here!
    %func2 = transform.apply_registered_pass "iree-linalg-ext-decompose-attention" to %func : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func2 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func2 : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func_3 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, 4, 1] subgroup_size = 64 sync_after_distribution = false : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    %func_9 = transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    %intrinsic = transform.param.constant #layout_f8 -> !transform.any_param

    %mma = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %mma1, %mma2 = transform.split_handle %mma : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %mma "iree.amdgpu.mma" = %intrinsic : !transform.any_op, !transform.any_param

    transform.apply_registered_pass "iree-llvmgpu-cast-type-to-fit-mma" to %func_9 : (!transform.any_op) -> (!transform.any_op)

    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.set_contraction_layout_attributes %contract1, %intrinsic { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %intrinsic : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    %distribute_func_2 = transform.iree.amdgpu_distribute_vectors %distribute_func : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    transform.iree.gpu_distribute_shared_memory_copy %distribute_func_2 : (!transform.any_op) -> ()
    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.reduce_shared_memory_bank_conflicts %func_11 : (!transform.any_op) -> ()
    transform.yield
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention(%attention: !transform.any_op {transform.readonly}) {
    %func = transform.get_parent_op %attention {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main, { llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" } }> -> !transform.any_param
    transform.annotate %func "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[3], 64 : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention_f8(%attention: !transform.any_op {transform.readonly}) {
    %func = transform.get_parent_op %attention {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main_f8, { llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" } }> -> !transform.any_param
    transform.annotate %func "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention_f8(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    // Ensures K2 is divisible by 64.
    %in1 = transform.get_operand %attention[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in1 = tensor<?x?x?xf8E4M3FNUZ> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in1[1], 64 : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

//===----------------------------------------------------------------------===//
// Tuning infra
//===----------------------------------------------------------------------===//

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                            %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    // transform.print %op {name = "Applied"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_broadcast_rhs_mmt_i8_i8_i32(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    // transform.print %root {name = "Generic"} : !transform.any_op
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?x?xi8>, %rhs: tensor<?x?xi8>, %out: tensor<?x?x?xi32>):
      %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%lhs, %rhs : tensor<?x?x?xi8>, tensor<?x?xi8>) outs(%out : tensor<?x?x?xi32>) {
        ^bb0(%in: i8, %in_0: i8, %acc: i32):
          %22 = arith.extsi %in : i8 to i32
          %23 = arith.extsi %in_0 : i8 to i32
          %24 = arith.muli %22, %23 : i32
          %25 = arith.addi %acc, %24 : i32
          linalg.yield %25 : i32
        } -> tensor<?x?x?xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }

// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Convolution tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_conv_2d_nhwc_hwcf_Bx32x32x1280x3x3x1280(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<?x34x34x1280xi8>, %rhs: tensor<3x3x1280x1280xi8>, %out: tensor<?x32x32x1280xi32>):
      %13 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%lhs, %rhs : tensor<?x34x34x1280xi8>, tensor<3x3x1280x1280xi8>)
        outs(%out : tensor<?x32x32x1280xi32>) -> tensor<?x32x32x1280xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 320, 1, 1, 160]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
          , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_Bx32x32x1280x3x3x2560(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<?x34x34x2560xi8>, %rhs: tensor<3x3x2560x1280xi8>, %out: tensor<?x32x32x1280xi32>):
      %13 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%lhs, %rhs : tensor<?x34x34x2560xi8>, tensor<3x3x2560x1280xi8>)
        outs(%out : tensor<?x32x32x1280xi32>) -> tensor<?x32x32x1280xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 320, 1, 1, 160]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
          , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_Bx64x64x1280x3x3x1280(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<?x66x66x1280xi8>, %rhs: tensor<3x3x1280x1280xi8>, %out: tensor<?x64x64x1280xi32>):
      %13 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%lhs, %rhs : tensor<?x66x66x1280xi8>, tensor<3x3x1280x1280xi8>)
        outs(%out : tensor<?x64x64x1280xi32>) -> tensor<?x64x64x1280xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 320, 1, 1, 160]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
          , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_Bx128x128x320x3x3x640(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<?x130x130x640xi8>, %rhs: tensor<3x3x640x320xi8>, %out: tensor<?x128x128x320xi32>):
      %13 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%lhs, %rhs : tensor<?x130x130x640xi8>, tensor<3x3x640x320xi8>)
        outs(%out : tensor<?x128x128x320xi32>) -> tensor<?x128x128x320xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 64, 1, 1, 128]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_Bx128x128x640x3x3x640(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<?x130x130x640xi8>, %rhs: tensor<3x3x640x640xi8>, %out: tensor<?x128x128x640xi32>):
      %13 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%lhs, %rhs : tensor<?x130x130x640xi8>, tensor<3x3x640x640xi8>)
        outs(%out : tensor<?x128x128x640xi32>) -> tensor<?x128x128x640xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 128, 1, 1, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_Bx128x128x320x3x3x960(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<?x130x130x960xi8>, %rhs: tensor<3x3x960x320xi8>, %out: tensor<?x128x128x320xi32>):
      %13 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%lhs, %rhs : tensor<?x130x130x960xi8>, tensor<3x3x960x320xi8>)
        outs(%out : tensor<?x128x128x320xi32>) -> tensor<?x128x128x320xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 320, 1, 1, 160]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>,
              subgroup_m_count = 1, subgroup_n_count = 5>
          , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_Bx128x128x320x3x3x320(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<?x130x130x320xi8>, %rhs: tensor<3x3x320x320xi8>, %out: tensor<?x128x128x320xi32>):
      %13 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%lhs, %rhs : tensor<?x130x130x320xi8>, tensor<3x3x320x320xi8>)
        outs(%out : tensor<?x128x128x320xi32>) -> tensor<?x128x128x320xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 64, 1, 1, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [128, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
              subgroup_m_count = 1, subgroup_n_count = 2>
          , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Batch matmul tuning
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Broadcast lhs mmt tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_broadcast_rhs_mmt_Bx1024x10240x1280(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x1024x1280xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 320, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
           subgroup_m_count = 4, subgroup_n_count = 2>
         , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx1024x1280x5120(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x1024x5120xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 80, 256]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
           subgroup_m_count = 4, subgroup_n_count = 1>
         , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx1024x1280x1280(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x1024x1280xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x1280xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 160, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
           subgroup_m_count = 2, subgroup_n_count = 2>
         , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x5120x640(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x4096x640xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<5120x640xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 320, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
           subgroup_m_count = 4, subgroup_n_count = 2>
         , prefetch_shared_memory, reorder_workgroups = "transpose"}>
      > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x640x640(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x4096x640xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<640x640xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 320, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
           subgroup_m_count = 4, subgroup_n_count = 2>
         , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }


  transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x640x2560(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x4096x2560xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<640x2560xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 160, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
           subgroup_m_count = 4, subgroup_n_count = 1>
         , prefetch_shared_memory}>
      > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Contraction tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_matmul_like_Bx20x1024x64x1280_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x1024x1280xi8>, %rhs: tensor<20x64x1280xi8>, %out: tensor<?x20x1024x64xi32>):
      %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x1024x1280xi8>, tensor<20x64x1280xi8>)
        outs(%out : tensor<?x20x1024x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %18 = arith.extsi %in : i8 to i32
        %19 = arith.extsi %in_0 : i8 to i32
        %20 = arith.muli %18, %19 : i32
        %21 = arith.addi %acc, %20 : i32
        linalg.yield %21 : i32
      } -> tensor<?x20x1024x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 160, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
            subgroup_m_count = 2, subgroup_n_count = 2>
        , prefetch_shared_memory}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_Bx20x64x64x2048_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x64x2048xi8>, %rhs: tensor<20x64x2048xi8>, %out: tensor<?x20x64x64xi32>):
      %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x64x2048xi8>, tensor<20x64x2048xi8>)
        outs(%out : tensor<?x20x64x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %18 = arith.extsi %in : i8 to i32
        %19 = arith.extsi %in_0 : i8 to i32
        %20 = arith.muli %18, %19 : i32
        %21 = arith.addi %acc, %20 : i32
        linalg.yield %21 : i32
      } -> tensor<?x20x64x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 16, 256, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
            subgroup_m_count = 1, subgroup_n_count = 4>
        , prefetch_shared_memory}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_Bx10x64x64x2048_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x64x2048xi8>, %rhs: tensor<10x64x2048xi8>, %out: tensor<?x10x64x64xi32>):
      %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x64x2048xi8>, tensor<10x64x2048xi8>)
        outs(%out : tensor<?x10x64x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %18 = arith.extsi %in : i8 to i32
        %19 = arith.extsi %in_0 : i8 to i32
        %20 = arith.muli %18, %19 : i32
        %21 = arith.addi %acc, %20 : i32
        linalg.yield %21 : i32
      } -> tensor<?x10x64x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 16, 320, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
            subgroup_m_count = 1, subgroup_n_count = 4>
        , prefetch_shared_memory}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_BxBx20x1024x64_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x1024x1280xi8>, %rhs: tensor<?x20x64x1280xi8>, %out: tensor<?x?x20x1024x64xi32>):
      %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                                             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                                             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x1024x1280xi8>, tensor<?x20x64x1280xi8>)
        outs(%out : tensor<?x?x20x1024x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %24 = arith.extsi %in : i8 to i32
        %25 = arith.extsi %in_0 : i8 to i32
        %26 = arith.muli %24, %25 : i32
        %27 = arith.addi %acc, %26 : i32
        linalg.yield %27 : i32
      } -> tensor<?x?x20x1024x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 256, 64, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>,
            subgroup_m_count = 2, subgroup_n_count = 2>
        , prefetch_shared_memory}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_BxBx20x64x64x2048_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x64x2048xi8>, %rhs: tensor<?x20x64x2048xi8>, %out: tensor<?x?x20x64x64xi32>):
      %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                                             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                                             affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x64x2048xi8>, tensor<?x20x64x2048xi8>)
        outs(%out : tensor<?x?x20x64x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %24 = arith.extsi %in : i8 to i32
        %25 = arith.extsi %in_0 : i8 to i32
        %26 = arith.muli %24, %25 : i32
        %27 = arith.addi %acc, %26 : i32
        linalg.yield %27 : i32
      } -> tensor<?x?x20x64x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 32, 160, 256]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
            subgroup_m_count = 2, subgroup_n_count = 2>
        , prefetch_shared_memory}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

// TUNING_SPEC_END DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // Attention.
        @match_attention -> @custom_attention,
        @match_attention_f8 -> @custom_attention_f8

        // TUNING_MATCH_BEGIN DO NOT REMOVE

        // Matmul.

        // Convolution.

        , @match_conv_2d_nhwc_hwcf_Bx32x32x1280x3x3x1280 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_Bx32x32x1280x3x3x2560 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_Bx64x64x1280x3x3x1280 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_Bx128x128x320x3x3x640 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_Bx128x128x640x3x3x640 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_Bx128x128x320x3x3x960 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_Bx128x128x320x3x3x320 -> @apply_op_config

        // Batch matmul.

        // Broadcast rhs mmt.
        , @match_broadcast_rhs_mmt_Bx1024x10240x1280 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx1024x1280x5120 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx1024x1280x1280 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx4096x5120x640 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx4096x640x640 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx4096x640x2560 -> @apply_op_config

        // Contration.
        , @match_matmul_like_Bx20x1024x64x1280_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_Bx20x64x64x2048_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_Bx10x64x64x2048_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_BxBx20x1024x64_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_BxBx20x64x64x2048_i8xi8xi32 -> @apply_op_config

        // TUNING_MATCH_END DO NOT REMOVE
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module