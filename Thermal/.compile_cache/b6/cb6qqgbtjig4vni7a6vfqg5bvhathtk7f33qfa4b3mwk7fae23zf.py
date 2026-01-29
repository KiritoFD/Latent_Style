# AOT ID: ['1_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vm/cvm26v4mnnk4kdfye7ex26mha7ylgp62zlfmhbnbkuz32q6y72pe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "bf16[128, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=tangents_1]
#   %sum_11 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0, 2, 3]), kwargs = {})
#   return %buf0
triton_red_fused_convolution_backward_0 = async_compile.triton('triton_red_fused_convolution_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1081344, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_convolution_backward_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 4)
    x1 = xindex // 4
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 4*r0_2 + 512*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ay/cayr5g5ks23wiuud6u3oi225bus3s7ay7se6qpbasvoth2zna55v.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf0 : Tensor "f32[4, 1024][1, 4]cuda:0" = PlaceHolder[target=buf0]
#   %sum_11 : Tensor "bf16[4][1]cuda:0" = PlaceHolder[target=sum_11]
#   %sum_11 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0, 2, 3]), kwargs = {})
#   %convert_element_type_484 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_11, torch.float32), kwargs = {})
#   return %sum_11,%convert_element_type_484
triton_red_fused__to_copy_convolution_backward_1 = async_compile.triton('triton_red_fused__to_copy_convolution_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_convolution_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8208, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_convolution_backward_1(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 4*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qy/cqy4i6hxcodco73xapztwwx65xtetc3rgf2ehis7clrkcupof6s7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_123 : Tensor "bf16[4, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=getitem_123]
#   %convert_element_type_483 : Tensor "f32[4, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_123, torch.float32), kwargs = {})
#   return %convert_element_type_483
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 46080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7n/c7nvq6scvfg2mxu37atiebzpofqvb7ibi4tvfiys5gruskx2vsbx.py
# Topologically Sorted Source Nodes: [h_22, input_42], Original ATen: [aten.silu, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_22 => convert_element_type_476, mul_72, sigmoid_40
#   input_42 => add_81, clone_56, mul_73, mul_74, sub_28, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, view_270, view_271
# Graph fragment:
#   %add_79 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_79]
#   %getitem_121 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=getitem_121]
#   %rsqrt_18 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_18]
#   %primals_150 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_150]
#   %primals_151 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_151]
#   %convert_element_type_476 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_72 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_56 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%mul_72,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_56, [128, 32, 4, 1024]), kwargs = {})
#   %sub_28 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %getitem_121), kwargs = {})
#   %mul_73 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_18), kwargs = {})
#   %view_271 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_73, [128, 128, 32, 32]), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_150, 0), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 2), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_17, 3), kwargs = {})
#   %mul_74 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_271, %unsqueeze_18), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_151, 0), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 2), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, 3), kwargs = {})
#   %add_81 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_21), kwargs = {})
#   return %add_81
triton_poi_fused_clone_native_group_norm_silu_3 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167773184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 131072
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/5c/c5cuqh74qjvxi2j7udoaxwkgl2ihqf7o3xlgw7zkq3vgjuxhj7hc.py
# Topologically Sorted Source Nodes: [h_22, input_42, input_43], Original ATen: [aten._to_copy, aten.fill, aten.silu, aten.clone, aten.sub, aten.mul, aten.add, aten.native_group_norm_backward]
# Source node to ATen node mapping:
#   h_22 => convert_element_type_476, mul_72, sigmoid_40
#   input_42 => clone_56
#   input_43 => sigmoid_41
# Graph fragment:
#   %getitem_122 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_122]
#   %add_81 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_81]
#   %add_79 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_79]
#   %convert_element_type_482 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_122, torch.float32), kwargs = {})
#   %full_default : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128, 32, 32], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_476 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_72 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_56 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%mul_72,), kwargs = {memory_format: torch.contiguous_format})
#   %sigmoid_41 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %sub_29 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_41), kwargs = {})
#   %mul_76 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sub_29), kwargs = {})
#   %add_82 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_76, 1), kwargs = {})
#   %mul_77 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_41, %add_82), kwargs = {})
#   %mul_78 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_482, %mul_77), kwargs = {})
#   %clone_57 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%mul_78,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_79 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_57, %clone_56), kwargs = {})
#   %view_272 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_79, [128, 128, 1024]), kwargs = {})
#   %sum_12 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_272, [2]), kwargs = {})
#   %view_273 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_57, [128, 128, 1024]), kwargs = {})
#   %sum_13 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_273, [2]), kwargs = {})
#   return %sum_12,%sum_13
triton_red_fused__to_copy_add_clone_fill_mul_native_group_norm_backward_silu_sub_4 = async_compile.triton('triton_red_fused__to_copy_add_clone_fill_mul_native_group_norm_backward_silu_sub_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_fill_mul_native_group_norm_backward_silu_sub_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134479872, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_fill_mul_native_group_norm_backward_silu_sub_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tl.sigmoid(tmp2)
        tmp4 = 1.0
        tmp5 = tmp4 - tmp3
        tmp6 = tmp2 * tmp5
        tmp7 = tmp6 + tmp4
        tmp8 = tmp3 * tmp7
        tmp9 = tmp1 * tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp9 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
        tmp18 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/mf/cmfawcak4hj5nlay52czdr2lebh772rviyp7corkv3nt7fwcy5va.py
# Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.native_group_norm, aten.native_group_norm_backward]
# Source node to ATen node mapping:
#   input_42 => squeeze_32, squeeze_33
# Graph fragment:
#   %sum_12 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=sum_12]
#   %sum_13 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=sum_13]
#   %getitem_121 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=getitem_121]
#   %rsqrt_18 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_18]
#   %squeeze_33 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_18, [2, 3]), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_33, -1), kwargs = {})
#   %squeeze_32 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_121, [2, 3]), kwargs = {})
#   %view_280 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_12, [128, 32, 4]), kwargs = {})
#   %view_281 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_13, [128, 32, 4]), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_32, -1), kwargs = {})
#   %mul_93 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_281, %unsqueeze_30), kwargs = {})
#   %sub_32 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_280, %mul_93), kwargs = {})
#   %mul_94 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_24), kwargs = {})
#   %sum_16 : Tensor "f32[32, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_94, [0]), kwargs = {})
#   %sum_17 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sum_13, [0]), kwargs = {})
#   return %sum_16,%sum_17
triton_red_fused_native_group_norm_native_group_norm_backward_5 = async_compile.triton('triton_red_fused_native_group_norm_native_group_norm_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_native_group_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 133120, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_group_norm_native_group_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x1 = xindex // 4
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x3 + 128*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x3 + 128*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + 32*r0_2), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + 32*r0_2), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 - tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/uf/cufcmbxd4osh7aefsjop6d3utlzjdotahipj5jsase7c6mhszt2e.py
# Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.native_group_norm, aten.native_group_norm_backward]
# Source node to ATen node mapping:
#   input_42 => squeeze_32, unsqueeze_16
# Graph fragment:
#   %sum_13 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=sum_13]
#   %primals_150 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_150]
#   %sum_15 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sum_15]
#   %getitem_121 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=getitem_121]
#   %sum_12 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=sum_12]
#   %unsqueeze_16 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_150, 0), kwargs = {})
#   %mul_80 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_12, %unsqueeze_16), kwargs = {})
#   %view_274 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_80, [128, 32, 4]), kwargs = {})
#   %sum_14 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_274, [2]), kwargs = {})
#   %mul_81 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_13, %unsqueeze_16), kwargs = {})
#   %view_275 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_81, [128, 32, 4]), kwargs = {})
#   %sum_15 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_275, [2]), kwargs = {})
#   %squeeze_32 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_121, [2, 3]), kwargs = {})
#   %mul_83 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_15, %squeeze_32), kwargs = {})
#   %sub_30 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_83, %sum_14), kwargs = {})
#   return %sum_15,%sub_30
triton_poi_fused_native_group_norm_native_group_norm_backward_6 = async_compile.triton('triton_poi_fused_native_group_norm_native_group_norm_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_native_group_norm_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 81920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_native_group_norm_backward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (4*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x2), None)
    tmp17 = tl.load(in_ptr3 + (4*x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (1 + 4*x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (2 + 4*x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (3 + 4*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp1
    tmp20 = tmp19 * tmp4
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp8
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp12
    tmp27 = tmp24 + tmp26
    tmp28 = tmp16 - tmp27
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vt/cvtyvn6auwq3sjhje6j5rnwwd72rsytti6xgbruaado5jzy3gpdn.py
# Topologically Sorted Source Nodes: [h_22, input_42, input_43], Original ATen: [aten._to_copy, aten.fill, aten.silu, aten.clone, aten.native_group_norm, aten.sub, aten.mul, aten.add, aten.native_group_norm_backward, aten.sigmoid]
# Source node to ATen node mapping:
#   h_22 => convert_element_type_476, mul_72, sigmoid_40
#   input_42 => clone_56, squeeze_32, squeeze_33, view_270
#   input_43 => sigmoid_41
# Graph fragment:
#   %getitem_122 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_122]
#   %add_81 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_81]
#   %rsqrt_18 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_18]
#   %primals_150 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_150]
#   %add_79 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_79]
#   %sub_30 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sub_30]
#   %getitem_121 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=getitem_121]
#   %sum_15 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sum_15]
#   %add_84 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0" = PlaceHolder[target=add_84]
#   %convert_element_type_482 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_122, torch.float32), kwargs = {})
#   %full_default : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128, 32, 32], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_476 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_72 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_56 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%mul_72,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_56, [128, 32, 4, 1024]), kwargs = {})
#   %sigmoid_41 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %sub_29 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_41), kwargs = {})
#   %mul_76 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sub_29), kwargs = {})
#   %add_82 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_76, 1), kwargs = {})
#   %mul_77 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_41, %add_82), kwargs = {})
#   %mul_78 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_482, %mul_77), kwargs = {})
#   %clone_57 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%mul_78,), kwargs = {memory_format: torch.contiguous_format})
#   %squeeze_33 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_18, [2, 3]), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_33, -1), kwargs = {})
#   %view_276 : Tensor "f32[1, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_150, [1, 32, 4]), kwargs = {})
#   %mul_82 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_24, %view_276), kwargs = {})
#   %squeeze_32 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_121, [2, 3]), kwargs = {})
#   %mul_84 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %squeeze_33), kwargs = {})
#   %mul_85 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %squeeze_33), kwargs = {})
#   %mul_86 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %squeeze_33), kwargs = {})
#   %mul_87 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, 0.000244140625), kwargs = {})
#   %neg : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_87,), kwargs = {})
#   %mul_88 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %squeeze_32), kwargs = {})
#   %mul_89 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_15, %squeeze_33), kwargs = {})
#   %mul_90 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, 0.000244140625), kwargs = {})
#   %sub_31 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_88, %mul_90), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[128, 32, 4, 1][128, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_82, -1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_87, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sub_31, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %view_277 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_57, [128, 32, 4, 1024]), kwargs = {})
#   %mul_91 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_277, %unsqueeze_25), kwargs = {})
#   %mul_92 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_270, %unsqueeze_27), kwargs = {})
#   %add_83 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %mul_92), kwargs = {})
#   %add_84 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_83, %unsqueeze_29), kwargs = {})
#   %view_279 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_84, [128, 128, 32, 32]), kwargs = {})
#   %convert_element_type_485 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_279, torch.bfloat16), kwargs = {})
#   %sigmoid_43 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %full_1 : Tensor "bf16[128, 32, 32, 128][131072, 4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 32, 32, 128], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_129 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_1, [0, 3, 1, 2]), kwargs = {})
#   %sub_33 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_129, %sigmoid_43), kwargs = {})
#   %mul_95 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sub_33), kwargs = {})
#   %add_85 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_95, 1), kwargs = {})
#   %mul_96 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_43, %add_85), kwargs = {})
#   %mul_97 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_485, %mul_96), kwargs = {})
#   return %add_84,%mul_97
triton_poi_fused__to_copy_add_clone_fill_mul_native_group_norm_native_group_norm_backward_sigmoid_silu_sub_7 = async_compile.triton('triton_poi_fused__to_copy_add_clone_fill_mul_native_group_norm_native_group_norm_backward_sigmoid_silu_sub_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_fill_mul_native_group_norm_native_group_norm_backward_sigmoid_silu_sub_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 134218240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_fill_mul_native_group_norm_native_group_norm_backward_sigmoid_silu_sub_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y5 = yindex
    x3 = xindex // 4
    y1 = yindex // 1024
    y0 = (yindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x4 + 128*y5), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x4 + 128*y5), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x4 + 128*y5), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 1.0
    tmp5 = tmp4 - tmp3
    tmp6 = tmp2 * tmp5
    tmp7 = tmp6 + tmp4
    tmp8 = tmp3 * tmp7
    tmp9 = tmp1 * tmp8
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp10
    tmp20 = tmp19 * tmp10
    tmp21 = tmp20 * tmp10
    tmp22 = 0.000244140625
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp13 + tmp24
    tmp26 = -tmp23
    tmp28 = tmp26 * tmp27
    tmp30 = tmp29 * tmp10
    tmp31 = tmp30 * tmp22
    tmp32 = tmp28 - tmp31
    tmp33 = tmp25 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.sigmoid(tmp14)
    tmp36 = tmp4 - tmp35
    tmp37 = tmp14 * tmp36
    tmp38 = tmp37 + tmp4
    tmp39 = tmp35 * tmp38
    tmp40 = tmp34 * tmp39
    tl.store(out_ptr1 + (y0 + 1024*x4 + 131072*y1), tmp40, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/uq/cuqwotjcjf6t3ofemzxouwqrjuu5v6iocgbehvqn4z4ftwfvzuh6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_97 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_97]
#   %sum_18 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=sum_18]
#   %sum_18 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_97, [0, 2, 3]), kwargs = {})
#   %convert_element_type_488 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_18, torch.float32), kwargs = {})
#   return %sum_18,%convert_element_type_488
triton_red_fused__to_copy_convolution_backward_8 = async_compile.triton('triton_red_fused__to_copy_convolution_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_convolution_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_convolution_backward_8(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 1024)
        r0_2 = r0_index // 1024
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0 + 131072*r0_2), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wo/cwo2zdb44vsmcfmcjbb2jxakjxz7tqz5cswv6tsxipg6baq6hg46.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_126 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_126]
#   %convert_element_type_487 : Tensor "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_126, torch.float32), kwargs = {})
#   return %convert_element_type_487
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 163840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wo/cwo7r2bu4j7qod6tivocgoetavjoh7bx4moicp3b7i2vs2m5mtr6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_125 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_125]
#   %mul_99 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_99]
#   %convert_element_type_486 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_125, torch.float32), kwargs = {})
#   %mul_100 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_486, %mul_99), kwargs = {})
#   %sum_19 : Tensor "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_100, [2, 3], True), kwargs = {dtype: torch.float32})
#   %view_283 : Tensor "f32[128, 128, 1024][131072, 1, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_100, [128, 128, 1024]), kwargs = {})
#   %convert_element_type_490 : Tensor "bf16[128, 128, 1024][131072, 1, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_283, torch.bfloat16), kwargs = {})
#   return %sum_19,%convert_element_type_490
triton_red_fused__to_copy_mul_sum_view_10 = async_compile.triton('triton_red_fused__to_copy_mul_sum_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33685504, 'r0_': 134217728}}
)
@triton.jit
def triton_red_fused__to_copy_mul_sum_view_10(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 1024*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp7 = tmp3.to(tl.float32)
        tl.store(out_ptr1 + (r0_2 + 1024*x3), tmp7, r0_mask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/fn/cfng75fzxqpl33jtm7soavrf5p7n7tg6ovdl6xbskli2xyehjfqk.py
# Topologically Sorted Source Nodes: [x_norm_16], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
# Source node to ATen node mapping:
#   x_norm_16 => clone_55, convert_element_type_467
# Graph fragment:
#   %getitem_125 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_125]
#   %mul_99 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_99]
#   %bmm_54 : Tensor "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_54]
#   %convolution_35 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_35]
#   %convert_element_type_486 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_125, torch.float32), kwargs = {})
#   %mul_100 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_486, %mul_99), kwargs = {})
#   %view_283 : Tensor "f32[128, 128, 1024][131072, 1, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_100, [128, 128, 1024]), kwargs = {})
#   %convert_element_type_499 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_54, torch.float32), kwargs = {})
#   %add_87 : Tensor "f32[128, 128, 1024][131072, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_283, %convert_element_type_499), kwargs = {})
#   %view_284 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_87, [128, 128, 32, 32]), kwargs = {})
#   %clone_60 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%view_284,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_467 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_35, torch.float32), kwargs = {})
#   %clone_55 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_467,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_101 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_60, %clone_55), kwargs = {})
#   %view_288 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_101, [128, 128, 1024]), kwargs = {})
#   %sum_20 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_288, [2]), kwargs = {})
#   %view_289 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_60, [128, 128, 1024]), kwargs = {})
#   %sum_21 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_289, [2]), kwargs = {})
#   return %sum_20,%sum_21
triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11 = async_compile.triton('triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67371008, 'r0_': 100663296}}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 4096*(((r0_2 % 32)) // 32) + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 32*(((r0_2 % 32)) // 32) + 1024*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 1024*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (x0 + 128*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
        tmp13 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, None)
    tl.store(out_ptr1 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7b/c7bdbc6sranhw2pvysv7nv4ke5ljpr3pdfzfrhh7roea2ghuotd5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_21 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=sum_21]
#   %squeeze_30 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=squeeze_30]
#   %sum_20 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=sum_20]
#   %sub_35 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sub_35]
#   %squeeze_31 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=squeeze_31]
#   %view_290 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_20, [128, 32, 4]), kwargs = {})
#   %sum_22 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_290, [2]), kwargs = {})
#   %view_291 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_21, [128, 32, 4]), kwargs = {})
#   %sum_23 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_291, [2]), kwargs = {})
#   %mul_103 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_23, %squeeze_30), kwargs = {})
#   %sub_35 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_103, %sum_22), kwargs = {})
#   %mul_104 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %squeeze_31), kwargs = {})
#   %mul_105 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %squeeze_31), kwargs = {})
#   %mul_106 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_105, %squeeze_31), kwargs = {})
#   %mul_107 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, 0.000244140625), kwargs = {})
#   %neg_1 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_107,), kwargs = {})
#   %mul_108 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %squeeze_30), kwargs = {})
#   %mul_109 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_23, %squeeze_31), kwargs = {})
#   %mul_110 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, 0.000244140625), kwargs = {})
#   %sub_36 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_108, %mul_110), kwargs = {})
#   return %sub_35,%sub_36
triton_poi_fused_native_group_norm_backward_12 = async_compile.triton('triton_poi_fused_native_group_norm_backward_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_backward_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x0), None)
    tmp9 = tl.load(in_ptr1 + (4*x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (2 + 4*x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (3 + 4*x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp8 - tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp17
    tmp20 = tmp19 * tmp17
    tmp21 = 0.000244140625
    tmp22 = tmp20 * tmp21
    tmp23 = -tmp22
    tmp24 = tmp23 * tmp7
    tmp25 = tmp6 * tmp17
    tmp26 = tmp25 * tmp21
    tmp27 = tmp24 - tmp26
    tl.store(out_ptr0 + (x0), tmp16, None)
    tl.store(in_out_ptr0 + (x0), tmp27, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/66/c66munv7zcv43glxhr376uz25ygqbhiro5uqvegv3mdjjqaqkl7g.py
# Topologically Sorted Source Nodes: [x_norm_16], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_norm_16 => clone_55, convert_element_type_467, view_263
# Graph fragment:
#   %getitem_125 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_125]
#   %mul_99 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_99]
#   %bmm_54 : Tensor "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_54]
#   %squeeze_31 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=squeeze_31]
#   %convolution_35 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_35]
#   %sub_35 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sub_35]
#   %sub_36 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sub_36]
#   %convert_element_type_486 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_125, torch.float32), kwargs = {})
#   %mul_100 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_486, %mul_99), kwargs = {})
#   %view_283 : Tensor "f32[128, 128, 1024][131072, 1, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_100, [128, 128, 1024]), kwargs = {})
#   %convert_element_type_499 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_54, torch.float32), kwargs = {})
#   %add_87 : Tensor "f32[128, 128, 1024][131072, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_283, %convert_element_type_499), kwargs = {})
#   %view_284 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_87, [128, 128, 32, 32]), kwargs = {})
#   %clone_60 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%view_284,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_467 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_35, torch.float32), kwargs = {})
#   %clone_55 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_467,), kwargs = {memory_format: torch.contiguous_format})
#   %view_291 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_21, [128, 32, 4]), kwargs = {})
#   %sum_23 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_291, [2]), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_31, -1), kwargs = {})
#   %full_default_2 : Tensor "f32[1, 32, 4][128, 4, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_102 : Tensor "f32[128, 32, 4][128, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_32, %full_default_2), kwargs = {})
#   %mul_104 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %squeeze_31), kwargs = {})
#   %mul_105 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %squeeze_31), kwargs = {})
#   %mul_106 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_105, %squeeze_31), kwargs = {})
#   %mul_107 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, 0.000244140625), kwargs = {})
#   %neg_1 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_107,), kwargs = {})
#   %mul_108 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %squeeze_30), kwargs = {})
#   %mul_109 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_23, %squeeze_31), kwargs = {})
#   %mul_110 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, 0.000244140625), kwargs = {})
#   %sub_36 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_108, %mul_110), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[128, 32, 4, 1][128, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_102, -1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_107, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sub_36, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %view_292 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_60, [128, 32, 4, 1024]), kwargs = {})
#   %mul_111 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_292, %unsqueeze_33), kwargs = {})
#   %view_263 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_55, [128, 32, 4, 1024]), kwargs = {})
#   %mul_112 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_263, %unsqueeze_35), kwargs = {})
#   %add_88 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_111, %mul_112), kwargs = {})
#   %add_89 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %unsqueeze_37), kwargs = {})
#   %view_294 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_89, [128, 128, 32, 32]), kwargs = {})
#   %convert_element_type_500 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_294, torch.bfloat16), kwargs = {})
#   return %convert_element_type_500
triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13 = async_compile.triton('triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 167772160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (y3 // 4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (y3 // 4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (y3 // 4), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13 * tmp7
    tmp15 = tmp14 * tmp7
    tmp16 = tmp15 * tmp7
    tmp17 = 0.000244140625
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp10 + tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 1024*y3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ve/cveghws5ludnl4j4pfpwr3gy4zonjuf3ntkzuzv67qca6vytfg74.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_97 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_97]
#   %getitem_128 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_128]
#   %mul_114 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_114]
#   %add_90 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %getitem_128), kwargs = {})
#   %mul_115 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_90, %mul_114), kwargs = {})
#   return %mul_115
triton_poi_fused_add_mul_14 = async_compile.triton('triton_poi_fused_add_mul_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_14(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_out_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 1024*y3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/a3/ca3n36emd3pphkzy7j7kbmrgysh4i3emvhquu3ntvtt44dxp52gv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_129 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=getitem_129]
#   %convert_element_type_501 : Tensor "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_129, torch.float32), kwargs = {})
#   return %convert_element_type_501
triton_poi_fused__to_copy_15 = async_compile.triton('triton_poi_fused__to_copy_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1474560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/xh/cxhk634qxv4uh6jtysypkrmdtrgnlkvznleokvpah4jttaqrvehc.py
# Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   input_41 => sigmoid_30
# Graph fragment:
#   %mul_169 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_169]
#   %getitem_152 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_152]
#   %add_118 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=add_118]
#   %convert_element_type_90 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_90]
#   %sum_53 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_53]
#   %addmm_24 : Tensor "bf16[128, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_24]
#   %add_118 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_169, %getitem_152), kwargs = {})
#   %mul_185 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_118, %convert_element_type_90), kwargs = {})
#   %sigmoid_30 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_24,), kwargs = {})
#   %sum_53 : Tensor "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_185, [2, 3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_571 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_53, torch.bfloat16), kwargs = {})
#   %squeeze_34 : Tensor "bf16[128, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_571, -1), kwargs = {})
#   %squeeze_35 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%squeeze_34, -1), kwargs = {})
#   %convert_element_type_572 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%squeeze_35, torch.float32), kwargs = {})
#   %convert_element_type_573 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_30, torch.float32), kwargs = {})
#   %sub_53 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_573), kwargs = {})
#   %mul_187 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_573, %sub_53), kwargs = {})
#   %mul_188 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_572, %mul_187), kwargs = {})
#   %convert_element_type_574 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_188, torch.bfloat16), kwargs = {})
#   return %add_118,%sum_53,%convert_element_type_574
triton_red_fused__to_copy_add_mul_sigmoid_sigmoid_backward_squeeze_sum_16 = async_compile.triton('triton_red_fused__to_copy_add_mul_sigmoid_sigmoid_backward_squeeze_sum_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sigmoid_sigmoid_backward_squeeze_sum_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67207168, 'r0_': 100663296}}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sigmoid_sigmoid_backward_squeeze_sum_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_2 + 1024*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tl.store(in_out_ptr0 + (r0_2 + 1024*x3), tmp2, r0_mask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp11 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tmp7.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 1.0
    tmp15 = tmp14 - tmp13
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dg/cdgtmmxo3vdas6z73pzeumzjbjrlx3nkhnqoupe7slvifly3awya.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_155 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_155]
#   %mul_193 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_193]
#   %mul_194 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_155, %mul_193), kwargs = {})
#   return %mul_194
triton_poi_fused_mul_17 = async_compile.triton('triton_poi_fused_mul_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67108864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/fr/cfrsnxcubxya6swtf6errpijk26goqgy5l446v6rjvw3ejyccrqy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.fill, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_18 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=mm_18]
#   %addmm_23 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_23]
#   %sigmoid_53 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_23,), kwargs = {})
#   %full_default_11 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([128, 256], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_54 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_11, %sigmoid_53), kwargs = {})
#   %mul_189 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_23, %sub_54), kwargs = {})
#   %add_119 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_189, 1), kwargs = {})
#   %mul_190 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_53, %add_119), kwargs = {})
#   %mul_191 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_18, %mul_190), kwargs = {})
#   return %mul_191
triton_poi_fused_add_fill_mul_sigmoid_sub_18 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_sub_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_sub_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/4x/c4xlhxxbwkjnm5tztr63nuwwcigtrxkpf3vpx4xihj3t23a4xjdi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_574 : Tensor "bf16[128, 128][128, 1]cuda:0" = PlaceHolder[target=convert_element_type_574]
#   %sum_54 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_574, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_54
triton_red_fused_sum_19 = async_compile.triton('triton_red_fused_sum_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33792, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_19(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/sc/cscutovgddqwd27dddkdzivilb3jpbkp2dwt35sh4vup3haji7e3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_191 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=mul_191]
#   %sum_55 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_191, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_55
triton_red_fused_sum_20 = async_compile.triton('triton_red_fused_sum_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67584, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_20(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7l/c7lwjorpcj443gzlaogkuptxhiygkesuwqmaptoc74w2oyh2wimf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_19 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=mm_19]
#   %convert_element_type_580 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_19, torch.float32), kwargs = {})
#   return %convert_element_type_580
triton_poi_fused__to_copy_21 = async_compile.triton('triton_poi_fused__to_copy_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 327680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/bp/cbpvrqmwtaiq6hooxt3o6f5fd7atviti54qok5mo24py76a3pj3o.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_21 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=mm_21]
#   %convert_element_type_587 : Tensor "f32[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_21, torch.float32), kwargs = {})
#   return %convert_element_type_587
triton_poi_fused__to_copy_22 = async_compile.triton('triton_poi_fused__to_copy_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 655360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ee/ceelkcm3dqo56mhnk7evqeeo7brxwzuxkh6je2oswu2sofhu5myl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_194 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_194]
#   %sum_57 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_194, [0, 2, 3]), kwargs = {})
#   return %buf156
triton_red_fused_convolution_backward_23 = async_compile.triton('triton_red_fused_convolution_backward_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17039360, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_convolution_backward_23(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/zr/czrlzcbrunkfrrbvk5w3mnsdajew477qedmpxqmyayom7kpk54nc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf156 : Tensor "f32[256, 128][1, 256]cuda:0" = PlaceHolder[target=buf156]
#   %sum_57 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=sum_57]
#   %sum_57 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_194, [0, 2, 3]), kwargs = {})
#   %convert_element_type_593 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_57, torch.float32), kwargs = {})
#   return %sum_57,%convert_element_type_593
triton_red_fused__to_copy_convolution_backward_24 = async_compile.triton('triton_red_fused__to_copy_convolution_backward_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_convolution_backward_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 133120, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_convolution_backward_24(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rz/crzoxh5iz5hfroc6tmosoi7uu3ogch77xmx4qmky3gd6ws7u525u.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_156 : Tensor "bf16[256, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=getitem_156]
#   %convert_element_type_589 : Tensor "f32[256, 128, 4, 4][2048, 1, 512, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_156, torch.float32), kwargs = {})
#   return %convert_element_type_589
triton_poi_fused__to_copy_25 = async_compile.triton('triton_poi_fused__to_copy_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5242880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qd/cqdyrtfbzhgnnvdxc5ptgsodoe3hvd3wnswa7tf3e2zwnsjheqvg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_158 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_158]
#   %mul_196 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=mul_196]
#   %convert_element_type_591 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_158, torch.float32), kwargs = {})
#   %mul_197 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_591, %mul_196), kwargs = {})
#   %sum_58 : Tensor "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_197, [2, 3], True), kwargs = {dtype: torch.float32})
#   %view_345 : Tensor "f32[128, 256, 256][65536, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_197, [128, 256, 256]), kwargs = {})
#   %convert_element_type_595 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_345, torch.bfloat16), kwargs = {})
#   return %sum_58,%convert_element_type_595
triton_red_fused__to_copy_mul_sum_view_26 = async_compile.triton('triton_red_fused__to_copy_mul_sum_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_view_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17039360, 'r0_': 67108864}}
)
@triton.jit
def triton_red_fused__to_copy_mul_sum_view_26(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 256*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp7 = tmp3.to(tl.float32)
        tl.store(out_ptr1 + (r0_2 + 256*x3), tmp7, r0_mask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2l/c2l7fkk5symuwlnscio7rezsvvpgssol2oehjuabpjcnvauc4h7j.py
# Topologically Sorted Source Nodes: [x_norm_11], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
# Source node to ATen node mapping:
#   x_norm_11 => clone_50, convert_element_type_388
# Graph fragment:
#   %getitem_158 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_158]
#   %mul_196 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=mul_196]
#   %bmm_74 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=bmm_74]
#   %convolution_24 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_24]
#   %convert_element_type_591 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_158, torch.float32), kwargs = {})
#   %mul_197 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_591, %mul_196), kwargs = {})
#   %view_345 : Tensor "f32[128, 256, 256][65536, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_197, [128, 256, 256]), kwargs = {})
#   %convert_element_type_604 : Tensor "f32[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_74, torch.float32), kwargs = {})
#   %add_122 : Tensor "f32[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_345, %convert_element_type_604), kwargs = {})
#   %view_346 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_122, [128, 256, 16, 16]), kwargs = {})
#   %clone_75 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%view_346,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_388 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %clone_50 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_388,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_198 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_75, %clone_50), kwargs = {})
#   %view_350 : Tensor "f32[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_198, [128, 256, 256]), kwargs = {})
#   %sum_59 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_350, [2]), kwargs = {})
#   %view_351 : Tensor "f32[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_75, [128, 256, 256]), kwargs = {})
#   %sum_60 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_351, [2]), kwargs = {})
#   return %sum_59,%sum_60
triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27 = async_compile.triton('triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34078720, 'r0_': 50331648}}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 4096*(((r0_2 % 16)) // 16) + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 16*(((r0_2 % 16)) // 16) + 256*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 256*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (x0 + 256*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
        tmp13 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, None)
    tl.store(out_ptr1 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/iv/civnghtme2pmivpzz4p237i6jhtuipespv2ayomti7d7kub5ccda.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_59 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=sum_59]
#   %view_352 : Tensor "f32[128, 32, 8][256, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_59, [128, 32, 8]), kwargs = {})
#   %sum_61 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_352, [2]), kwargs = {})
#   return %sum_61
triton_per_fused_native_group_norm_backward_28 = async_compile.triton('triton_per_fused_native_group_norm_backward_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_native_group_norm_backward_28(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ux/cuxec4om6hwotasn276weuk6ndfbwg3u53pz2mqgwl7caik26tqs.py
# Topologically Sorted Source Nodes: [x_norm_11], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_norm_11 => clone_50, convert_element_type_388, view_228
# Graph fragment:
#   %getitem_158 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_158]
#   %mul_196 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=mul_196]
#   %bmm_74 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=bmm_74]
#   %squeeze_21 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=squeeze_21]
#   %convolution_24 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_24]
#   %sum_62 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sum_62]
#   %squeeze_20 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=squeeze_20]
#   %sum_61 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sum_61]
#   %add_124 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0" = PlaceHolder[target=add_124]
#   %convert_element_type_591 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_158, torch.float32), kwargs = {})
#   %mul_197 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_591, %mul_196), kwargs = {})
#   %view_345 : Tensor "f32[128, 256, 256][65536, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_197, [128, 256, 256]), kwargs = {})
#   %convert_element_type_604 : Tensor "f32[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_74, torch.float32), kwargs = {})
#   %add_122 : Tensor "f32[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_345, %convert_element_type_604), kwargs = {})
#   %view_346 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_122, [128, 256, 16, 16]), kwargs = {})
#   %clone_75 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%view_346,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_388 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %clone_50 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_388,), kwargs = {memory_format: torch.contiguous_format})
#   %unsqueeze_62 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_21, -1), kwargs = {})
#   %full_default_13 : Tensor "f32[1, 32, 8][256, 8, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 8], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_199 : Tensor "f32[128, 32, 8][256, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_62, %full_default_13), kwargs = {})
#   %mul_200 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_62, %squeeze_20), kwargs = {})
#   %sub_57 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_200, %sum_61), kwargs = {})
#   %mul_201 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %squeeze_21), kwargs = {})
#   %mul_202 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_201, %squeeze_21), kwargs = {})
#   %mul_203 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %squeeze_21), kwargs = {})
#   %mul_204 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_203, 0.00048828125), kwargs = {})
#   %neg_6 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_204,), kwargs = {})
#   %mul_205 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_6, %squeeze_20), kwargs = {})
#   %mul_206 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_62, %squeeze_21), kwargs = {})
#   %mul_207 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, 0.00048828125), kwargs = {})
#   %sub_58 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_205, %mul_207), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[128, 32, 8, 1][256, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_199, -1), kwargs = {})
#   %unsqueeze_64 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_204, -1), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_64, -1), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sub_58, -1), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, -1), kwargs = {})
#   %view_354 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_75, [128, 32, 8, 256]), kwargs = {})
#   %mul_208 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_354, %unsqueeze_63), kwargs = {})
#   %view_228 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_50, [128, 32, 8, 256]), kwargs = {})
#   %mul_209 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_228, %unsqueeze_65), kwargs = {})
#   %add_123 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_208, %mul_209), kwargs = {})
#   %add_124 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_123, %unsqueeze_67), kwargs = {})
#   %view_356 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_124, [128, 256, 16, 16]), kwargs = {})
#   %convert_element_type_605 : Tensor "bf16[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_356, torch.bfloat16), kwargs = {})
#   return %add_124,%convert_element_type_605
triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29 = async_compile.triton('triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83886080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x3 = xindex // 65536
    x5 = ((xindex // 256) % 256)
    x6 = xindex // 256
    x4 = xindex
    x8 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x5 + 256*x0 + 4096*(((x0 % 16)) // 16) + 65536*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + 16*(((x0 % 16)) // 16) + 256*x6), None)
    tmp4 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x8), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x5 + 256*x0 + 65536*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x8), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x8), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x8), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp7
    tmp19 = tmp18 * tmp7
    tmp20 = tmp19 * tmp7
    tmp21 = 0.00048828125
    tmp22 = tmp20 * tmp21
    tmp23 = tmp12 * tmp22
    tmp24 = tmp10 + tmp23
    tmp25 = -tmp22
    tmp26 = tmp25 * tmp14
    tmp27 = tmp13 * tmp7
    tmp28 = tmp27 * tmp21
    tmp29 = tmp26 - tmp28
    tmp30 = tmp24 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp31, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ka/ckamtsoiv76jr4hievc4ilxx4qiobyedhpu7x5zvjm377e3zmmgz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_605 : Tensor "bf16[128, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=convert_element_type_605]
#   %sum_63 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=sum_63]
#   %sum_63 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_605, [0, 2, 3]), kwargs = {})
#   %convert_element_type_607 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_63, torch.float32), kwargs = {})
#   return %sum_63,%convert_element_type_607
triton_red_fused__to_copy_convolution_backward_30 = async_compile.triton('triton_red_fused__to_copy_convolution_backward_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_convolution_backward_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_convolution_backward_30(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 32768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 256)
        r0_2 = r0_index // 256
        tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0 + 65536*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/r5/cr5rrat73xm5sjurtcppcaebdnfkwqercslzvquh46zoso25yltd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_162 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=getitem_162]
#   %convert_element_type_606 : Tensor "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_162, torch.float32), kwargs = {})
#   return %convert_element_type_606
triton_poi_fused__to_copy_31 = async_compile.triton('triton_poi_fused__to_copy_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/do/cdon2gagtpl2tkonx2j25st5zi3whhmtcynxbydut47ssczypdum.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_161 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_161]
#   %getitem_89 : Tensor "b8[128, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=getitem_89]
#   %view_357 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_161, [128, 256, 256]), kwargs = {})
#   %permute_185 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_357, [0, 2, 1]), kwargs = {})
#   %convert_element_type_608 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_89, torch.bfloat16), kwargs = {})
#   %mul_210 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_608, 1.1111111111111112), kwargs = {})
#   %mul_211 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_185, %mul_210), kwargs = {})
#   return %mul_211
triton_poi_fused_native_dropout_backward_transpose_view_32 = async_compile.triton('triton_poi_fused_native_dropout_backward_transpose_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i1', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_backward_transpose_view_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 58720256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_dropout_backward_transpose_view_32(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ic/cicxmuaiamzpjkwinp2zufhlaehf77ilogjzfaiy5npwqf3dncsm.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_211 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=mul_211]
#   %view_357 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_161, [128, 256, 256]), kwargs = {})
#   %permute_185 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_357, [0, 2, 1]), kwargs = {})
#   %convert_element_type_608 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_89, torch.bfloat16), kwargs = {})
#   %mul_210 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_608, 1.1111111111111112), kwargs = {})
#   %mul_211 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_185, %mul_210), kwargs = {})
#   %view_358 : Tensor "bf16[32768, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_211, [32768, 256]), kwargs = {})
#   %sum_64 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_358, [0], True), kwargs = {dtype: torch.float32})
#   return %buf184
triton_red_fused_native_dropout_backward_sum_transpose_view_33 = async_compile.triton('triton_red_fused_native_dropout_backward_sum_transpose_view_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_dropout_backward_sum_transpose_view_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17039360, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_dropout_backward_sum_transpose_view_33(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/4p/c4pcwxcieap7ihag3mjfkeiz36z5g7cyjgfzl4xktwt5qaunxywm.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf184 : Tensor "f32[1, 256, 128][32768, 1, 256]cuda:0" = PlaceHolder[target=buf184]
#   %view_357 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_161, [128, 256, 256]), kwargs = {})
#   %permute_185 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_357, [0, 2, 1]), kwargs = {})
#   %convert_element_type_608 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_89, torch.bfloat16), kwargs = {})
#   %mul_210 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_608, 1.1111111111111112), kwargs = {})
#   %mul_211 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_185, %mul_210), kwargs = {})
#   %view_358 : Tensor "bf16[32768, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_211, [32768, 256]), kwargs = {})
#   %sum_64 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_358, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_64
triton_red_fused_native_dropout_backward_sum_transpose_view_34 = async_compile.triton('triton_red_fused_native_dropout_backward_sum_transpose_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_dropout_backward_sum_transpose_view_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 133120, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_dropout_backward_sum_transpose_view_34(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/6s/c6s3c37mipxxkvnczlwuhmv2yar37vmcb7mill4ipdjtwygthbsv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_22 : Tensor "bf16[32768, 256][256, 1]cuda:0" = PlaceHolder[target=mm_22]
#   %view_360 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_22, [128, 256, 256]), kwargs = {})
#   %view_361 : Tensor "bf16[128, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_360, [128, 256, 4, 64]), kwargs = {})
#   %permute_190 : Tensor "bf16[128, 4, 256, 64][65536, 64, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_361, [0, 2, 1, 3]), kwargs = {})
#   %clone_77 : Tensor "bf16[128, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_190,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_77
triton_poi_fused_clone_transpose_view_35 = async_compile.triton('triton_poi_fused_clone_transpose_view_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 256)
    x2 = ((xindex // 16384) % 4)
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 256*x1 + 65536*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ob/cobtdtxpb5l3kvwyb6p3tzyovlqklczodmlyb7fuk7om7fos2bfd.py
# Topologically Sorted Source Nodes: [matmul_18, attn_19], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   attn_19 => div_10, exp_10, sum_10
#   matmul_18 => view_220
# Graph fragment:
#   %bmm_38 : Tensor "bf16[512, 256, 1][256, 1, 1]cuda:0" = PlaceHolder[target=bmm_38]
#   %bmm_77 : Tensor "bf16[512, 256, 1][256, 1, 1]cuda:0" = PlaceHolder[target=bmm_77]
#   %view_364 : Tensor "bf16[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_77, [128, 4, 256, 1]), kwargs = {})
#   %convert_element_type_620 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_364, torch.float32), kwargs = {})
#   %view_220 : Tensor "bf16[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_38, [128, 4, 256, 1]), kwargs = {})
#   %convert_element_type_default_26 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_220, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_26, 1), kwargs = {})
#   %amax_default : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 0.125), kwargs = {})
#   %exp_10 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_1,), kwargs = {})
#   %sum_10 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_10, [-1], True), kwargs = {})
#   %div_10 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_10, %sum_10), kwargs = {})
#   %mul_212 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_620, %div_10), kwargs = {})
#   %sum_65 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_212, [-1], True), kwargs = {})
#   %neg_7 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_10,), kwargs = {})
#   %fma : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.fma.default](args = (%neg_7, %sum_65, %mul_212), kwargs = {})
#   %convert_element_type_621 : Tensor "bf16[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%fma, torch.bfloat16), kwargs = {})
#   %mul_213 : Tensor "bf16[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_621, 0.125), kwargs = {})
#   return %mul_213
triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36 = async_compile.triton('triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 - tmp3
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = (tmp7 / tmp7)
    tmp9 = -tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp8
    tmp13 = libdevice.fma(tmp9, tmp12, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/73/c734slwafahqtarj23p3ujly7olray2ixmrzjbnjv3q7huukvzem.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_78 : Tensor "bf16[512, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_78]
#   %bmm_76 : Tensor "bf16[512, 1, 64][64, 64, 1]cuda:0" = PlaceHolder[target=bmm_76]
#   %view_363 : Tensor "bf16[128, 4, 1, 64][256, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_76, [128, 4, 1, 64]), kwargs = {})
#   %view_366 : Tensor "bf16[128, 4, 64, 1][256, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_78, [128, 4, 64, 1]), kwargs = {})
#   %permute_195 : Tensor "bf16[128, 4, 1, 64][256, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_366, [0, 1, 3, 2]), kwargs = {})
#   %permute_196 : Tensor "bf16[128, 1, 4, 64][256, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_363, [0, 2, 1, 3]), kwargs = {})
#   %view_368 : Tensor "bf16[128, 1, 256][256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_196, [128, 1, 256]), kwargs = {})
#   %permute_197 : Tensor "bf16[128, 1, 4, 64][256, 1, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_195, [0, 2, 1, 3]), kwargs = {})
#   %view_369 : Tensor "bf16[128, 1, 256][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_197, [128, 1, 256]), kwargs = {})
#   %cat_8 : Tensor "bf16[128, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_369, %view_368], 2), kwargs = {})
#   return %cat_8
triton_poi_fused_cat_transpose_view_37 = async_compile.triton('triton_poi_fused_cat_transpose_view_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_transpose_view_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_transpose_view_37(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (256*x1 + ((-256) + x0)), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/kt/cktkiybxr6pjmyojjiieicz26gwxlzayetaeku7ljcgxhul77k5n.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_24 : Tensor "bf16[512, 256][256, 1]cuda:0" = PlaceHolder[target=mm_24]
#   %convert_element_type_630 : Tensor "f32[512, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_24, torch.float32), kwargs = {})
#   return %convert_element_type_630
triton_poi_fused__to_copy_38 = async_compile.triton('triton_poi_fused__to_copy_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1310720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/f6/cf6yknav232tvyah43gawxuqdpasj3noitkmqduo6gvzloxahxwi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_79 : Tensor "bf16[512, 256, 64][16384, 64, 1]cuda:0" = PlaceHolder[target=bmm_79]
#   %view_367 : Tensor "bf16[128, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_79, [128, 4, 256, 64]), kwargs = {})
#   %permute_198 : Tensor "bf16[128, 256, 4, 64][65536, 64, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_367, [0, 2, 1, 3]), kwargs = {})
#   %clone_78 : Tensor "bf16[128, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_198,), kwargs = {memory_format: torch.contiguous_format})
#   %view_370 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_78, [128, 256, 256]), kwargs = {})
#   %view_371 : Tensor "bf16[32768, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_370, [32768, 256]), kwargs = {})
#   return %view_371
triton_poi_fused__unsafe_view_clone_transpose_view_39 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (64*((x1 % 256)) + 16384*(x0 // 64) + 65536*(x1 // 256) + ((x0 % 64))), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/25/c25ozctt55rlzeyqwavtyz5dqde3nvlr45irxfpm2wfbly46v2su.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_194 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_194]
#   %getitem_161 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_161]
#   %mm_27 : Tensor "bf16[32768, 256][256, 1]cuda:0" = PlaceHolder[target=mm_27]
#   %mul_215 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_215]
#   %add_125 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %getitem_161), kwargs = {})
#   %view_372 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_27, [128, 256, 256]), kwargs = {})
#   %permute_207 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_372, [0, 2, 1]), kwargs = {})
#   %view_373 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_207, [128, 256, 16, 16]), kwargs = {})
#   %add_127 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %view_373), kwargs = {})
#   %mul_216 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %mul_215), kwargs = {})
#   return %mul_216
triton_poi_fused_add_mul_transpose_view_40 = async_compile.triton('triton_poi_fused_add_mul_transpose_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_transpose_view_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_transpose_view_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hp/chpsblulcokoyrb53ewe4676q4px56phcmifca4ijtuezs6r3lu7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_282 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_282]
#   %getitem_185 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_185]
#   %mm_51 : Tensor "bf16[32768, 256][256, 1]cuda:0" = PlaceHolder[target=mm_51]
#   %add_161 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_282, %getitem_185), kwargs = {})
#   %view_488 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_51, [128, 256, 256]), kwargs = {})
#   %permute_331 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_488, [0, 2, 1]), kwargs = {})
#   %view_489 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_331, [128, 256, 16, 16]), kwargs = {})
#   %add_163 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, %view_489), kwargs = {})
#   return %add_163
triton_poi_fused_add_transpose_view_41 = async_compile.triton('triton_poi_fused_add_transpose_view_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_transpose_view_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83886080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_transpose_view_41(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/v7/cv7fxzmo4xtpdf3yly7aqghiwdtbo4h7sqjq5l6cgeaykgrz2hfv.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.mul, aten.sigmoid, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_27 => sigmoid_18
# Graph fragment:
#   %add_163 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_163]
#   %convert_element_type_152 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convert_element_type_152]
#   %sum_102 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_102]
#   %addmm_17 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %mul_302 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %convert_element_type_152), kwargs = {})
#   %sigmoid_18 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_17,), kwargs = {})
#   %sum_102 : Tensor "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_302, [2, 3], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_816 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_102, torch.bfloat16), kwargs = {})
#   %squeeze_41 : Tensor "bf16[128, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%convert_element_type_816, -1), kwargs = {})
#   %squeeze_42 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%squeeze_41, -1), kwargs = {})
#   %convert_element_type_817 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%squeeze_42, torch.float32), kwargs = {})
#   %convert_element_type_818 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_18, torch.float32), kwargs = {})
#   %sub_75 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_818), kwargs = {})
#   %mul_304 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_818, %sub_75), kwargs = {})
#   %mul_305 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_817, %mul_304), kwargs = {})
#   %convert_element_type_819 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_305, torch.bfloat16), kwargs = {})
#   %sum_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_163, [0, 2, 3]), kwargs = {})
#   return %sum_102,%buf399,%convert_element_type_819
triton_red_fused__to_copy_convolution_backward_mul_sigmoid_sigmoid_backward_squeeze_sum_42 = async_compile.triton('triton_red_fused__to_copy_convolution_backward_mul_sigmoid_sigmoid_backward_squeeze_sum_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_convolution_backward_mul_sigmoid_sigmoid_backward_squeeze_sum_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34013184, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_convolution_backward_mul_sigmoid_sigmoid_backward_squeeze_sum_42(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp7 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp8, None)
    tmp12 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tmp5.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr2 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wa/cwah6m7zwmpn5t6dimr7zoao3rcqisph4bu57pcomdr4hwgkeja5.py
# Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._to_copy, aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   out_24 => convert_element_type_233, mul_32, sub_12
# Graph fragment:
#   %getitem_188 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_188]
#   %add_30 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=add_30]
#   %getitem_49 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_49]
#   %rsqrt_7 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_7]
#   %convert_element_type_834 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_188, torch.float32), kwargs = {})
#   %view_492 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_834, [128, 512, 64]), kwargs = {})
#   %permute_341 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_492, [0, 2, 1]), kwargs = {})
#   %convert_element_type_233 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.float32), kwargs = {})
#   %sub_12 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_49), kwargs = {})
#   %mul_32 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_315 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_341, %mul_32), kwargs = {})
#   %sum_108 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_315, [0, 1]), kwargs = {})
#   %sum_109 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_341, [0, 1]), kwargs = {})
#   return %buf408,%buf410
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_43 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17301504, 'r0_': 65536}}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/eo/ceo556gd326i7yk3owmjgzoqmhggp73uk37vyws4jp4lcbfnogta.py
# Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._to_copy, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   out_24 => convert_element_type_233, mul_32, sub_12
# Graph fragment:
#   %getitem_188 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_188]
#   %primals_76 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_76]
#   %add_30 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=add_30]
#   %getitem_49 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_49]
#   %rsqrt_7 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_7]
#   %sum_106 : Tensor "f32[128, 64, 1][64, 1, 8192]cuda:0" = PlaceHolder[target=sum_106]
#   %sum_107 : Tensor "f32[128, 64, 1][64, 1, 8192]cuda:0" = PlaceHolder[target=sum_107]
#   %convert_element_type_834 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_188, torch.float32), kwargs = {})
#   %view_492 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_834, [128, 512, 64]), kwargs = {})
#   %permute_341 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_492, [0, 2, 1]), kwargs = {})
#   %mul_310 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_341, %primals_76), kwargs = {})
#   %mul_311 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, 512), kwargs = {})
#   %sum_106 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_310, [2], True), kwargs = {})
#   %convert_element_type_233 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.float32), kwargs = {})
#   %sub_12 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_49), kwargs = {})
#   %mul_32 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_312 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %mul_32), kwargs = {})
#   %sum_107 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_312, [2], True), kwargs = {})
#   %mul_313 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %sum_107), kwargs = {})
#   %sub_78 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_311, %sum_106), kwargs = {})
#   %sub_79 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_78, %mul_313), kwargs = {})
#   %div_11 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_7, 512), kwargs = {})
#   %mul_314 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_11, %sub_79), kwargs = {})
#   %convert_element_type_837 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_314, torch.bfloat16), kwargs = {})
#   return %sum_106,%sum_107,%convert_element_type_837
triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_44 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 65536, 'r0_': 33556480}}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = 0.001953125
    tmp18 = tmp11 * tmp17
    tmp19 = 512.0
    tmp20 = tmp3 * tmp19
    tmp21 = tmp20 - tmp6
    tmp22 = tmp12 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp25, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/4j/c4jstow572svj47rpnzecld57afdze54y5zab7irwzum4pjftguz.py
# Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._to_copy, aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   out_24 => convert_element_type_233, mul_32, sub_12
# Graph fragment:
#   %buf408 : Tensor "f32[512, 64][1, 512]cuda:0" = PlaceHolder[target=buf408]
#   %convert_element_type_834 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_188, torch.float32), kwargs = {})
#   %view_492 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_834, [128, 512, 64]), kwargs = {})
#   %permute_341 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_492, [0, 2, 1]), kwargs = {})
#   %convert_element_type_233 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.float32), kwargs = {})
#   %sub_12 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_49), kwargs = {})
#   %mul_32 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_315 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_341, %mul_32), kwargs = {})
#   %sum_108 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_315, [0, 1]), kwargs = {})
#   return %sum_108
triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 135168, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/oo/coobelt7h4ebkmdh7fi3qtrhmldaw4oyz7jvgxm5dllizovkmrrl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_837 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_837]
#   %view_493 : Tensor "bf16[8192, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_837, [8192, 512]), kwargs = {})
#   %sum_110 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_493, [0], True), kwargs = {dtype: torch.float32})
#   return %buf415
triton_red_fused_sum_view_46 = async_compile.triton('triton_red_fused_sum_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8650752, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_46(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/mg/cmgqx4b2geu3j6x27esrieq5567dccgtkkjexcf7tb72zigokhiy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_57 : Tensor "bf16[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm_57]
#   %convert_element_type_843 : Tensor "f32[512, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_57, torch.float32), kwargs = {})
#   return %convert_element_type_843
triton_poi_fused__to_copy_47 = async_compile.triton('triton_poi_fused__to_copy_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/jf/cjfzhrdcof5t5bnr2zbnxyprkwtsmyqrshb3yxojfyniie57u33y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_189 : Tensor "bf16[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0" = PlaceHolder[target=getitem_189]
#   %convert_element_type_835 : Tensor "f32[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_189, torch.float32), kwargs = {})
#   return %convert_element_type_835
triton_poi_fused__to_copy_48 = async_compile.triton('triton_poi_fused__to_copy_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/gn/cgn52ao5avvf5g4q5dx4fjv2kohjsvm3juu44sm6kupzx7bmwdhj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_56 : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_56]
#   %view_495 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_56, [128, 64, 512]), kwargs = {})
#   %view_496 : Tensor "bf16[128, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_495, [128, 64, 8, 64]), kwargs = {})
#   %permute_346 : Tensor "bf16[128, 8, 64, 64][32768, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_496, [0, 2, 1, 3]), kwargs = {})
#   %clone_103 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_346,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_103
triton_poi_fused_clone_transpose_view_49 = async_compile.triton('triton_poi_fused_clone_transpose_view_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2b/c2bcj3ugeqe5pig3mh6o5rlkhjqwedckdxaxvryypsnltpenmae6.py
# Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.sub, aten._softmax, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   attn_9 => div_5, exp_5
#   matmul_8 => view_112
# Graph fragment:
#   %bmm_113 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_113]
#   %bmm_20 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %amax_default_5 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0" = PlaceHolder[target=amax_default_5]
#   %sum_5 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0" = PlaceHolder[target=sum_5]
#   %sum_111 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 65536]cuda:0" = PlaceHolder[target=sum_111]
#   %view_499 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_113, [128, 8, 64, 64]), kwargs = {})
#   %convert_element_type_849 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_499, torch.float32), kwargs = {})
#   %view_112 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_20, [128, 8, 64, 64]), kwargs = {})
#   %convert_element_type_default_31 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_112, torch.float32), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_31, 1), kwargs = {})
#   %sub_tensor_5 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_10, %amax_default_5), kwargs = {})
#   %mul_tensor_11 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_5, 0.125), kwargs = {})
#   %exp_5 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_11,), kwargs = {})
#   %div_5 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_5), kwargs = {})
#   %mul_316 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_849, %div_5), kwargs = {})
#   %sum_111 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_316, [-1], True), kwargs = {})
#   %neg_16 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_5,), kwargs = {})
#   %fma_5 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.fma.default](args = (%neg_16, %sum_111, %mul_316), kwargs = {})
#   %convert_element_type_850 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%fma_5, torch.bfloat16), kwargs = {})
#   %mul_317 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_850, 0.125), kwargs = {})
#   return %sum_111,%mul_317
triton_per_fused__softmax__softmax_backward_data__to_copy_mul_sub_view_50 = async_compile.triton('triton_per_fused__softmax__softmax_backward_data__to_copy_mul_sub_view_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data__to_copy_mul_sub_view_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 33554432}}
)
@triton.jit
def triton_per_fused__softmax__softmax_backward_data__to_copy_mul_sub_view_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), None).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp8 = 0.125
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.exp(tmp9)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tmp1 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = -tmp12
    tmp18 = libdevice.fma(tmp17, tmp16, tmp13)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp8
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp20, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/xg/cxgftr3tlkolpvhposgjayhureq5rejtticxbqqxsiyjrionhzoi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_112 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_112]
#   %bmm_114 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_114]
#   %bmm_115 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_115]
#   %view_498 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_112, [128, 8, 64, 64]), kwargs = {})
#   %view_501 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_114, [128, 8, 64, 64]), kwargs = {})
#   %view_502 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_115, [128, 8, 64, 64]), kwargs = {})
#   %permute_351 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_501, [0, 1, 3, 2]), kwargs = {})
#   %full_default_23 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 8, 64, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %view_498, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %permute_351, 0, 1), kwargs = {})
#   %add_166 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %view_502, 0, 0), kwargs = {})
#   %add_167 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_166, %select_scatter_default_2), kwargs = {})
#   %permute_352 : Tensor "bf16[128, 64, 3, 8, 64][32768, 64, 4194304, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_167, [1, 3, 0, 2, 4]), kwargs = {})
#   %clone_104 : Tensor "bf16[128, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_352,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_104
triton_poi_fused_add_clone_permute_select_backward_transpose_view_51 = async_compile.triton('triton_poi_fused_add_clone_permute_select_backward_transpose_view_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_permute_select_backward_transpose_view_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 8388608, 'x': 67108864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_permute_select_backward_transpose_view_51(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 1536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex // 512
    x2 = (xindex % 64)
    x3 = ((xindex // 64) % 8)
    y0 = (yindex % 64)
    y1 = yindex // 64
    x5 = (xindex % 512)
    x7 = xindex
    y6 = yindex
    tmp3 = tl.load(in_ptr0 + (x2 + 64*y0 + 4096*x3 + 32768*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (y0 + 64*x5 + 32768*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x2 + 64*y0 + 4096*x3 + 32768*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x4
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1, 1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tl.store(out_ptr0 + (x7 + 1536*y6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/id/cidtcogbcjdfy4gjyidh737rhknalocysc4zdd7w6m4atfm2py4f.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_104 : Tensor "bf16[128, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0" = PlaceHolder[target=clone_104]
#   %view_498 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_112, [128, 8, 64, 64]), kwargs = {})
#   %view_501 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_114, [128, 8, 64, 64]), kwargs = {})
#   %view_502 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_115, [128, 8, 64, 64]), kwargs = {})
#   %permute_351 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_501, [0, 1, 3, 2]), kwargs = {})
#   %full_default_23 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 8, 64, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %view_498, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %permute_351, 0, 1), kwargs = {})
#   %add_166 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %view_502, 0, 0), kwargs = {})
#   %add_167 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_166, %select_scatter_default_2), kwargs = {})
#   %permute_352 : Tensor "bf16[128, 64, 3, 8, 64][32768, 64, 4194304, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_167, [1, 3, 0, 2, 4]), kwargs = {})
#   %clone_104 : Tensor "bf16[128, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_352,), kwargs = {memory_format: torch.contiguous_format})
#   %view_503 : Tensor "bf16[128, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_104, [128, 64, 1536]), kwargs = {})
#   %view_504 : Tensor "bf16[8192, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_503, [8192, 1536]), kwargs = {})
#   %sum_112 : Tensor "f32[1, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_504, [0], True), kwargs = {dtype: torch.float32})
#   return %buf428
triton_red_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_52 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25509888, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_52(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 36864
    r0_numel = 342
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 1536
    x0 = (xindex % 1536)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 342*x1
        tmp1 = tl.full([1, 1], 8192, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 1536*r0_2 + 525312*x1), r0_mask & tmp2, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/6q/c6qzxhduktz6hwmq2f5xhvsytaxchrdeqreuwsgloyk672voiyd2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf428 : Tensor "f32[1, 1536, 24][36864, 1, 1536]cuda:0" = PlaceHolder[target=buf428]
#   %view_498 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_112, [128, 8, 64, 64]), kwargs = {})
#   %view_501 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_114, [128, 8, 64, 64]), kwargs = {})
#   %view_502 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_115, [128, 8, 64, 64]), kwargs = {})
#   %permute_351 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_501, [0, 1, 3, 2]), kwargs = {})
#   %full_default_23 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 8, 64, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %view_498, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %permute_351, 0, 1), kwargs = {})
#   %add_166 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_23, %view_502, 0, 0), kwargs = {})
#   %add_167 : Tensor "bf16[3, 128, 8, 64, 64][4194304, 32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_166, %select_scatter_default_2), kwargs = {})
#   %permute_352 : Tensor "bf16[128, 64, 3, 8, 64][32768, 64, 4194304, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_167, [1, 3, 0, 2, 4]), kwargs = {})
#   %clone_104 : Tensor "bf16[128, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_352,), kwargs = {memory_format: torch.contiguous_format})
#   %view_503 : Tensor "bf16[128, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_104, [128, 64, 1536]), kwargs = {})
#   %view_504 : Tensor "bf16[8192, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_503, [8192, 1536]), kwargs = {})
#   %sum_112 : Tensor "f32[1, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_504, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_112
triton_per_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_53 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 159744, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_53(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
    r0_numel = 24
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1536*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/yv/cyvfag2nrp5lbrk2ksm5dpp522xyvjl5gl36sdgvqfeb4d54sjn7.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.view, aten._to_copy, aten.native_layer_norm_backward, aten.silu, aten.transpose, aten.native_layer_norm, aten.add, aten.sigmoid, aten.fill, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   h_8 => convert_element_type_213, convert_element_type_214, mul_28, sigmoid_16
#   view_50 => view_106
#   x_flat_6 => permute_56
#   x_norm_6 => convert_element_type_215, mul_29, sub_10
# Graph fragment:
#   %mm_58 : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_58]
#   %primals_70 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_70]
#   %add_27 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %getitem_47 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_47]
#   %rsqrt_6 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %convert_element_type_837 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_837]
#   %sum_113 : Tensor "f32[128, 64, 1][64, 1, 8192]cuda:0" = PlaceHolder[target=sum_113]
#   %sum_114 : Tensor "f32[128, 64, 1][64, 1, 8192]cuda:0" = PlaceHolder[target=sum_114]
#   %view_506 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_58, [128, 64, 512]), kwargs = {})
#   %convert_element_type_860 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_506, torch.float32), kwargs = {})
#   %mul_319 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_860, %primals_70), kwargs = {})
#   %mul_320 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_319, 512), kwargs = {})
#   %sum_113 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_319, [2], True), kwargs = {})
#   %convert_element_type_213 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_28 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [128, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_215 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_56, torch.float32), kwargs = {})
#   %sub_10 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_215, %getitem_47), kwargs = {})
#   %mul_29 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_6), kwargs = {})
#   %mul_321 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_319, %mul_29), kwargs = {})
#   %sum_114 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_321, [2], True), kwargs = {})
#   %mul_322 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %sum_114), kwargs = {})
#   %sub_81 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_320, %sum_113), kwargs = {})
#   %sub_82 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_81, %mul_322), kwargs = {})
#   %div_12 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_6, 512), kwargs = {})
#   %mul_323 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %sub_82), kwargs = {})
#   %convert_element_type_863 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_323, torch.bfloat16), kwargs = {})
#   %add_168 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_837, %convert_element_type_863), kwargs = {})
#   %permute_357 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_168, [0, 2, 1]), kwargs = {})
#   %view_507 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_357, [128, 512, 8, 8]), kwargs = {})
#   %sigmoid_65 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %full_36 : Tensor "bf16[128, 8, 8, 512][32768, 4096, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 8, 8, 512], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_359 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_36, [0, 3, 1, 2]), kwargs = {})
#   %sub_83 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_359, %sigmoid_65), kwargs = {})
#   %mul_325 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sub_83), kwargs = {})
#   %add_169 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_325, 1), kwargs = {})
#   %mul_326 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_65, %add_169), kwargs = {})
#   %mul_327 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_507, %mul_326), kwargs = {})
#   return %sum_113,%sum_114,%mul_327
triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_sigmoid_silu_sub_transpose_view_54 = async_compile.triton('triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_sigmoid_silu_sub_transpose_view_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_sigmoid_silu_sub_transpose_view_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 65536, 'r0_': 41945088}}
)
@triton.jit
def triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_sigmoid_silu_sub_transpose_view_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 - tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp22 = 0.001953125
    tmp23 = tmp15 * tmp22
    tmp24 = 512.0
    tmp25 = tmp3 * tmp24
    tmp26 = tmp25 - tmp6
    tmp27 = tmp16 * tmp20
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp21 + tmp30
    tmp32 = tl.sigmoid(tmp7)
    tmp33 = 1.0
    tmp34 = tmp33 - tmp32
    tmp35 = tmp7 * tmp34
    tmp36 = tmp35 + tmp33
    tmp37 = tmp32 * tmp36
    tmp38 = tmp31 * tmp37
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp38, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/tf/ctfn7xyy45nyv3drrugaivgyuuzqte44ihsg6sds7dmpkjfgjwuj.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.view, aten._to_copy, aten.silu, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   h_8 => convert_element_type_213, convert_element_type_214, mul_28, sigmoid_16
#   view_50 => view_106
#   x_flat_6 => permute_56
#   x_norm_6 => convert_element_type_215, mul_29, sub_10
# Graph fragment:
#   %mm_58 : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_58]
#   %add_27 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %getitem_47 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_47]
#   %rsqrt_6 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %view_506 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_58, [128, 64, 512]), kwargs = {})
#   %convert_element_type_860 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_506, torch.float32), kwargs = {})
#   %convert_element_type_213 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_28 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [128, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_215 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_56, torch.float32), kwargs = {})
#   %sub_10 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_215, %getitem_47), kwargs = {})
#   %mul_29 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_6), kwargs = {})
#   %mul_324 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_860, %mul_29), kwargs = {})
#   %sum_115 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_324, [0, 1]), kwargs = {})
#   %sum_116 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_860, [0, 1]), kwargs = {})
#   return %buf433,%buf435
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_silu_transpose_view_55 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_silu_transpose_view_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_silu_transpose_view_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17301504, 'r0_': 65536}}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_silu_transpose_view_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tmp1 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/g3/cg3xge7p2apr7odkvqtkzlj3hdiraoeiw4h22he2k4rq36aw5ka3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_327 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=mul_327]
#   %sum_117 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_327, [0, 2, 3]), kwargs = {})
#   return %buf438
triton_red_fused_convolution_backward_56 = async_compile.triton('triton_red_fused_convolution_backward_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8650752, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_convolution_backward_56(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lh/clhywd3fm6g275ztxnmu737tzivb3snsxokccmor6m5asrxfr4uk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf438 : Tensor "f32[512, 64][1, 512]cuda:0" = PlaceHolder[target=buf438]
#   %sum_117 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=sum_117]
#   %sum_117 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_327, [0, 2, 3]), kwargs = {})
#   %convert_element_type_866 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_117, torch.float32), kwargs = {})
#   return %sum_117,%convert_element_type_866
triton_per_fused__to_copy_convolution_backward_57 = async_compile.triton('triton_per_fused__to_copy_convolution_backward_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_convolution_backward_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 135168, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_convolution_backward_57(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/zc/czc2o4r6jmfylkuuydoflllzcjpjtpmattt5irpzmzmgafwxjrly.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_59 : Tensor "bf16[1536, 512][512, 1]cuda:0" = PlaceHolder[target=mm_59]
#   %convert_element_type_861 : Tensor "f32[1536, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_59, torch.float32), kwargs = {})
#   return %convert_element_type_861
triton_poi_fused__to_copy_58 = async_compile.triton('triton_poi_fused__to_copy_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7864320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_58(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/yb/cyb7ezsppf6isqfbtqbyjy5tv2fagupbef6ow45vhjo33uafbzhu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_191 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_191]
#   %mul_329 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=mul_329]
#   %convert_element_type_864 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_191, torch.float32), kwargs = {})
#   %mul_330 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_864, %mul_329), kwargs = {})
#   %sum_118 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_330, [2, 3], True), kwargs = {dtype: torch.float32})
#   %view_508 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_330, [128, 512, 64]), kwargs = {})
#   %convert_element_type_868 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_508, torch.bfloat16), kwargs = {})
#   return %sum_118,%convert_element_type_868
triton_per_fused__to_copy_mul_sum_view_59 = async_compile.triton('triton_per_fused__to_copy_mul_sum_view_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mul_sum_view_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8912896, 'r0_': 33554432}}
)
@triton.jit
def triton_per_fused__to_copy_mul_sum_view_59(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 32768*x1), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 64*x3), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = tmp3.to(tl.float32)
    tl.store(out_ptr1 + (r0_2 + 64*x3), tmp7, None)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pt/cptxqhlwizh3qzwmqvoewjpovhshk5ozi6sdrvkq5rc2eo52puij.py
# Topologically Sorted Source Nodes: [x_norm_5], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
# Source node to ATen node mapping:
#   x_norm_5 => clone_21, convert_element_type_204
# Graph fragment:
#   %getitem_191 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_191]
#   %mul_329 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=mul_329]
#   %bmm_118 : Tensor "bf16[128, 512, 64][32768, 64, 1]cuda:0" = PlaceHolder[target=bmm_118]
#   %convolution_13 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_13]
#   %convert_element_type_864 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_191, torch.float32), kwargs = {})
#   %mul_330 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_864, %mul_329), kwargs = {})
#   %view_508 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_330, [128, 512, 64]), kwargs = {})
#   %convert_element_type_877 : Tensor "f32[128, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_118, torch.float32), kwargs = {})
#   %add_171 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_508, %convert_element_type_877), kwargs = {})
#   %view_509 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_171, [128, 512, 8, 8]), kwargs = {})
#   %clone_107 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%view_509,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_204 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %clone_21 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_204,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_331 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_107, %clone_21), kwargs = {})
#   %view_513 : Tensor "f32[128, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_331, [128, 512, 64]), kwargs = {})
#   %sum_119 : Tensor "f32[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_513, [2]), kwargs = {})
#   %view_514 : Tensor "f32[128, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_107, [128, 512, 64]), kwargs = {})
#   %sum_120 : Tensor "f32[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_514, [2]), kwargs = {})
#   return %sum_119,%sum_120
triton_per_fused__to_copy_add_clone_mul_native_group_norm_backward_view_60 = async_compile.triton('triton_per_fused__to_copy_add_clone_mul_native_group_norm_backward_view_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mul_native_group_norm_backward_view_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17825792, 'r0_': 25165824}}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mul_native_group_norm_backward_view_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 4096*(((r0_2 % 8)) // 8) + 32768*x1), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 8*(((r0_2 % 8)) // 8) + 64*x3), None)
    tmp4 = tl.load(in_ptr2 + (r0_2 + 64*x3), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0 + 512*r0_2 + 32768*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/a3/ca3cd4jixze32zbyihaoa5rcbzza3sddajms72pg6j2dnv6nfdbh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_119 : Tensor "f32[128, 512][512, 1]cuda:0" = PlaceHolder[target=sum_119]
#   %view_515 : Tensor "f32[128, 32, 16][512, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_119, [128, 32, 16]), kwargs = {})
#   %sum_121 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_515, [2]), kwargs = {})
#   return %sum_121
triton_per_fused_native_group_norm_backward_61 = async_compile.triton('triton_per_fused_native_group_norm_backward_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768, 'r0_': 262144}}
)
@triton.jit
def triton_per_fused_native_group_norm_backward_61(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dv/cdvrkzzhy3wcuhpevc4u5bhwnxcan7o3ethtn3blleqz3qovtpkk.py
# Topologically Sorted Source Nodes: [x_norm_5], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_norm_5 => clone_21, convert_element_type_204, view_99
# Graph fragment:
#   %getitem_191 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_191]
#   %mul_329 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=mul_329]
#   %bmm_118 : Tensor "bf16[128, 512, 64][32768, 64, 1]cuda:0" = PlaceHolder[target=bmm_118]
#   %squeeze_11 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=squeeze_11]
#   %convolution_13 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_13]
#   %sum_122 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sum_122]
#   %squeeze_10 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=squeeze_10]
#   %sum_121 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=sum_121]
#   %add_173 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0" = PlaceHolder[target=add_173]
#   %convert_element_type_864 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_191, torch.float32), kwargs = {})
#   %mul_330 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_864, %mul_329), kwargs = {})
#   %view_508 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_330, [128, 512, 64]), kwargs = {})
#   %convert_element_type_877 : Tensor "f32[128, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm_118, torch.float32), kwargs = {})
#   %add_171 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_508, %convert_element_type_877), kwargs = {})
#   %view_509 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_171, [128, 512, 8, 8]), kwargs = {})
#   %clone_107 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%view_509,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_204 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %clone_21 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_204,), kwargs = {memory_format: torch.contiguous_format})
#   %unsqueeze_92 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_11, -1), kwargs = {})
#   %full_default_27 : Tensor "f32[1, 32, 16][512, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 16], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_332 : Tensor "f32[128, 32, 16][512, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_92, %full_default_27), kwargs = {})
#   %mul_333 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_122, %squeeze_10), kwargs = {})
#   %sub_85 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_333, %sum_121), kwargs = {})
#   %mul_334 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %squeeze_11), kwargs = {})
#   %mul_335 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %squeeze_11), kwargs = {})
#   %mul_336 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_335, %squeeze_11), kwargs = {})
#   %mul_337 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_336, 0.0009765625), kwargs = {})
#   %neg_17 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_337,), kwargs = {})
#   %mul_338 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_17, %squeeze_10), kwargs = {})
#   %mul_339 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_122, %squeeze_11), kwargs = {})
#   %mul_340 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_339, 0.0009765625), kwargs = {})
#   %sub_86 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_338, %mul_340), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[128, 32, 16, 1][512, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_332, -1), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_337, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %unsqueeze_96 : Tensor "f32[128, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sub_86, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %view_517 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_107, [128, 32, 16, 64]), kwargs = {})
#   %mul_341 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_517, %unsqueeze_93), kwargs = {})
#   %view_99 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [128, 32, 16, 64]), kwargs = {})
#   %mul_342 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_99, %unsqueeze_95), kwargs = {})
#   %add_172 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_341, %mul_342), kwargs = {})
#   %add_173 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_172, %unsqueeze_97), kwargs = {})
#   %view_519 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_173, [128, 512, 8, 8]), kwargs = {})
#   %convert_element_type_878 : Tensor "bf16[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_519, torch.bfloat16), kwargs = {})
#   return %add_173,%convert_element_type_878
triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_62 = async_compile.triton('triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x3 = xindex // 32768
    x5 = ((xindex // 64) % 512)
    x6 = xindex // 64
    x4 = xindex
    x8 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x5 + 512*x0 + 4096*(((x0 % 8)) // 8) + 32768*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + 8*(((x0 % 8)) // 8) + 64*x6), None)
    tmp4 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x8), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x5 + 512*x0 + 32768*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x8), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x8), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x8), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp7
    tmp19 = tmp18 * tmp7
    tmp20 = tmp19 * tmp7
    tmp21 = 0.0009765625
    tmp22 = tmp20 * tmp21
    tmp23 = tmp12 * tmp22
    tmp24 = tmp10 + tmp23
    tmp25 = -tmp22
    tmp26 = tmp25 * tmp14
    tmp27 = tmp13 * tmp7
    tmp28 = tmp27 * tmp21
    tmp29 = tmp26 - tmp28
    tmp30 = tmp24 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp31, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ny/cnyl7hb6mwuuagjmn4upmaqjkmg7oj43ncxdy5sl6xjpr2fvvwan.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_878 : Tensor "bf16[128, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=convert_element_type_878]
#   %sum_123 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=sum_123]
#   %sum_123 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_878, [0, 2, 3]), kwargs = {})
#   %convert_element_type_880 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_123, torch.float32), kwargs = {})
#   return %sum_123,%convert_element_type_880
triton_red_fused__to_copy_convolution_backward_63 = async_compile.triton('triton_red_fused__to_copy_convolution_backward_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_convolution_backward_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_convolution_backward_63(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 64)
        r0_2 = r0_index // 64
        tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0 + 32768*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/x3/cx3bhdetzzgwjvmzmxlgyytxhmqqisa76o6bgacfoqpqmycrniyd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_194 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_194]
#   %getitem_40 : Tensor "b8[128, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=getitem_40]
#   %view_520 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_194, [128, 512, 64]), kwargs = {})
#   %permute_366 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_520, [0, 2, 1]), kwargs = {})
#   %convert_element_type_881 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_40, torch.bfloat16), kwargs = {})
#   %mul_343 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_881, 1.1111111111111112), kwargs = {})
#   %mul_344 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_366, %mul_343), kwargs = {})
#   return %mul_344
triton_poi_fused_native_dropout_backward_transpose_view_64 = async_compile.triton('triton_poi_fused_native_dropout_backward_transpose_view_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i1', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_backward_transpose_view_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 29360128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_dropout_backward_transpose_view_64(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/p6/cp6ut72pusxhiqywqapbxoy4sxdq3gxsht5yua36kgxslunbhze2.py
# Topologically Sorted Source Nodes: [matmul_6, attn_7], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   attn_7 => div_4, exp_4, sum_4
#   matmul_6 => view_91
# Graph fragment:
#   %bmm_16 : Tensor "bf16[1024, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_16]
#   %bmm_121 : Tensor "bf16[1024, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_121]
#   %view_527 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_121, [128, 8, 64, 1]), kwargs = {})
#   %convert_element_type_893 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_527, torch.float32), kwargs = {})
#   %view_91 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_16, [128, 8, 64, 1]), kwargs = {})
#   %convert_element_type_default_32 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_91, torch.float32), kwargs = {})
#   %mul_tensor_12 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_32, 1), kwargs = {})
#   %amax_default_6 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_12, [-1], True), kwargs = {})
#   %sub_tensor_6 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_12, %amax_default_6), kwargs = {})
#   %mul_tensor_13 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_6, 0.125), kwargs = {})
#   %exp_4 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_13,), kwargs = {})
#   %sum_4 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_4, [-1], True), kwargs = {})
#   %div_4 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_4, %sum_4), kwargs = {})
#   %mul_345 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_893, %div_4), kwargs = {})
#   %sum_125 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_345, [-1], True), kwargs = {})
#   %neg_18 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_4,), kwargs = {})
#   %fma_6 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.fma.default](args = (%neg_18, %sum_125, %mul_345), kwargs = {})
#   %convert_element_type_894 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%fma_6, torch.bfloat16), kwargs = {})
#   %mul_346 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_894, 0.125), kwargs = {})
#   return %mul_346
triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_65 = async_compile.triton('triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_65(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 - tmp3
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = (tmp7 / tmp7)
    tmp9 = -tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp8
    tmp13 = libdevice.fma(tmp9, tmp12, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/d7/cd77msxgpqj3omx2dktjweh5uwbhvsznvshvajbbj3o3h5znhgiz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_122 : Tensor "bf16[1024, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_122]
#   %bmm_120 : Tensor "bf16[1024, 1, 64][64, 64, 1]cuda:0" = PlaceHolder[target=bmm_120]
#   %view_526 : Tensor "bf16[128, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_120, [128, 8, 1, 64]), kwargs = {})
#   %view_529 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_122, [128, 8, 64, 1]), kwargs = {})
#   %permute_376 : Tensor "bf16[128, 8, 1, 64][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_529, [0, 1, 3, 2]), kwargs = {})
#   %permute_377 : Tensor "bf16[128, 1, 8, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_526, [0, 2, 1, 3]), kwargs = {})
#   %view_531 : Tensor "bf16[128, 1, 512][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_377, [128, 1, 512]), kwargs = {})
#   %permute_378 : Tensor "bf16[128, 1, 8, 64][512, 1, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_376, [0, 2, 1, 3]), kwargs = {})
#   %view_532 : Tensor "bf16[128, 1, 512][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_378, [128, 1, 512]), kwargs = {})
#   %cat_18 : Tensor "bf16[128, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_532, %view_531], 2), kwargs = {})
#   return %cat_18
triton_poi_fused_cat_transpose_view_66 = async_compile.triton('triton_poi_fused_cat_transpose_view_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_transpose_view_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_transpose_view_66(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1024, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (512*x1 + ((-512) + x0)), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/yp/cyp7snaym7lbl7jjutjvxcdm7shjwtrvjttlizfxw62evnuoucka.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_195 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=getitem_195]
#   %convert_element_type_879 : Tensor "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_195, torch.float32), kwargs = {})
#   return %convert_element_type_879
triton_poi_fused__to_copy_67 = async_compile.triton('triton_poi_fused__to_copy_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_67(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/5f/c5fa4h77j525o2uimpgc5szw2wk5gf5z7xzvxcat2qmjbjhlxvnj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_123 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_123]
#   %view_530 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_123, [128, 8, 64, 64]), kwargs = {})
#   %permute_379 : Tensor "bf16[128, 64, 8, 64][32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_530, [0, 2, 1, 3]), kwargs = {})
#   %clone_110 : Tensor "bf16[128, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_379,), kwargs = {memory_format: torch.contiguous_format})
#   %view_533 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_110, [128, 64, 512]), kwargs = {})
#   %view_534 : Tensor "bf16[8192, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_533, [8192, 512]), kwargs = {})
#   return %view_534
triton_poi_fused__unsafe_view_clone_transpose_view_68 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_68(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (64*((x1 % 64)) + 4096*(x0 // 64) + 32768*(x1 // 64) + ((x0 % 64))), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/3h/c3hnfacmxecvxxnyu3rb35btb3bk36leqi2penbvjhpag5fswwy7.py
# Topologically Sorted Source Nodes: [add_17], Original ATen: [aten.fill, aten.add, aten.view, aten.transpose, aten.sigmoid, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   add_17 => add_22
# Graph fragment:
#   %mul_327 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=mul_327]
#   %getitem_194 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_194]
#   %mm_65 : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_65]
#   %convolution_12 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_12]
#   %convolution_10 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %full_36 : Tensor "bf16[128, 8, 8, 512][32768, 4096, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 8, 8, 512], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_359 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_36, [0, 3, 1, 2]), kwargs = {})
#   %add_174 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_327, %getitem_194), kwargs = {})
#   %view_535 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_65, [128, 64, 512]), kwargs = {})
#   %permute_388 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_535, [0, 2, 1]), kwargs = {})
#   %view_536 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_388, [128, 512, 8, 8]), kwargs = {})
#   %add_176 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_174, %view_536), kwargs = {})
#   %add_22 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_67 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %sub_87 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_359, %sigmoid_67), kwargs = {})
#   %mul_347 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sub_87), kwargs = {})
#   %add_177 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_347, 1), kwargs = {})
#   %mul_348 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_67, %add_177), kwargs = {})
#   %mul_349 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_176, %mul_348), kwargs = {})
#   return %mul_349
triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_69 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_69', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 58720256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_69(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp7 * tmp10
    tmp12 = tmp11 + tmp9
    tmp13 = tmp8 * tmp12
    tmp14 = tmp4 * tmp13
    tl.store(in_out_ptr0 + (x0), tmp14, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/e7/ce7mwtv3ew56h5so4tvgqll7i5mnf32hewdbhr7nqf736nsb4o4k.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.clone, aten._unsafe_view, aten.cat, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_117 : Tensor "bf16[128, 512, 12][6144, 12, 1]cuda:0" = PlaceHolder[target=bmm_117]
#   %bmm_119 : Tensor "bf16[128, 12, 512][6144, 512, 1]cuda:0" = PlaceHolder[target=bmm_119]
#   %sum_118 : Tensor "f32[128, 512, 1, 1][512, 1, 65536, 65536]cuda:0" = PlaceHolder[target=sum_118]
#   %bmm_125 : Tensor "bf16[128, 512, 12][6144, 12, 1]cuda:0" = PlaceHolder[target=bmm_125]
#   %bmm_127 : Tensor "bf16[128, 12, 512][6144, 512, 1]cuda:0" = PlaceHolder[target=bmm_127]
#   %sum_127 : Tensor "f32[128, 512, 1, 1][512, 1, 65536, 65536]cuda:0" = PlaceHolder[target=sum_127]
#   %convert_element_type_867 : Tensor "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_118, torch.bfloat16), kwargs = {})
#   %permute_365 : Tensor "bf16[128, 512, 12][6144, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_119, [0, 2, 1]), kwargs = {})
#   %view_510 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_867, [128, 512]), kwargs = {})
#   %clone_106 : Tensor "bf16[128, 512, 12][6144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_365,), kwargs = {memory_format: torch.contiguous_format})
#   %view_511 : Tensor "bf16[128, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_106, [128, 6144]), kwargs = {})
#   %view_512 : Tensor "bf16[128, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_117, [128, 6144]), kwargs = {})
#   %cat_17 : Tensor "bf16[128, 12800][12800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_512, %view_511, %view_510], 1), kwargs = {})
#   %convert_element_type_912 : Tensor "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_127, torch.bfloat16), kwargs = {})
#   %permute_396 : Tensor "bf16[128, 512, 12][6144, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_127, [0, 2, 1]), kwargs = {})
#   %view_539 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_912, [128, 512]), kwargs = {})
#   %clone_112 : Tensor "bf16[128, 512, 12][6144, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_396,), kwargs = {memory_format: torch.contiguous_format})
#   %view_540 : Tensor "bf16[128, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_112, [128, 6144]), kwargs = {})
#   %view_541 : Tensor "bf16[128, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_125, [128, 6144]), kwargs = {})
#   %cat_19 : Tensor "bf16[128, 12800][12800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_541, %view_540, %view_539], 1), kwargs = {})
#   %add_180 : Tensor "bf16[128, 12800][12800, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_17, %cat_19), kwargs = {})
#   return %add_180
triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_70 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26214400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 12800)
    x1 = xindex // 12800
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6144, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (6144*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 12288, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (512*((((-6144) + x0) % 12)) + 6144*x1 + (((((-6144) + x0) // 12) % 512))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12800, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (512*x1 + ((-12288) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp11, tmp15, tmp16)
    tmp18 = tl.where(tmp9, tmp10, tmp17)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tmp20 = tl.load(in_ptr3 + (6144*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (512*((((-6144) + x0) % 12)) + 6144*x1 + (((((-6144) + x0) // 12) % 512))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (512*x1 + ((-12288) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp11, tmp23, tmp24)
    tmp26 = tl.where(tmp9, tmp21, tmp25)
    tmp27 = tl.where(tmp4, tmp20, tmp26)
    tmp28 = tmp19 + tmp27
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/3k/c3kd3d7336keguy7q5jbu4yeicq4er2k2hzz25se24g33nge4oqe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_180 : Tensor "bf16[128, 12800][12800, 1]cuda:0" = PlaceHolder[target=add_180]
#   %sum_170 : Tensor "f32[1, 12800][12800, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_180, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_170
triton_red_fused_sum_71 = async_compile.triton('triton_red_fused_sum_71', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3379200, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_71(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12800
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 12800*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pz/cpzvyeljmrojj6adyem2kmihv46gsvuyrrbsgggkidq64gq2b5cy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_85 : Tensor "bf16[12800, 256][256, 1]cuda:0" = PlaceHolder[target=mm_85]
#   %convert_element_type_1089 : Tensor "f32[12800, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_85, torch.float32), kwargs = {})
#   return %convert_element_type_1089
triton_poi_fused__to_copy_72 = async_compile.triton('triton_poi_fused__to_copy_72', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_72(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3276800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/yn/cynfahtk2ba3o2ybagamiu6tjlytyidu3duoptp33lclae32zz5y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_349 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=mul_349]
#   %getitem_200 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=getitem_200]
#   %mm_71 : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_71]
#   %add_183 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_349, %getitem_200), kwargs = {})
#   %view_564 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_71, [128, 64, 512]), kwargs = {})
#   %permute_419 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_564, [0, 2, 1]), kwargs = {})
#   %view_565 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_419, [128, 512, 8, 8]), kwargs = {})
#   %add_185 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_183, %view_565), kwargs = {})
#   return %add_185
triton_poi_fused_add_transpose_view_73 = async_compile.triton('triton_poi_fused_add_transpose_view_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_transpose_view_73', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_transpose_view_73(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/c3/cc3baxgkidfe2t44wfz5q7bpl23udxpxedmxds2xcgmbfznbrszx.py
# Topologically Sorted Source Nodes: [input_27, unsqueeze_5, gate], Original ATen: [aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   gate => unsqueeze_8
#   input_27 => sigmoid_18
#   unsqueeze_5 => unsqueeze_7
# Graph fragment:
#   %add_163 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_163]
#   %addmm_17 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %getitem_203 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_203]
#   %mul_370 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_370]
#   %sigmoid_18 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_17,), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[128, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_18, -1), kwargs = {})
#   %unsqueeze_8 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, -1), kwargs = {})
#   %mul_303 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %unsqueeze_8), kwargs = {})
#   %add_186 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_303, %getitem_203), kwargs = {})
#   %mul_371 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_186, %mul_370), kwargs = {})
#   return %mul_371
triton_poi_fused_add_mul_sigmoid_unsqueeze_74 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_unsqueeze_74', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_unsqueeze_74', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83951616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_unsqueeze_74(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 65536
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + 256*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/te/ctehldefkmxed4xiuvq3qradbu2mctvpxcwbgljxmfco4rpytry3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_204 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=getitem_204]
#   %convert_element_type_954 : Tensor "f32[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_204, torch.float32), kwargs = {})
#   return %convert_element_type_954
triton_poi_fused__to_copy_75 = async_compile.triton('triton_poi_fused__to_copy_75', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_75', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/aq/caqx66hccjmmy2qppvasoytpm3njx6blq36xgm3sgmglmhow4gau.py
# Topologically Sorted Source Nodes: [add_9], Original ATen: [aten.fill, aten.add, aten.view, aten.transpose, aten.sigmoid, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   add_9 => add_12
# Graph fragment:
#   %mul_371 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_371]
#   %getitem_209 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=getitem_209]
#   %mm_77 : Tensor "bf16[32768, 256][256, 1]cuda:0" = PlaceHolder[target=mm_77]
#   %convolution_7 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_7]
#   %convolution_5 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %full_17 : Tensor "bf16[128, 16, 16, 256][65536, 4096, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 16, 16, 256], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_178 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%full_17, [0, 3, 1, 2]), kwargs = {})
#   %add_193 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %getitem_209), kwargs = {})
#   %view_593 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_77, [128, 256, 256]), kwargs = {})
#   %permute_450 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_593, [0, 2, 1]), kwargs = {})
#   %view_594 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_450, [128, 256, 16, 16]), kwargs = {})
#   %add_195 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_193, %view_594), kwargs = {})
#   %add_12 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_71 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %sub_95 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_178, %sigmoid_71), kwargs = {})
#   %mul_391 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sub_95), kwargs = {})
#   %add_196 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_391, 1), kwargs = {})
#   %mul_392 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_71, %add_196), kwargs = {})
#   %mul_393 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_195, %mul_392), kwargs = {})
#   return %mul_393
triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_76 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_76', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_76', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 117440512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_76(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp7 * tmp10
    tmp12 = tmp11 + tmp9
    tmp13 = tmp8 * tmp12
    tmp14 = tmp4 * tmp13
    tl.store(in_out_ptr0 + (x0), tmp14, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vt/cvtmd2ryyqrfxdalypazqsmaargcc5joyplhwencyws4a6e44324.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.clone, aten._unsafe_view, aten.cat, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_73 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0" = PlaceHolder[target=bmm_73]
#   %bmm_75 : Tensor "bf16[128, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=bmm_75]
#   %sum_58 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_58]
#   %bmm_81 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0" = PlaceHolder[target=bmm_81]
#   %bmm_83 : Tensor "bf16[128, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=bmm_83]
#   %sum_67 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_67]
#   %add_131 : Tensor "bf16[128, 6400][6400, 1]cuda:0" = PlaceHolder[target=add_131]
#   %bmm_89 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0" = PlaceHolder[target=bmm_89]
#   %bmm_91 : Tensor "bf16[128, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=bmm_91]
#   %sum_76 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_76]
#   %bmm_97 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0" = PlaceHolder[target=bmm_97]
#   %bmm_99 : Tensor "bf16[128, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=bmm_99]
#   %sum_85 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_85]
#   %add_149 : Tensor "bf16[128, 6400][6400, 1]cuda:0" = PlaceHolder[target=add_149]
#   %bmm_105 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0" = PlaceHolder[target=bmm_105]
#   %bmm_107 : Tensor "bf16[128, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=bmm_107]
#   %sum_94 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_94]
#   %bmm_133 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0" = PlaceHolder[target=bmm_133]
#   %bmm_135 : Tensor "bf16[128, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=bmm_135]
#   %sum_137 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_137]
#   %add_190 : Tensor "bf16[128, 6400][6400, 1]cuda:0" = PlaceHolder[target=add_190]
#   %bmm_141 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0" = PlaceHolder[target=bmm_141]
#   %bmm_143 : Tensor "bf16[128, 12, 256][3072, 256, 1]cuda:0" = PlaceHolder[target=bmm_143]
#   %sum_146 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]cuda:0" = PlaceHolder[target=sum_146]
#   %convert_element_type_594 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_58, torch.bfloat16), kwargs = {})
#   %permute_184 : Tensor "bf16[128, 256, 12][3072, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_75, [0, 2, 1]), kwargs = {})
#   %view_347 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_594, [128, 256]), kwargs = {})
#   %clone_74 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_184,), kwargs = {memory_format: torch.contiguous_format})
#   %view_348 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_74, [128, 3072]), kwargs = {})
#   %view_349 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_73, [128, 3072]), kwargs = {})
#   %cat_7 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_349, %view_348, %view_347], 1), kwargs = {})
#   %convert_element_type_639 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_67, torch.bfloat16), kwargs = {})
#   %permute_215 : Tensor "bf16[128, 256, 12][3072, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_83, [0, 2, 1]), kwargs = {})
#   %view_376 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_639, [128, 256]), kwargs = {})
#   %clone_80 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_215,), kwargs = {memory_format: torch.contiguous_format})
#   %view_377 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_80, [128, 3072]), kwargs = {})
#   %view_378 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_81, [128, 3072]), kwargs = {})
#   %cat_9 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_378, %view_377, %view_376], 1), kwargs = {})
#   %add_131 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_7, %cat_9), kwargs = {})
#   %convert_element_type_684 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_76, torch.bfloat16), kwargs = {})
#   %permute_246 : Tensor "bf16[128, 256, 12][3072, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_91, [0, 2, 1]), kwargs = {})
#   %view_405 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_684, [128, 256]), kwargs = {})
#   %clone_86 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_246,), kwargs = {memory_format: torch.contiguous_format})
#   %view_406 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_86, [128, 3072]), kwargs = {})
#   %view_407 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_89, [128, 3072]), kwargs = {})
#   %cat_11 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_407, %view_406, %view_405], 1), kwargs = {})
#   %add_140 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_131, %cat_11), kwargs = {})
#   %convert_element_type_729 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_85, torch.bfloat16), kwargs = {})
#   %permute_277 : Tensor "bf16[128, 256, 12][3072, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_99, [0, 2, 1]), kwargs = {})
#   %view_434 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_729, [128, 256]), kwargs = {})
#   %clone_92 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_277,), kwargs = {memory_format: torch.contiguous_format})
#   %view_435 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_92, [128, 3072]), kwargs = {})
#   %view_436 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_97, [128, 3072]), kwargs = {})
#   %cat_13 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_436, %view_435, %view_434], 1), kwargs = {})
#   %add_149 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_140, %cat_13), kwargs = {})
#   %convert_element_type_774 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_94, torch.bfloat16), kwargs = {})
#   %permute_308 : Tensor "bf16[128, 256, 12][3072, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_107, [0, 2, 1]), kwargs = {})
#   %view_463 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_774, [128, 256]), kwargs = {})
#   %clone_98 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_308,), kwargs = {memory_format: torch.contiguous_format})
#   %view_464 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_98, [128, 3072]), kwargs = {})
#   %view_465 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_105, [128, 3072]), kwargs = {})
#   %cat_15 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_465, %view_464, %view_463], 1), kwargs = {})
#   %add_158 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %cat_15), kwargs = {})
#   %convert_element_type_959 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_137, torch.bfloat16), kwargs = {})
#   %permute_427 : Tensor "bf16[128, 256, 12][3072, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_135, [0, 2, 1]), kwargs = {})
#   %view_568 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_959, [128, 256]), kwargs = {})
#   %clone_118 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_427,), kwargs = {memory_format: torch.contiguous_format})
#   %view_569 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_118, [128, 3072]), kwargs = {})
#   %view_570 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_133, [128, 3072]), kwargs = {})
#   %cat_21 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_570, %view_569, %view_568], 1), kwargs = {})
#   %add_190 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_158, %cat_21), kwargs = {})
#   %convert_element_type_1004 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_146, torch.bfloat16), kwargs = {})
#   %permute_458 : Tensor "bf16[128, 256, 12][3072, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_143, [0, 2, 1]), kwargs = {})
#   %view_597 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1004, [128, 256]), kwargs = {})
#   %clone_124 : Tensor "bf16[128, 256, 12][3072, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_458,), kwargs = {memory_format: torch.contiguous_format})
#   %view_598 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_124, [128, 3072]), kwargs = {})
#   %view_599 : Tensor "bf16[128, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_141, [128, 3072]), kwargs = {})
#   %cat_23 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_599, %view_598, %view_597], 1), kwargs = {})
#   %add_199 : Tensor "bf16[128, 6400][6400, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_190, %cat_23), kwargs = {})
#   return %add_131,%add_149,%add_190,%add_199
triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_77 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_77', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*fp32', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'in_ptr14': '*fp32', 'in_ptr15': '*bf16', 'in_ptr16': '*bf16', 'in_ptr17': '*fp32', 'in_ptr18': '*bf16', 'in_ptr19': '*bf16', 'in_ptr20': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_77', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37683200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_77(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6400)
    x1 = xindex // 6400
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3072, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (3072*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 6144, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (256*((((-3072) + x0) % 12)) + 3072*x1 + (((((-3072) + x0) // 12) % 256))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 6400, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (256*x1 + ((-6144) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp11, tmp15, tmp16)
    tmp18 = tl.where(tmp9, tmp10, tmp17)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tmp20 = tl.load(in_ptr3 + (3072*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (256*((((-3072) + x0) % 12)) + 3072*x1 + (((((-3072) + x0) // 12) % 256))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (256*x1 + ((-6144) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp11, tmp23, tmp24)
    tmp26 = tl.where(tmp9, tmp21, tmp25)
    tmp27 = tl.where(tmp4, tmp20, tmp26)
    tmp28 = tmp19 + tmp27
    tmp29 = tl.load(in_ptr6 + (3072*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr7 + (256*((((-3072) + x0) % 12)) + 3072*x1 + (((((-3072) + x0) // 12) % 256))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr8 + (256*x1 + ((-6144) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp11, tmp32, tmp33)
    tmp35 = tl.where(tmp9, tmp30, tmp34)
    tmp36 = tl.where(tmp4, tmp29, tmp35)
    tmp37 = tmp28 + tmp36
    tmp38 = tl.load(in_ptr9 + (3072*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr10 + (256*((((-3072) + x0) % 12)) + 3072*x1 + (((((-3072) + x0) // 12) % 256))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr11 + (256*x1 + ((-6144) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp11, tmp41, tmp42)
    tmp44 = tl.where(tmp9, tmp39, tmp43)
    tmp45 = tl.where(tmp4, tmp38, tmp44)
    tmp46 = tmp37 + tmp45
    tmp47 = tl.load(in_ptr12 + (3072*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr13 + (256*((((-3072) + x0) % 12)) + 3072*x1 + (((((-3072) + x0) // 12) % 256))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp49 = tl.load(in_ptr14 + (256*x1 + ((-6144) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp11, tmp50, tmp51)
    tmp53 = tl.where(tmp9, tmp48, tmp52)
    tmp54 = tl.where(tmp4, tmp47, tmp53)
    tmp55 = tmp46 + tmp54
    tmp56 = tl.load(in_ptr15 + (3072*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp57 = tl.load(in_ptr16 + (256*((((-3072) + x0) % 12)) + 3072*x1 + (((((-3072) + x0) // 12) % 256))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp58 = tl.load(in_ptr17 + (256*x1 + ((-6144) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp11, tmp59, tmp60)
    tmp62 = tl.where(tmp9, tmp57, tmp61)
    tmp63 = tl.where(tmp4, tmp56, tmp62)
    tmp64 = tmp55 + tmp63
    tmp65 = tl.load(in_ptr18 + (3072*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp66 = tl.load(in_ptr19 + (256*((((-3072) + x0) % 12)) + 3072*x1 + (((((-3072) + x0) // 12) % 256))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr20 + (256*x1 + ((-6144) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp11, tmp68, tmp69)
    tmp71 = tl.where(tmp9, tmp66, tmp70)
    tmp72 = tl.where(tmp4, tmp65, tmp71)
    tmp73 = tmp64 + tmp72
    tl.store(in_out_ptr0 + (x2), tmp73, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/u6/cu64regkvrqj7doiwih5fiyjfls4tgcx46ejfko2lfdwrfg22mm2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_199 : Tensor "bf16[128, 6400][6400, 1]cuda:0" = PlaceHolder[target=add_199]
#   %sum_172 : Tensor "f32[1, 6400][6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_199, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_172
triton_red_fused_sum_78 = async_compile.triton('triton_red_fused_sum_78', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_78', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1689600, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_78(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6400
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 6400*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qu/cquqzxdi4xfc576jjphd6at6ubwmgnkjax42zct76hmstpxn6x5d.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_89 : Tensor "bf16[6400, 256][256, 1]cuda:0" = PlaceHolder[target=mm_89]
#   %convert_element_type_1103 : Tensor "f32[6400, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_89, torch.float32), kwargs = {})
#   return %convert_element_type_1103
triton_poi_fused__to_copy_79 = async_compile.triton('triton_poi_fused__to_copy_79', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_79', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_79(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/q2/cq2gkv6b3p5zzbsfjcoslfhtccjyufzxnky7wp5i5dq4m6uomfez.py
# Topologically Sorted Source Nodes: [input_41, unsqueeze_12, gate_1], Original ATen: [aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   gate_1 => unsqueeze_15
#   input_41 => sigmoid_30
#   unsqueeze_12 => unsqueeze_14
# Graph fragment:
#   %add_118 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=add_118]
#   %addmm_24 : Tensor "bf16[128, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_24]
#   %getitem_218 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_218]
#   %mul_414 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_414]
#   %sigmoid_30 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_24,), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[128, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_30, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %mul_186 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_118, %unsqueeze_15), kwargs = {})
#   %add_205 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_186, %getitem_218), kwargs = {})
#   %mul_415 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %mul_414), kwargs = {})
#   return %mul_415
triton_poi_fused_add_mul_sigmoid_unsqueeze_80 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_unsqueeze_80', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_unsqueeze_80', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67141632, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_unsqueeze_80(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_out_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 1024*y3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lu/cluozxxvvsdvby46iepdv4zo7hhqu3u2fp7hmnxf2oyxq4wi2h34.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_219 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=getitem_219]
#   %convert_element_type_1046 : Tensor "f32[256, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_219, torch.float32), kwargs = {})
#   return %convert_element_type_1046
triton_poi_fused__to_copy_81 = async_compile.triton('triton_poi_fused__to_copy_81', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_81', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2949120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_81(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/gv/cgvata5apw6d6z4j76kmzigkgu6xqkkwxmlicsk4v73o2omhtwf4.py
# Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.fill, aten.add, aten.sigmoid, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   add_2 => add_3
# Graph fragment:
#   %mul_415 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_415]
#   %getitem_224 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_224]
#   %convolution_2 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_2]
#   %convolution : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution]
#   %full_1 : Tensor "bf16[128, 32, 32, 128][131072, 4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 32, 32, 128], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_129 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%full_1, [0, 3, 1, 2]), kwargs = {})
#   %add_212 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_415, %getitem_224), kwargs = {})
#   %add_3 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution), kwargs = {})
#   %sigmoid_75 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %sub_103 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_129, %sigmoid_75), kwargs = {})
#   %mul_431 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sub_103), kwargs = {})
#   %add_213 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_431, 1), kwargs = {})
#   %mul_432 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_75, %add_213), kwargs = {})
#   %mul_433 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_212, %mul_432), kwargs = {})
#   return %mul_433
triton_poi_fused_add_fill_mul_sigmoid_sub_82 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_sub_82', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_82', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 100663296, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_sub_82(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_out_ptr0 + (y0 + 1024*x2 + 131072*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp5 * tmp8
    tmp10 = tmp9 + tmp7
    tmp11 = tmp6 * tmp10
    tmp12 = tmp2 * tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + 1024*x2 + 131072*y1), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pj/cpjlmlb3iiajdi5cfsftglzlhgzoghtnwsflwnayv23ckr5t2btg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.clone, aten._unsafe_view, aten.cat, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %bmm_53 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0" = PlaceHolder[target=bmm_53]
#   %bmm_55 : Tensor "bf16[128, 12, 128][1536, 128, 1]cuda:0" = PlaceHolder[target=bmm_55]
#   %sum_19 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_19]
#   %bmm_57 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %bmm_59 : Tensor "bf16[128, 12, 128][1536, 128, 1]cuda:0" = PlaceHolder[target=bmm_59]
#   %sum_26 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_26]
#   %add_94 : Tensor "bf16[128, 3200][3200, 1]cuda:0" = PlaceHolder[target=add_94]
#   %bmm_61 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0" = PlaceHolder[target=bmm_61]
#   %bmm_63 : Tensor "bf16[128, 12, 128][1536, 128, 1]cuda:0" = PlaceHolder[target=bmm_63]
#   %sum_33 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_33]
#   %bmm_65 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0" = PlaceHolder[target=bmm_65]
#   %bmm_67 : Tensor "bf16[128, 12, 128][1536, 128, 1]cuda:0" = PlaceHolder[target=bmm_67]
#   %sum_40 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_40]
#   %add_108 : Tensor "bf16[128, 3200][3200, 1]cuda:0" = PlaceHolder[target=add_108]
#   %bmm_69 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0" = PlaceHolder[target=bmm_69]
#   %bmm_71 : Tensor "bf16[128, 12, 128][1536, 128, 1]cuda:0" = PlaceHolder[target=bmm_71]
#   %sum_47 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_47]
#   %bmm_149 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0" = PlaceHolder[target=bmm_149]
#   %bmm_151 : Tensor "bf16[128, 12, 128][1536, 128, 1]cuda:0" = PlaceHolder[target=bmm_151]
#   %sum_156 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_156]
#   %add_209 : Tensor "bf16[128, 3200][3200, 1]cuda:0" = PlaceHolder[target=add_209]
#   %bmm_153 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0" = PlaceHolder[target=bmm_153]
#   %bmm_155 : Tensor "bf16[128, 12, 128][1536, 128, 1]cuda:0" = PlaceHolder[target=bmm_155]
#   %sum_163 : Tensor "f32[128, 128, 1, 1][128, 1, 16384, 16384]cuda:0" = PlaceHolder[target=sum_163]
#   %convert_element_type_489 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_19, torch.bfloat16), kwargs = {})
#   %permute_135 : Tensor "bf16[128, 128, 12][1536, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_55, [0, 2, 1]), kwargs = {})
#   %view_285 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_489, [128, 128]), kwargs = {})
#   %clone_59 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_135,), kwargs = {memory_format: torch.contiguous_format})
#   %view_286 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_59, [128, 1536]), kwargs = {})
#   %view_287 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_53, [128, 1536]), kwargs = {})
#   %cat_2 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_287, %view_286, %view_285], 1), kwargs = {})
#   %convert_element_type_506 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.bfloat16), kwargs = {})
#   %permute_143 : Tensor "bf16[128, 128, 12][1536, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_59, [0, 2, 1]), kwargs = {})
#   %view_297 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_506, [128, 128]), kwargs = {})
#   %clone_62 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_143,), kwargs = {memory_format: torch.contiguous_format})
#   %view_298 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_62, [128, 1536]), kwargs = {})
#   %view_299 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_57, [128, 1536]), kwargs = {})
#   %cat_3 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_299, %view_298, %view_297], 1), kwargs = {})
#   %add_94 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_2, %cat_3), kwargs = {})
#   %convert_element_type_523 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_33, torch.bfloat16), kwargs = {})
#   %permute_151 : Tensor "bf16[128, 128, 12][1536, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_63, [0, 2, 1]), kwargs = {})
#   %view_309 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_523, [128, 128]), kwargs = {})
#   %clone_65 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_151,), kwargs = {memory_format: torch.contiguous_format})
#   %view_310 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_65, [128, 1536]), kwargs = {})
#   %view_311 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_61, [128, 1536]), kwargs = {})
#   %cat_4 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_311, %view_310, %view_309], 1), kwargs = {})
#   %add_101 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %cat_4), kwargs = {})
#   %convert_element_type_540 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_40, torch.bfloat16), kwargs = {})
#   %permute_159 : Tensor "bf16[128, 128, 12][1536, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_67, [0, 2, 1]), kwargs = {})
#   %view_321 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_540, [128, 128]), kwargs = {})
#   %clone_68 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_159,), kwargs = {memory_format: torch.contiguous_format})
#   %view_322 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_68, [128, 1536]), kwargs = {})
#   %view_323 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_65, [128, 1536]), kwargs = {})
#   %cat_5 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_323, %view_322, %view_321], 1), kwargs = {})
#   %add_108 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_101, %cat_5), kwargs = {})
#   %convert_element_type_557 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_47, torch.bfloat16), kwargs = {})
#   %permute_167 : Tensor "bf16[128, 128, 12][1536, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_71, [0, 2, 1]), kwargs = {})
#   %view_333 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_557, [128, 128]), kwargs = {})
#   %clone_71 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_167,), kwargs = {memory_format: torch.contiguous_format})
#   %view_334 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_71, [128, 1536]), kwargs = {})
#   %view_335 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_69, [128, 1536]), kwargs = {})
#   %cat_6 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_335, %view_334, %view_333], 1), kwargs = {})
#   %add_115 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %cat_6), kwargs = {})
#   %convert_element_type_1051 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_156, torch.bfloat16), kwargs = {})
#   %permute_489 : Tensor "bf16[128, 128, 12][1536, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_151, [0, 2, 1]), kwargs = {})
#   %view_626 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1051, [128, 128]), kwargs = {})
#   %clone_130 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_489,), kwargs = {memory_format: torch.contiguous_format})
#   %view_627 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_130, [128, 1536]), kwargs = {})
#   %view_628 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_149, [128, 1536]), kwargs = {})
#   %cat_25 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_628, %view_627, %view_626], 1), kwargs = {})
#   %add_209 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_115, %cat_25), kwargs = {})
#   %convert_element_type_1068 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_163, torch.bfloat16), kwargs = {})
#   %permute_497 : Tensor "bf16[128, 128, 12][1536, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_155, [0, 2, 1]), kwargs = {})
#   %view_638 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1068, [128, 128]), kwargs = {})
#   %clone_133 : Tensor "bf16[128, 128, 12][1536, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_497,), kwargs = {memory_format: torch.contiguous_format})
#   %view_639 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_133, [128, 1536]), kwargs = {})
#   %view_640 : Tensor "bf16[128, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_153, [128, 1536]), kwargs = {})
#   %cat_26 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_640, %view_639, %view_638], 1), kwargs = {})
#   %add_216 : Tensor "bf16[128, 3200][3200, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_209, %cat_26), kwargs = {})
#   return %add_94,%add_108,%add_209,%add_216
triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_83 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_83', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*fp32', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'in_ptr14': '*fp32', 'in_ptr15': '*bf16', 'in_ptr16': '*bf16', 'in_ptr17': '*fp32', 'in_ptr18': '*bf16', 'in_ptr19': '*bf16', 'in_ptr20': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_83', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18841600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_83(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3200)
    x1 = xindex // 3200
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1536, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 3072, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 3200, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp11, tmp15, tmp16)
    tmp18 = tl.where(tmp9, tmp10, tmp17)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tmp20 = tl.load(in_ptr3 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp11, tmp23, tmp24)
    tmp26 = tl.where(tmp9, tmp21, tmp25)
    tmp27 = tl.where(tmp4, tmp20, tmp26)
    tmp28 = tmp19 + tmp27
    tmp29 = tl.load(in_ptr6 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr7 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr8 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp11, tmp32, tmp33)
    tmp35 = tl.where(tmp9, tmp30, tmp34)
    tmp36 = tl.where(tmp4, tmp29, tmp35)
    tmp37 = tmp28 + tmp36
    tmp38 = tl.load(in_ptr9 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr10 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr11 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp11, tmp41, tmp42)
    tmp44 = tl.where(tmp9, tmp39, tmp43)
    tmp45 = tl.where(tmp4, tmp38, tmp44)
    tmp46 = tmp37 + tmp45
    tmp47 = tl.load(in_ptr12 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr13 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp49 = tl.load(in_ptr14 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp11, tmp50, tmp51)
    tmp53 = tl.where(tmp9, tmp48, tmp52)
    tmp54 = tl.where(tmp4, tmp47, tmp53)
    tmp55 = tmp46 + tmp54
    tmp56 = tl.load(in_ptr15 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp57 = tl.load(in_ptr16 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp58 = tl.load(in_ptr17 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp11, tmp59, tmp60)
    tmp62 = tl.where(tmp9, tmp57, tmp61)
    tmp63 = tl.where(tmp4, tmp56, tmp62)
    tmp64 = tmp55 + tmp63
    tmp65 = tl.load(in_ptr18 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp66 = tl.load(in_ptr19 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr20 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp11, tmp68, tmp69)
    tmp71 = tl.where(tmp9, tmp66, tmp70)
    tmp72 = tl.where(tmp4, tmp65, tmp71)
    tmp73 = tmp64 + tmp72
    tl.store(in_out_ptr0 + (x2), tmp73, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/x2/cx2honthzqmthvhjizfmkjwzbqzrvgweq3rqn3nfmt7njvduh4n4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.fill, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_96 : Tensor "bf16[128, 512][512, 1]cuda:0" = PlaceHolder[target=mm_96]
#   %addmm_2 : Tensor "bf16[128, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %sigmoid_80 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_2,), kwargs = {})
#   %full_default_41 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 512], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_110 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_41, %sigmoid_80), kwargs = {})
#   %mul_458 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_2, %sub_110), kwargs = {})
#   %add_226 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_458, 1), kwargs = {})
#   %mul_459 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_80, %add_226), kwargs = {})
#   %mul_460 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_96, %mul_459), kwargs = {})
#   return %mul_460
triton_poi_fused_add_fill_mul_sigmoid_sub_84 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_sub_84', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_84', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_sub_84(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7s/c7ssb3pzu7g3gprmeh5haye72lm3n5hdacgoham45jxy47jfb24d.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.slice]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_98 : Tensor "bf16[128, 512][512, 1]cuda:0" = PlaceHolder[target=mm_98]
#   %convert_element_type_1138 : Tensor "f32[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_98, torch.float32), kwargs = {})
#   %slice_1 : Tensor "f32[128, 256][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1138, 1, 0, 256), kwargs = {})
#   %convert_element_type_1141 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1141
triton_poi_fused__to_copy_slice_85 = async_compile.triton('triton_poi_fused__to_copy_slice_85', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_slice_85', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_slice_85(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ub/cubo43odhr2tefr7spice2c4k6a26fkmssrcusl6wuvfqrcsdsku.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.fill, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_100 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_100]
#   %addmm : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=addmm]
#   %sigmoid_81 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm,), kwargs = {})
#   %full_default_42 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_111 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_42, %sigmoid_81), kwargs = {})
#   %mul_461 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, %sub_111), kwargs = {})
#   %add_227 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_461, 1), kwargs = {})
#   %mul_462 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_81, %add_227), kwargs = {})
#   %mul_463 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_100, %mul_462), kwargs = {})
#   return %mul_463
triton_poi_fused_add_fill_mul_sigmoid_sub_86 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_sub_86', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_86', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_sub_86(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/37/c37v6hlaudgn3ejcdi2qs5mz6ovjplqveopfsvzw4ma3iniqaohe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_460 : Tensor "bf16[128, 512][512, 1]cuda:0" = PlaceHolder[target=mul_460]
#   %sum_177 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_460, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_177
triton_red_fused_sum_87 = async_compile.triton('triton_red_fused_sum_87', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_87', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 135168, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_87(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/p5/cp5fzvembg6twya5htfxtpgtcaliyoiz2o66kd6d7ufl5ejqrb6d.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_463 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mul_463]
#   %sum_179 : Tensor "f32[1, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_463, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_179
triton_red_fused_sum_88 = async_compile.triton('triton_red_fused_sum_88', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_88', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 270336, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_88(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qj/cqj4na2tdx53r37dnojn36xalufiwyvm2yikv3sjbu4g5ljlkdzr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_216 : Tensor "bf16[128, 3200][3200, 1]cuda:0" = PlaceHolder[target=add_216]
#   %sum_174 : Tensor "f32[1, 3200][3200, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_216, [0], True), kwargs = {dtype: torch.float32})
#   return %sum_174
triton_red_fused_sum_89 = async_compile.triton('triton_red_fused_sum_89', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_89', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 844800, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_89(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3200
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3200*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/bd/cbd4dwp23vuj4jqfhfedwax3gshyj34zv5lalj4i4a7yrvxiimwc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_93 : Tensor "bf16[3200, 256][256, 1]cuda:0" = PlaceHolder[target=mm_93]
#   %convert_element_type_1117 : Tensor "f32[3200, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_93, torch.float32), kwargs = {})
#   return %convert_element_type_1117
triton_poi_fused__to_copy_90 = async_compile.triton('triton_poi_fused__to_copy_90', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_90', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_90(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hw/chwwumwsu2thyrc7hx7acacvof3ud55dwnnscv5mjbhyefoqyduw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_433 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=mul_433]
#   %getitem_230 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=getitem_230]
#   %add_219 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_433, %getitem_230), kwargs = {})
#   return %add_219
triton_poi_fused_add_91 = async_compile.triton('triton_poi_fused_add_91', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_91', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 33554432, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_91(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_out_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 1024*y3), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_70, primals_76, primals_150, primals_151, convert_element_type_2, addmm, convert_element_type_7, convert_element_type_15, addmm_2, convert_element_type_20, addmm_3, addmm_4, convert_element_type_32, addmm_6, convert_element_type_44, addmm_8, convert_element_type_56, convert_element_type_63, convert_element_type_64, convolution, convert_element_type_66, convolution_1, squeeze, squeeze_1, convert_element_type_74, convert_element_type_75, convolution_2, convert_element_type_77, convert_element_type_79, convolution_3, squeeze_2, squeeze_3, convert_element_type_87, convert_element_type_88, convert_element_type_90, convert_element_type_92, convolution_5, bmm_4, view_27, getitem_13, add_8, convert_element_type_111, convolution_6, squeeze_4, squeeze_5, convert_element_type_119, convert_element_type_120, convolution_7, view_38, bmm_8, view_50, getitem_22, add_13, convert_element_type_141, convolution_8, squeeze_6, squeeze_7, convert_element_type_149, convert_element_type_150, convert_element_type_152, convert_element_type_154, convolution_10, bmm_12, view_73, getitem_31, add_18, convert_element_type_173, convolution_11, squeeze_8, squeeze_9, convert_element_type_181, convert_element_type_182, convolution_12, view_84, bmm_16, view_96, getitem_40, add_23, convert_element_type_203, convolution_13, squeeze_10, squeeze_11, convert_element_type_211, convert_element_type_212, add_27, getitem_47, rsqrt_6, view_107, bmm_20, amax_default_5, sum_5, view_117, add_30, getitem_49, rsqrt_7, convert_element_type_235, convert_element_type_236, addmm_16, convert_element_type_243, addmm_17, view_121, bmm_22, view_133, getitem_53, add_34, convert_element_type_267, convolution_16, squeeze_12, squeeze_13, convert_element_type_275, convert_element_type_276, view_144, bmm_26, view_156, getitem_62, add_39, convert_element_type_297, convolution_18, squeeze_14, squeeze_15, convert_element_type_305, convert_element_type_306, view_167, bmm_30, view_179, getitem_71, add_44, convert_element_type_327, convolution_20, squeeze_16, squeeze_17, convert_element_type_335, convert_element_type_336, view_190, bmm_34, view_202, getitem_80, add_49, convert_element_type_357, convolution_22, squeeze_18, squeeze_19, convert_element_type_365, convert_element_type_366, view_213, bmm_38, view_225, getitem_89, add_54, convert_element_type_387, convolution_24, squeeze_20, squeeze_21, convert_element_type_395, convert_element_type_396, convert_element_type_398, convert_element_type_400, addmm_23, convert_element_type_407, addmm_24, add_59, convert_element_type_414, convolution_27, squeeze_22, squeeze_23, convert_element_type_422, convert_element_type_423, convert_element_type_425, convert_element_type_427, convolution_29, squeeze_24, squeeze_25, convert_element_type_435, convert_element_type_436, convert_element_type_438, convert_element_type_440, convolution_31, squeeze_26, squeeze_27, convert_element_type_448, convert_element_type_449, convert_element_type_451, convert_element_type_453, convolution_33, squeeze_28, squeeze_29, convert_element_type_461, convert_element_type_462, convert_element_type_464, convert_element_type_466, convolution_35, squeeze_30, squeeze_31, convert_element_type_474, convert_element_type_475, add_79, getitem_121, rsqrt_18, convert_element_type_480, convert_element_type_481, mul_99, permute_131, permute_132, permute_133, permute_134, mul_114, mul_117, permute_140, permute_142, mul_132, mul_135, permute_148, permute_150, mul_150, mul_153, permute_156, permute_158, mul_168, mul_171, permute_164, permute_166, permute_168, permute_173, mul_193, mul_196, permute_180, permute_181, permute_182, permute_183, permute_186, permute_191, permute_192, permute_193, permute_194, permute_201, permute_205, mul_215, mul_218, permute_212, permute_214, permute_217, permute_222, permute_223, permute_224, permute_225, permute_232, permute_236, mul_237, mul_240, permute_243, permute_245, permute_248, permute_253, permute_254, permute_255, permute_256, permute_263, permute_267, mul_259, mul_262, permute_274, permute_276, permute_279, permute_284, permute_285, permute_286, permute_287, permute_294, permute_298, mul_281, mul_284, permute_305, permute_307, permute_310, permute_315, permute_316, permute_317, permute_318, permute_325, permute_329, permute_332, permute_337, permute_342, permute_347, permute_348, permute_349, permute_350, permute_353, mul_329, permute_361, permute_362, permute_363, permute_364, permute_367, permute_372, permute_373, permute_374, permute_375, permute_382, permute_386, mul_351, permute_393, permute_395, permute_398, permute_403, permute_404, permute_405, permute_406, permute_413, permute_417, mul_370, mul_373, permute_424, permute_426, permute_429, permute_434, permute_435, permute_436, permute_437, permute_444, permute_448, mul_395, permute_455, permute_457, permute_460, permute_465, permute_466, permute_467, permute_468, permute_475, permute_479, mul_414, mul_417, permute_486, permute_488, mul_435, permute_494, permute_496, permute_498, permute_503, permute_507, permute_512, permute_516, permute_521, permute_525, permute_530, permute_534, tangents_1 = args
        args.clear()
        assert_size_stride(primals_70, (512, ), (1, ))
        assert_size_stride(primals_76, (512, ), (1, ))
        assert_size_stride(primals_150, (128, ), (1, ))
        assert_size_stride(primals_151, (128, ), (1, ))
        assert_size_stride(convert_element_type_2, (128, 256), (256, 1))
        assert_size_stride(addmm, (128, 1024), (1024, 1))
        assert_size_stride(convert_element_type_7, (128, 1024), (1024, 1))
        assert_size_stride(convert_element_type_15, (128, 512), (512, 1))
        assert_size_stride(addmm_2, (128, 512), (512, 1))
        assert_size_stride(convert_element_type_20, (128, 512), (512, 1))
        assert_size_stride(addmm_3, (128, 256), (256, 1))
        assert_size_stride(addmm_4, (128, 256), (256, 1))
        assert_size_stride(convert_element_type_32, (128, 256), (256, 1))
        assert_size_stride(addmm_6, (128, 256), (256, 1))
        assert_size_stride(convert_element_type_44, (128, 256), (256, 1))
        assert_size_stride(addmm_8, (128, 256), (256, 1))
        assert_size_stride(convert_element_type_56, (128, 256), (256, 1))
        assert_size_stride(convert_element_type_63, (128, 4, 3, 3), (36, 1, 12, 4))
        assert_size_stride(convert_element_type_64, (128, 4, 32, 32), (4096, 1, 128, 4))
        assert_size_stride(convolution, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_66, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_1, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze, (128, 32), (32, 1))
        assert_size_stride(squeeze_1, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_74, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(convert_element_type_75, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(convolution_2, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_77, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_79, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_3, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze_2, (128, 32), (32, 1))
        assert_size_stride(squeeze_3, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_87, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(convert_element_type_88, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(convert_element_type_90, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_92, (256, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_5, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(bmm_4, (512, 256, 1), (256, 1, 1))
        assert_size_stride(view_27, (32768, 256), (256, 1))
        assert_size_stride(getitem_13, (128, 256, 256), (65536, 256, 1))
        assert_size_stride(add_8, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_111, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_6, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(squeeze_4, (128, 32), (32, 1))
        assert_size_stride(squeeze_5, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_119, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(convert_element_type_120, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(convolution_7, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(view_38, (32768, 256), (256, 1))
        assert_size_stride(bmm_8, (512, 256, 1), (256, 1, 1))
        assert_size_stride(view_50, (32768, 256), (256, 1))
        assert_size_stride(getitem_22, (128, 256, 256), (65536, 256, 1))
        assert_size_stride(add_13, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_141, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_8, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(squeeze_6, (128, 32), (32, 1))
        assert_size_stride(squeeze_7, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_149, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(convert_element_type_150, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(convert_element_type_152, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_154, (512, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_10, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(bmm_12, (1024, 64, 1), (64, 1, 1))
        assert_size_stride(view_73, (8192, 512), (512, 1))
        assert_size_stride(getitem_31, (128, 64, 512), (32768, 512, 1))
        assert_size_stride(add_18, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(convert_element_type_173, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(convolution_11, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(squeeze_8, (128, 32), (32, 1))
        assert_size_stride(squeeze_9, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_181, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(convert_element_type_182, (128, 512, 8, 8), (32768, 64, 8, 1))
        assert_size_stride(convolution_12, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(view_84, (8192, 512), (512, 1))
        assert_size_stride(bmm_16, (1024, 64, 1), (64, 1, 1))
        assert_size_stride(view_96, (8192, 512), (512, 1))
        assert_size_stride(getitem_40, (128, 64, 512), (32768, 512, 1))
        assert_size_stride(add_23, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(convert_element_type_203, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(convolution_13, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(squeeze_10, (128, 32), (32, 1))
        assert_size_stride(squeeze_11, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_211, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(convert_element_type_212, (128, 512, 8, 8), (32768, 64, 8, 1))
        assert_size_stride(add_27, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(getitem_47, (128, 64, 1), (64, 1, 1))
        assert_size_stride(rsqrt_6, (128, 64, 1), (64, 1, 1))
        assert_size_stride(view_107, (8192, 512), (512, 1))
        assert_size_stride(bmm_20, (1024, 64, 64), (4096, 64, 1))
        assert_size_stride(amax_default_5, (128, 8, 64, 1), (512, 64, 1, 1))
        assert_size_stride(sum_5, (128, 8, 64, 1), (512, 64, 1, 1))
        assert_size_stride(view_117, (8192, 512), (512, 1))
        assert_size_stride(add_30, (128, 64, 512), (32768, 512, 1))
        assert_size_stride(getitem_49, (128, 64, 1), (64, 1, 1))
        assert_size_stride(rsqrt_7, (128, 64, 1), (64, 1, 1))
        assert_size_stride(convert_element_type_235, (512, 256, 4, 4), (4096, 1, 1024, 256))
        assert_size_stride(convert_element_type_236, (128, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(addmm_16, (128, 256), (256, 1))
        assert_size_stride(convert_element_type_243, (128, 256), (256, 1))
        assert_size_stride(addmm_17, (128, 256), (256, 1))
        assert_size_stride(view_121, (32768, 256), (256, 1))
        assert_size_stride(bmm_22, (512, 256, 1), (256, 1, 1))
        assert_size_stride(view_133, (32768, 256), (256, 1))
        assert_size_stride(getitem_53, (128, 256, 256), (65536, 256, 1))
        assert_size_stride(add_34, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_267, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_16, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(squeeze_12, (128, 32), (32, 1))
        assert_size_stride(squeeze_13, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_275, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(convert_element_type_276, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(view_144, (32768, 256), (256, 1))
        assert_size_stride(bmm_26, (512, 256, 1), (256, 1, 1))
        assert_size_stride(view_156, (32768, 256), (256, 1))
        assert_size_stride(getitem_62, (128, 256, 256), (65536, 256, 1))
        assert_size_stride(add_39, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_297, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_18, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(squeeze_14, (128, 32), (32, 1))
        assert_size_stride(squeeze_15, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_305, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(convert_element_type_306, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(view_167, (32768, 256), (256, 1))
        assert_size_stride(bmm_30, (512, 256, 1), (256, 1, 1))
        assert_size_stride(view_179, (32768, 256), (256, 1))
        assert_size_stride(getitem_71, (128, 256, 256), (65536, 256, 1))
        assert_size_stride(add_44, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_327, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_20, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(squeeze_16, (128, 32), (32, 1))
        assert_size_stride(squeeze_17, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_335, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(convert_element_type_336, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(view_190, (32768, 256), (256, 1))
        assert_size_stride(bmm_34, (512, 256, 1), (256, 1, 1))
        assert_size_stride(view_202, (32768, 256), (256, 1))
        assert_size_stride(getitem_80, (128, 256, 256), (65536, 256, 1))
        assert_size_stride(add_49, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_357, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_22, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(squeeze_18, (128, 32), (32, 1))
        assert_size_stride(squeeze_19, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_365, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(convert_element_type_366, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(view_213, (32768, 256), (256, 1))
        assert_size_stride(bmm_38, (512, 256, 1), (256, 1, 1))
        assert_size_stride(view_225, (32768, 256), (256, 1))
        assert_size_stride(getitem_89, (128, 256, 256), (65536, 256, 1))
        assert_size_stride(add_54, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_387, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(convolution_24, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(squeeze_20, (128, 32), (32, 1))
        assert_size_stride(squeeze_21, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_395, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(convert_element_type_396, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(convert_element_type_398, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(convert_element_type_400, (256, 128, 4, 4), (2048, 1, 512, 128))
        assert_size_stride(addmm_23, (128, 256), (256, 1))
        assert_size_stride(convert_element_type_407, (128, 256), (256, 1))
        assert_size_stride(addmm_24, (128, 128), (128, 1))
        assert_size_stride(add_59, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_414, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_27, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze_22, (128, 32), (32, 1))
        assert_size_stride(squeeze_23, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_422, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(convert_element_type_423, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(convert_element_type_425, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_427, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_29, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze_24, (128, 32), (32, 1))
        assert_size_stride(squeeze_25, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_435, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(convert_element_type_436, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(convert_element_type_438, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_440, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_31, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze_26, (128, 32), (32, 1))
        assert_size_stride(squeeze_27, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_448, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(convert_element_type_449, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(convert_element_type_451, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_453, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_33, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze_28, (128, 32), (32, 1))
        assert_size_stride(squeeze_29, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_461, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(convert_element_type_462, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(convert_element_type_464, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(convert_element_type_466, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_35, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze_30, (128, 32), (32, 1))
        assert_size_stride(squeeze_31, (128, 32), (32, 1))
        assert_size_stride(convert_element_type_474, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(convert_element_type_475, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(add_79, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(getitem_121, (128, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(rsqrt_18, (128, 32, 1, 1), (32, 1, 1, 1))
        assert_size_stride(convert_element_type_480, (4, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convert_element_type_481, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(mul_99, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(permute_131, (128, 12, 128), (3200, 1, 12))
        assert_size_stride(permute_132, (128, 1024, 12), (12288, 1, 1024))
        assert_size_stride(permute_133, (128, 128, 12), (3200, 12, 1))
        assert_size_stride(permute_134, (128, 1024, 128), (131072, 1, 1024))
        assert_size_stride(mul_114, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(mul_117, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(permute_140, (128, 1024, 12), (12288, 1, 1024))
        assert_size_stride(permute_142, (128, 1024, 128), (131072, 1, 1024))
        assert_size_stride(mul_132, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(mul_135, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(permute_148, (128, 1024, 12), (12288, 1, 1024))
        assert_size_stride(permute_150, (128, 1024, 128), (131072, 1, 1024))
        assert_size_stride(mul_150, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(mul_153, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(permute_156, (128, 1024, 12), (12288, 1, 1024))
        assert_size_stride(permute_158, (128, 1024, 128), (131072, 1, 1024))
        assert_size_stride(mul_168, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(mul_171, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(permute_164, (128, 1024, 12), (12288, 1, 1024))
        assert_size_stride(permute_166, (128, 1024, 128), (131072, 1, 1024))
        assert_size_stride(permute_168, (128, 256), (256, 1))
        assert_size_stride(permute_173, (256, 256), (256, 1))
        assert_size_stride(mul_193, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(mul_196, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(permute_180, (128, 12, 256), (6400, 1, 12))
        assert_size_stride(permute_181, (128, 256, 12), (3072, 1, 256))
        assert_size_stride(permute_182, (128, 256, 12), (6400, 12, 1))
        assert_size_stride(permute_183, (128, 256, 256), (65536, 1, 256))
        assert_size_stride(permute_186, (256, 256), (256, 1))
        assert_size_stride(permute_191, (512, 1, 256), (256, 1, 1))
        assert_size_stride(permute_192, (512, 64, 1), (64, 1, 64))
        assert_size_stride(permute_193, (512, 64, 256), (16384, 1, 64))
        assert_size_stride(permute_194, (512, 1, 64), (64, 1, 1))
        assert_size_stride(permute_201, (512, 256), (256, 1))
        assert_size_stride(permute_205, (256, 256), (256, 1))
        assert_size_stride(mul_215, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(mul_218, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(permute_212, (128, 256, 12), (3072, 1, 256))
        assert_size_stride(permute_214, (128, 256, 256), (65536, 1, 256))
        assert_size_stride(permute_217, (256, 256), (256, 1))
        assert_size_stride(permute_222, (512, 1, 256), (256, 1, 1))
        assert_size_stride(permute_223, (512, 64, 1), (64, 1, 64))
        assert_size_stride(permute_224, (512, 64, 256), (16384, 1, 64))
        assert_size_stride(permute_225, (512, 1, 64), (64, 1, 1))
        assert_size_stride(permute_232, (512, 256), (256, 1))
        assert_size_stride(permute_236, (256, 256), (256, 1))
        assert_size_stride(mul_237, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(mul_240, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(permute_243, (128, 256, 12), (3072, 1, 256))
        assert_size_stride(permute_245, (128, 256, 256), (65536, 1, 256))
        assert_size_stride(permute_248, (256, 256), (256, 1))
        assert_size_stride(permute_253, (512, 1, 256), (256, 1, 1))
        assert_size_stride(permute_254, (512, 64, 1), (64, 1, 64))
        assert_size_stride(permute_255, (512, 64, 256), (16384, 1, 64))
        assert_size_stride(permute_256, (512, 1, 64), (64, 1, 1))
        assert_size_stride(permute_263, (512, 256), (256, 1))
        assert_size_stride(permute_267, (256, 256), (256, 1))
        assert_size_stride(mul_259, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(mul_262, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(permute_274, (128, 256, 12), (3072, 1, 256))
        assert_size_stride(permute_276, (128, 256, 256), (65536, 1, 256))
        assert_size_stride(permute_279, (256, 256), (256, 1))
        assert_size_stride(permute_284, (512, 1, 256), (256, 1, 1))
        assert_size_stride(permute_285, (512, 64, 1), (64, 1, 64))
        assert_size_stride(permute_286, (512, 64, 256), (16384, 1, 64))
        assert_size_stride(permute_287, (512, 1, 64), (64, 1, 1))
        assert_size_stride(permute_294, (512, 256), (256, 1))
        assert_size_stride(permute_298, (256, 256), (256, 1))
        assert_size_stride(mul_281, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(mul_284, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(permute_305, (128, 256, 12), (3072, 1, 256))
        assert_size_stride(permute_307, (128, 256, 256), (65536, 1, 256))
        assert_size_stride(permute_310, (256, 256), (256, 1))
        assert_size_stride(permute_315, (512, 1, 256), (256, 1, 1))
        assert_size_stride(permute_316, (512, 64, 1), (64, 1, 64))
        assert_size_stride(permute_317, (512, 64, 256), (16384, 1, 64))
        assert_size_stride(permute_318, (512, 1, 64), (64, 1, 1))
        assert_size_stride(permute_325, (512, 256), (256, 1))
        assert_size_stride(permute_329, (256, 256), (256, 1))
        assert_size_stride(permute_332, (256, 256), (256, 1))
        assert_size_stride(permute_337, (256, 256), (256, 1))
        assert_size_stride(permute_342, (512, 512), (512, 1))
        assert_size_stride(permute_347, (1024, 64, 64), (4096, 1, 64))
        assert_size_stride(permute_348, (1024, 64, 64), (4096, 1, 64))
        assert_size_stride(permute_349, (1024, 64, 64), (4096, 1, 64))
        assert_size_stride(permute_350, (1024, 64, 64), (4096, 1, 64))
        assert_size_stride(permute_353, (1536, 512), (512, 1))
        assert_size_stride(mul_329, (128, 512, 8, 8), (32768, 64, 8, 1))
        assert_size_stride(permute_361, (128, 12, 512), (12800, 1, 12))
        assert_size_stride(permute_362, (128, 64, 12), (768, 1, 64))
        assert_size_stride(permute_363, (128, 512, 12), (12800, 12, 1))
        assert_size_stride(permute_364, (128, 64, 512), (32768, 1, 64))
        assert_size_stride(permute_367, (512, 512), (512, 1))
        assert_size_stride(permute_372, (1024, 1, 64), (64, 1, 1))
        assert_size_stride(permute_373, (1024, 64, 1), (64, 1, 64))
        assert_size_stride(permute_374, (1024, 64, 64), (4096, 1, 64))
        assert_size_stride(permute_375, (1024, 1, 64), (64, 1, 1))
        assert_size_stride(permute_382, (1024, 256), (256, 1))
        assert_size_stride(permute_386, (512, 512), (512, 1))
        assert_size_stride(mul_351, (128, 512, 8, 8), (32768, 64, 8, 1))
        assert_size_stride(permute_393, (128, 64, 12), (768, 1, 64))
        assert_size_stride(permute_395, (128, 64, 512), (32768, 1, 64))
        assert_size_stride(permute_398, (512, 512), (512, 1))
        assert_size_stride(permute_403, (1024, 1, 64), (64, 1, 1))
        assert_size_stride(permute_404, (1024, 64, 1), (64, 1, 64))
        assert_size_stride(permute_405, (1024, 64, 64), (4096, 1, 64))
        assert_size_stride(permute_406, (1024, 1, 64), (64, 1, 1))
        assert_size_stride(permute_413, (1024, 256), (256, 1))
        assert_size_stride(permute_417, (512, 512), (512, 1))
        assert_size_stride(mul_370, (128, 256, 16, 16), (65536, 1, 4096, 256))
        assert_size_stride(mul_373, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(permute_424, (128, 256, 12), (3072, 1, 256))
        assert_size_stride(permute_426, (128, 256, 256), (65536, 1, 256))
        assert_size_stride(permute_429, (256, 256), (256, 1))
        assert_size_stride(permute_434, (512, 1, 256), (256, 1, 1))
        assert_size_stride(permute_435, (512, 64, 1), (64, 1, 64))
        assert_size_stride(permute_436, (512, 64, 256), (16384, 1, 64))
        assert_size_stride(permute_437, (512, 1, 64), (64, 1, 1))
        assert_size_stride(permute_444, (512, 256), (256, 1))
        assert_size_stride(permute_448, (256, 256), (256, 1))
        assert_size_stride(mul_395, (128, 256, 16, 16), (65536, 256, 16, 1))
        assert_size_stride(permute_455, (128, 256, 12), (3072, 1, 256))
        assert_size_stride(permute_457, (128, 256, 256), (65536, 1, 256))
        assert_size_stride(permute_460, (256, 256), (256, 1))
        assert_size_stride(permute_465, (512, 1, 256), (256, 1, 1))
        assert_size_stride(permute_466, (512, 64, 1), (64, 1, 64))
        assert_size_stride(permute_467, (512, 64, 256), (16384, 1, 64))
        assert_size_stride(permute_468, (512, 1, 64), (64, 1, 1))
        assert_size_stride(permute_475, (512, 256), (256, 1))
        assert_size_stride(permute_479, (256, 256), (256, 1))
        assert_size_stride(mul_414, (128, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(mul_417, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(permute_486, (128, 1024, 12), (12288, 1, 1024))
        assert_size_stride(permute_488, (128, 1024, 128), (131072, 1, 1024))
        assert_size_stride(mul_435, (128, 128, 32, 32), (131072, 1024, 32, 1))
        assert_size_stride(permute_494, (128, 1024, 12), (12288, 1, 1024))
        assert_size_stride(permute_496, (128, 1024, 128), (131072, 1, 1024))
        assert_size_stride(permute_498, (12800, 256), (256, 1))
        assert_size_stride(permute_503, (256, 256), (256, 1))
        assert_size_stride(permute_507, (6400, 256), (256, 1))
        assert_size_stride(permute_512, (256, 256), (256, 1))
        assert_size_stride(permute_516, (3200, 256), (256, 1))
        assert_size_stride(permute_521, (256, 256), (256, 1))
        assert_size_stride(permute_525, (256, 512), (512, 1))
        assert_size_stride(permute_530, (512, 512), (512, 1))
        assert_size_stride(permute_534, (256, 1024), (1024, 1))
        assert_size_stride(tangents_1, (128, 4, 32, 32), (4096, 1, 128, 4))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((4, 1024), (1, 4), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_0.run(tangents_1, buf0, 4096, 128, stream=stream0)
            buf6 = empty_strided_cuda((4, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_1.run(buf0, buf6, 4, 1024, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf2 = torch.ops.aten.convolution_backward.default(tangents_1, convert_element_type_481, convert_element_type_480, [4], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_480
            del convert_element_type_481
            del tangents_1
            buf3 = buf2[0]
            assert_size_stride(buf3, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf3, 16, 'torch.ops.aten.convolution_backward.default')
            buf4 = buf2[1]
            assert_size_stride(buf4, (4, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf4, 16, 'torch.ops.aten.convolution_backward.default')
            del buf2
            buf5 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf4, buf5, 4608, stream=stream0)
            del buf4
            buf7 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
            # Topologically Sorted Source Nodes: [h_22, input_42], Original ATen: [aten.silu, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_native_group_norm_silu_3.run(add_79, getitem_121, rsqrt_18, primals_150, primals_151, buf7, 16777216, stream=stream0)
            del primals_151
            buf8 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            buf9 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [h_22, input_42, input_43], Original ATen: [aten._to_copy, aten.fill, aten.silu, aten.clone, aten.sub, aten.mul, aten.add, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_fill_mul_native_group_norm_backward_silu_sub_4.run(buf3, buf7, add_79, buf8, buf9, 16384, 1024, stream=stream0)
            buf13 = empty_strided_cuda((32, 4), (4, 1), torch.float32)
            buf14 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.native_group_norm, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_group_norm_native_group_norm_backward_5.run(buf8, buf9, getitem_121, rsqrt_18, buf13, buf14, 128, 128, stream=stream0)
            buf10 = reinterpret_tensor(buf0, (128, 32), (32, 1), 0); del buf0  # reuse
            buf11 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.native_group_norm, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_native_group_norm_backward_6.run(buf9, primals_150, getitem_121, buf8, buf10, buf11, 4096, stream=stream0)
            del buf8
            del buf9
            buf15 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_22, input_42, input_43], Original ATen: [aten._to_copy, aten.fill, aten.silu, aten.clone, aten.native_group_norm, aten.sub, aten.mul, aten.add, aten.native_group_norm_backward, aten.sigmoid]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_fill_mul_native_group_norm_native_group_norm_backward_sigmoid_silu_sub_7.run(buf3, buf7, rsqrt_18, primals_150, add_79, buf11, getitem_121, buf10, buf15, 131072, 128, stream=stream0)
            del add_79
            del buf10
            del buf7
            del getitem_121
            del primals_150
            del rsqrt_18
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf17 = torch.ops.aten.convolution_backward.default(buf15, convert_element_type_475, convert_element_type_474, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_474
            del convert_element_type_475
            buf18 = buf17[0]
            assert_size_stride(buf18, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf18, 16, 'torch.ops.aten.convolution_backward.default')
            buf19 = buf17[1]
            assert_size_stride(buf19, (128, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf19, 16, 'torch.ops.aten.convolution_backward.default')
            del buf17
            buf21 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf15, buf21, 128, 131072, stream=stream0)
            buf20 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf19, buf20, 16384, stream=stream0)
            del buf19
            buf22 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 16384, 16384), torch.float32)
            buf23 = reinterpret_tensor(buf3, (128, 128, 1024), (131072, 1024, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_10.run(buf18, mul_99, buf22, buf23, 16384, 1024, stream=stream0)
            buf25 = empty_strided_cuda((128, 128, 12), (1536, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf23, permute_132, out=buf25)
            del permute_132
            buf24 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_131, buf23, out=buf24)
            buf27 = empty_strided_cuda((128, 12, 128), (1536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf24, permute_134, out=buf27)
            del permute_134
            buf26 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf24, out=buf26)
            buf28 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            buf29 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_16], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11.run(buf18, mul_99, buf26, convolution_35, buf28, buf29, 16384, 1024, stream=stream0)
            buf30 = buf11; del buf11  # reuse
            buf31 = squeeze_30; del squeeze_30  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_backward_12.run(buf31, buf29, buf28, squeeze_31, buf30, 4096, stream=stream0)
            buf32 = reinterpret_tensor(buf26, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [x_norm_16], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13.run(buf32, buf18, mul_99, squeeze_31, convolution_35, buf30, buf31, 16384, 1024, stream=stream0)
            del buf18
            del buf30
            del convolution_35
            del mul_99
            del squeeze_31
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf34 = torch.ops.aten.convolution_backward.default(buf32, convert_element_type_464, convert_element_type_466, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_464
            del convert_element_type_466
            buf38 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf32, buf38, 128, 131072, stream=stream0)
            del buf32
            buf35 = buf34[0]
            assert_size_stride(buf35, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf35, 16, 'torch.ops.aten.convolution_backward.default')
            buf39 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf39, buf35, mul_114, 16384, 1024, stream=stream0)
            del mul_114
            buf36 = buf34[1]
            assert_size_stride(buf36, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf36, 16, 'torch.ops.aten.convolution_backward.default')
            del buf34
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf41 = torch.ops.aten.convolution_backward.default(buf39, convert_element_type_462, convert_element_type_461, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_461
            del convert_element_type_462
            buf42 = buf41[0]
            assert_size_stride(buf42, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf42, 16, 'torch.ops.aten.convolution_backward.default')
            buf43 = buf41[1]
            assert_size_stride(buf43, (128, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf43, 16, 'torch.ops.aten.convolution_backward.default')
            del buf41
            buf45 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf39, buf45, 128, 131072, stream=stream0)
            buf44 = reinterpret_tensor(buf29, (128, 128, 1, 1), (128, 1, 128, 128), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf43, buf44, 16384, stream=stream0)
            del buf43
            buf37 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(buf36, buf37, 147456, stream=stream0)
            del buf36
            buf46 = reinterpret_tensor(buf28, (128, 128, 1, 1), (128, 1, 16384, 16384), 0); del buf28  # reuse
            buf47 = reinterpret_tensor(buf35, (128, 128, 1024), (131072, 1024, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_10.run(buf42, mul_117, buf46, buf47, 16384, 1024, stream=stream0)
            buf49 = empty_strided_cuda((128, 128, 12), (1536, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf47, permute_140, out=buf49)
            del permute_140
            buf48 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_131, buf47, out=buf48)
            buf51 = empty_strided_cuda((128, 12, 128), (1536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf48, permute_142, out=buf51)
            del permute_142
            buf50 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf48, out=buf50)
            buf53 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            buf54 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_15], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11.run(buf42, mul_117, buf50, convolution_33, buf53, buf54, 16384, 1024, stream=stream0)
            buf55 = buf31; del buf31  # reuse
            buf56 = squeeze_28; del squeeze_28  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_backward_12.run(buf56, buf54, buf53, squeeze_29, buf55, 4096, stream=stream0)
            buf57 = reinterpret_tensor(buf50, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [x_norm_15], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13.run(buf57, buf42, mul_117, squeeze_29, convolution_33, buf55, buf56, 16384, 1024, stream=stream0)
            del buf42
            del buf55
            del convolution_33
            del mul_117
            del squeeze_29
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf59 = torch.ops.aten.convolution_backward.default(buf57, convert_element_type_451, convert_element_type_453, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_451
            del convert_element_type_453
            buf63 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf57, buf63, 128, 131072, stream=stream0)
            del buf57
            buf60 = buf59[0]
            assert_size_stride(buf60, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf60, 16, 'torch.ops.aten.convolution_backward.default')
            buf64 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf64, buf60, mul_132, 16384, 1024, stream=stream0)
            del mul_132
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf66 = torch.ops.aten.convolution_backward.default(buf64, convert_element_type_449, convert_element_type_448, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_448
            del convert_element_type_449
            buf67 = buf66[0]
            assert_size_stride(buf67, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf67, 16, 'torch.ops.aten.convolution_backward.default')
            buf61 = buf59[1]
            assert_size_stride(buf61, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf61, 16, 'torch.ops.aten.convolution_backward.default')
            del buf59
            buf68 = buf66[1]
            assert_size_stride(buf68, (128, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf68, 16, 'torch.ops.aten.convolution_backward.default')
            del buf66
            buf70 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf64, buf70, 128, 131072, stream=stream0)
            buf69 = reinterpret_tensor(buf54, (128, 128, 1, 1), (128, 1, 128, 128), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf68, buf69, 16384, stream=stream0)
            del buf68
            buf62 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(buf61, buf62, 147456, stream=stream0)
            del buf61
            buf71 = reinterpret_tensor(buf53, (128, 128, 1, 1), (128, 1, 16384, 16384), 0); del buf53  # reuse
            buf72 = reinterpret_tensor(buf60, (128, 128, 1024), (131072, 1024, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_10.run(buf67, mul_135, buf71, buf72, 16384, 1024, stream=stream0)
            buf74 = empty_strided_cuda((128, 128, 12), (1536, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf72, permute_148, out=buf74)
            del permute_148
            buf73 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_131, buf72, out=buf73)
            buf76 = empty_strided_cuda((128, 12, 128), (1536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf73, permute_150, out=buf76)
            del permute_150
            buf75 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf73, out=buf75)
            buf77 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            buf78 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_14], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11.run(buf67, mul_135, buf75, convolution_31, buf77, buf78, 16384, 1024, stream=stream0)
            buf79 = buf56; del buf56  # reuse
            buf80 = squeeze_26; del squeeze_26  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_backward_12.run(buf80, buf78, buf77, squeeze_27, buf79, 4096, stream=stream0)
            buf81 = reinterpret_tensor(buf75, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [x_norm_14], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13.run(buf81, buf67, mul_135, squeeze_27, convolution_31, buf79, buf80, 16384, 1024, stream=stream0)
            del buf67
            del buf79
            del convolution_31
            del mul_135
            del squeeze_27
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf83 = torch.ops.aten.convolution_backward.default(buf81, convert_element_type_438, convert_element_type_440, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_438
            del convert_element_type_440
            buf87 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf81, buf87, 128, 131072, stream=stream0)
            del buf81
            buf84 = buf83[0]
            assert_size_stride(buf84, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf84, 16, 'torch.ops.aten.convolution_backward.default')
            buf88 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf88, buf84, mul_150, 16384, 1024, stream=stream0)
            del mul_150
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf90 = torch.ops.aten.convolution_backward.default(buf88, convert_element_type_436, convert_element_type_435, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_435
            del convert_element_type_436
            buf91 = buf90[0]
            assert_size_stride(buf91, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf91, 16, 'torch.ops.aten.convolution_backward.default')
            buf85 = buf83[1]
            assert_size_stride(buf85, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf85, 16, 'torch.ops.aten.convolution_backward.default')
            del buf83
            buf92 = buf90[1]
            assert_size_stride(buf92, (128, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf92, 16, 'torch.ops.aten.convolution_backward.default')
            del buf90
            buf94 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf88, buf94, 128, 131072, stream=stream0)
            buf93 = reinterpret_tensor(buf78, (128, 128, 1, 1), (128, 1, 128, 128), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf92, buf93, 16384, stream=stream0)
            del buf92
            buf86 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(buf85, buf86, 147456, stream=stream0)
            del buf85
            buf95 = reinterpret_tensor(buf77, (128, 128, 1, 1), (128, 1, 16384, 16384), 0); del buf77  # reuse
            buf96 = reinterpret_tensor(buf84, (128, 128, 1024), (131072, 1024, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_10.run(buf91, mul_153, buf95, buf96, 16384, 1024, stream=stream0)
            buf98 = empty_strided_cuda((128, 128, 12), (1536, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf96, permute_156, out=buf98)
            del permute_156
            buf97 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_131, buf96, out=buf97)
            buf100 = empty_strided_cuda((128, 12, 128), (1536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf97, permute_158, out=buf100)
            del permute_158
            buf99 = buf96; del buf96  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf97, out=buf99)
            buf102 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            buf103 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_13], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11.run(buf91, mul_153, buf99, convolution_29, buf102, buf103, 16384, 1024, stream=stream0)
            buf104 = buf80; del buf80  # reuse
            buf105 = squeeze_24; del squeeze_24  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_backward_12.run(buf105, buf103, buf102, squeeze_25, buf104, 4096, stream=stream0)
            buf106 = reinterpret_tensor(buf99, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [x_norm_13], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13.run(buf106, buf91, mul_153, squeeze_25, convolution_29, buf104, buf105, 16384, 1024, stream=stream0)
            del buf104
            del buf91
            del convolution_29
            del mul_153
            del squeeze_25
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf108 = torch.ops.aten.convolution_backward.default(buf106, convert_element_type_425, convert_element_type_427, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_425
            del convert_element_type_427
            buf112 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf106, buf112, 128, 131072, stream=stream0)
            del buf106
            buf109 = buf108[0]
            assert_size_stride(buf109, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf109, 16, 'torch.ops.aten.convolution_backward.default')
            buf113 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_14.run(buf113, buf109, mul_168, 16384, 1024, stream=stream0)
            del mul_168
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf115 = torch.ops.aten.convolution_backward.default(buf113, convert_element_type_423, convert_element_type_422, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_422
            del convert_element_type_423
            buf116 = buf115[0]
            assert_size_stride(buf116, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf116, 16, 'torch.ops.aten.convolution_backward.default')
            buf110 = buf108[1]
            assert_size_stride(buf110, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf110, 16, 'torch.ops.aten.convolution_backward.default')
            del buf108
            buf117 = buf115[1]
            assert_size_stride(buf117, (128, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf117, 16, 'torch.ops.aten.convolution_backward.default')
            del buf115
            buf119 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf113, buf119, 128, 131072, stream=stream0)
            buf118 = reinterpret_tensor(buf103, (128, 128, 1, 1), (128, 1, 128, 128), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf117, buf118, 16384, stream=stream0)
            buf111 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(buf110, buf111, 147456, stream=stream0)
            del buf110
            buf120 = reinterpret_tensor(buf102, (128, 128, 1, 1), (128, 1, 16384, 16384), 0); del buf102  # reuse
            buf121 = reinterpret_tensor(buf109, (128, 128, 1024), (131072, 1024, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_10.run(buf116, mul_171, buf120, buf121, 16384, 1024, stream=stream0)
            buf123 = empty_strided_cuda((128, 128, 12), (1536, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf121, permute_164, out=buf123)
            del permute_164
            buf122 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_131, buf121, out=buf122)
            buf125 = empty_strided_cuda((128, 12, 128), (1536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf122, permute_166, out=buf125)
            del permute_166
            buf124 = buf121; del buf121  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf122, out=buf124)
            buf126 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            buf127 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_12], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11.run(buf116, mul_171, buf124, convolution_27, buf126, buf127, 16384, 1024, stream=stream0)
            buf128 = buf105; del buf105  # reuse
            buf129 = squeeze_22; del squeeze_22  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_backward_12.run(buf129, buf127, buf126, squeeze_23, buf128, 4096, stream=stream0)
            buf130 = reinterpret_tensor(buf124, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [x_norm_12], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13.run(buf130, buf116, mul_171, squeeze_23, convolution_27, buf128, buf129, 16384, 1024, stream=stream0)
            del buf116
            del convolution_27
            del mul_171
            del squeeze_23
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf132 = torch.ops.aten.convolution_backward.default(buf130, add_59, convert_element_type_414, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_59
            del convert_element_type_414
            buf137 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf130, buf137, 128, 131072, stream=stream0)
            del buf130
            buf133 = buf132[0]
            assert_size_stride(buf133, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf133, 16, 'torch.ops.aten.convolution_backward.default')
            buf135 = buf113; del buf113  # reuse
            buf139 = reinterpret_tensor(buf117, (128, 128), (128, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sigmoid_sigmoid_backward_squeeze_sum_16.run(buf135, buf133, convert_element_type_90, addmm_24, buf139, 16384, 1024, stream=stream0)
            del buf133
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf150 = torch.ops.aten.convolution_backward.default(buf135, convert_element_type_398, convert_element_type_400, [128], [2, 2], [1, 1], [1, 1], True, [0, 0], 1, [True, True, False])
            del convert_element_type_398
            del convert_element_type_400
            buf151 = buf150[0]
            assert_size_stride(buf151, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf151, 16, 'torch.ops.aten.convolution_backward.default')
            buf155 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_17.run(buf155, mul_193, 8388608, stream=stream0)
            del mul_193
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf158 = torch.ops.aten.convolution_backward.default(buf155, convert_element_type_396, convert_element_type_395, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_395
            del convert_element_type_396
            buf159 = buf158[0]
            assert_size_stride(buf159, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf159, 16, 'torch.ops.aten.convolution_backward.default')
            buf134 = buf132[1]
            assert_size_stride(buf134, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf134, 16, 'torch.ops.aten.convolution_backward.default')
            del buf132
            buf140 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf139, permute_168, out=buf140)
            del permute_168
            buf144 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.fill, aten.sub, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf144, addmm_23, 32768, stream=stream0)
            del addmm_23
            buf145 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf144, permute_173, out=buf145)
            del permute_173
            buf141 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (128, 128), (1, 128), 0), convert_element_type_407, out=buf141)
            del convert_element_type_407
            buf142 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_19.run(buf139, buf142, 128, 128, stream=stream0)
            del buf139
            buf152 = buf150[1]
            assert_size_stride(buf152, (256, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf152, 16, 'torch.ops.aten.convolution_backward.default')
            del buf150
            buf160 = buf158[1]
            assert_size_stride(buf160, (256, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf160, 16, 'torch.ops.aten.convolution_backward.default')
            del buf158
            buf154 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf135, buf154, 128, 131072, stream=stream0)
            buf147 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf144, buf147, 256, 128, stream=stream0)
            buf143 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(buf141, buf143, 32768, stream=stream0)
            buf146 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf144, (256, 128), (1, 256), 0), addmm_3, out=buf146)
            buf148 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf146, buf148, 65536, stream=stream0)
            buf156 = empty_strided_cuda((256, 128), (1, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf155, buf156, 32768, 256, stream=stream0)
            buf162 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf156, buf162, 256, 128, stream=stream0)
            buf161 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf160, buf161, 65536, stream=stream0)
            buf136 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(buf134, buf136, 147456, stream=stream0)
            del buf134
            buf153 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(buf152, buf153, 524288, stream=stream0)
            del buf152
            buf163 = reinterpret_tensor(buf156, (128, 256, 1, 1), (256, 1, 32768, 32768), 0); del buf156  # reuse
            buf164 = empty_strided_cuda((128, 256, 256), (65536, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_26.run(buf159, mul_196, buf163, buf164, 32768, 256, stream=stream0)
            buf166 = empty_strided_cuda((128, 256, 12), (3072, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf164, permute_181, out=buf166)
            del permute_181
            buf165 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_180, buf164, out=buf165)
            buf168 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf165, permute_183, out=buf168)
            del permute_183
            buf167 = buf164; del buf164  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf165, out=buf167)
            buf169 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            buf170 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_11], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27.run(buf159, mul_196, buf167, convolution_24, buf169, buf170, 32768, 256, stream=stream0)
            buf171 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf169, buf171, 4096, 8, stream=stream0)
            buf172 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf170, buf172, 4096, 8, stream=stream0)
            buf174 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_11], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29.run(buf159, mul_196, buf167, squeeze_21, convolution_24, buf172, squeeze_20, buf171, buf174, 8388608, stream=stream0)
            del buf159
            del convolution_24
            del mul_196
            del squeeze_20
            del squeeze_21
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf176 = torch.ops.aten.convolution_backward.default(buf174, add_54, convert_element_type_387, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_54
            del convert_element_type_387
            buf180 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_30.run(buf174, buf180, 256, 32768, stream=stream0)
            buf177 = buf176[0]
            assert_size_stride(buf177, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf177, 16, 'torch.ops.aten.convolution_backward.default')
            buf178 = buf176[1]
            assert_size_stride(buf178, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf178, 16, 'torch.ops.aten.convolution_backward.default')
            del buf176
            buf179 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(buf178, buf179, 589824, stream=stream0)
            del buf178
            buf181 = reinterpret_tensor(buf174, (128, 256, 256), (65536, 256, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_32.run(buf177, getitem_89, buf181, 8388608, stream=stream0)
            del getitem_89
            buf183 = reinterpret_tensor(buf160, (256, 256), (256, 1), 0); del buf160  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf181, (256, 32768), (1, 256), 0), view_225, out=buf183)
            del view_225
            buf184 = reinterpret_tensor(buf170, (1, 256, 128), (32768, 1, 256), 0); del buf170  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_33.run(buf181, buf184, 32768, 256, stream=stream0)
            buf182 = reinterpret_tensor(buf167, (32768, 256), (256, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf181, (32768, 256), (256, 1), 0), permute_186, out=buf182)
            del permute_186
            buf185 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_34.run(buf184, buf185, 256, 128, stream=stream0)
            buf187 = reinterpret_tensor(buf181, (128, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_35.run(buf182, buf187, 8388608, stream=stream0)
            buf188 = reinterpret_tensor(buf144, (512, 1, 64), (64, 64, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_191, reinterpret_tensor(buf187, (512, 256, 64), (16384, 64, 1), 0), out=buf188)
            del permute_191
            buf189 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf187, (512, 256, 64), (16384, 64, 1), 0), permute_192, out=buf189)
            del permute_192
            buf190 = reinterpret_tensor(bmm_38, (128, 4, 256, 1), (1024, 256, 1, 1), 0); del bmm_38  # reuse
            # Topologically Sorted Source Nodes: [matmul_18, attn_19], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36.run(buf190, buf189, 131072, stream=stream0)
            buf191 = reinterpret_tensor(buf141, (512, 64, 1), (64, 1, 1), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [matmul_18, attn_19], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_193, reinterpret_tensor(buf190, (512, 256, 1), (256, 1, 0), 0), out=buf191)
            del permute_193
            buf193 = reinterpret_tensor(buf146, (128, 1, 512), (512, 512, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_37.run(buf191, buf188, buf193, 65536, stream=stream0)
            buf195 = reinterpret_tensor(buf191, (128, 256), (256, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf145, reinterpret_tensor(buf193, (128, 512), (512, 1), 0), permute_201, alpha=1, beta=1, out=buf195)
            del permute_201
            buf186 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf183, buf186, 65536, stream=stream0)
            del buf183
            buf194 = reinterpret_tensor(buf189, (512, 256), (256, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf193, (512, 128), (1, 512), 0), addmm_3, out=buf194)
            buf196 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf194, buf196, 131072, stream=stream0)
            del buf194
            buf192 = reinterpret_tensor(buf187, (512, 256, 64), (16384, 64, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [matmul_18, attn_19], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf190, (512, 256, 1), (256, 1, 0), 0), permute_194, out=buf192)
            del permute_194
            buf197 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_39.run(buf192, buf197, 8388608, stream=stream0)
            buf198 = reinterpret_tensor(buf193, (256, 256), (256, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (256, 32768), (1, 256), 0), view_213, out=buf198)
            del view_213
            buf199 = reinterpret_tensor(buf192, (32768, 256), (256, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf197, permute_205, out=buf199)
            del buf197
            del permute_205
            buf201 = buf155; del buf155  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_transpose_view_40.run(buf201, buf177, buf199, mul_215, 8388608, stream=stream0)
            del mul_215
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf204 = torch.ops.aten.convolution_backward.default(buf201, convert_element_type_366, convert_element_type_365, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_365
            del convert_element_type_366
            buf205 = buf204[0]
            assert_size_stride(buf205, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf205, 16, 'torch.ops.aten.convolution_backward.default')
            buf206 = buf204[1]
            assert_size_stride(buf206, (256, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf206, 16, 'torch.ops.aten.convolution_backward.default')
            del buf204
            buf200 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf198, buf200, 65536, stream=stream0)
            buf202 = reinterpret_tensor(buf184, (256, 128), (1, 256), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf201, buf202, 32768, 256, stream=stream0)
            buf208 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf202, buf208, 256, 128, stream=stream0)
            buf207 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf206, buf207, 65536, stream=stream0)
            buf209 = reinterpret_tensor(buf202, (128, 256, 1, 1), (256, 1, 32768, 32768), 0); del buf202  # reuse
            buf210 = reinterpret_tensor(buf199, (128, 256, 256), (65536, 256, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_26.run(buf205, mul_218, buf209, buf210, 32768, 256, stream=stream0)
            buf212 = reinterpret_tensor(buf165, (128, 256, 12), (3072, 12, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf210, permute_212, out=buf212)
            del permute_212
            buf211 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_180, buf210, out=buf211)
            buf214 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf211, permute_214, out=buf214)
            del permute_214
            buf213 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf211, out=buf213)
            buf216 = buf169; del buf169  # reuse
            buf217 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_10], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27.run(buf205, mul_218, buf213, convolution_22, buf216, buf217, 32768, 256, stream=stream0)
            buf218 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf216, buf218, 4096, 8, stream=stream0)
            buf219 = buf171; del buf171  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf217, buf219, 4096, 8, stream=stream0)
            buf221 = reinterpret_tensor(buf177, (128, 256, 16, 16), (65536, 256, 16, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [x_norm_10], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29.run(buf205, mul_218, buf213, squeeze_19, convolution_22, buf219, squeeze_18, buf218, buf221, 8388608, stream=stream0)
            del buf205
            del convolution_22
            del mul_218
            del squeeze_18
            del squeeze_19
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf223 = torch.ops.aten.convolution_backward.default(buf221, add_49, convert_element_type_357, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_49
            del convert_element_type_357
            buf227 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_30.run(buf221, buf227, 256, 32768, stream=stream0)
            buf224 = buf223[0]
            assert_size_stride(buf224, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf224, 16, 'torch.ops.aten.convolution_backward.default')
            buf225 = buf223[1]
            assert_size_stride(buf225, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf225, 16, 'torch.ops.aten.convolution_backward.default')
            del buf223
            buf226 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(buf225, buf226, 589824, stream=stream0)
            del buf225
            buf228 = reinterpret_tensor(buf221, (128, 256, 256), (65536, 256, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_32.run(buf224, getitem_80, buf228, 8388608, stream=stream0)
            del getitem_80
            buf230 = reinterpret_tensor(buf206, (256, 256), (256, 1), 0); del buf206  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (256, 32768), (1, 256), 0), view_202, out=buf230)
            del view_202
            buf231 = reinterpret_tensor(buf217, (1, 256, 128), (32768, 1, 256), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_33.run(buf228, buf231, 32768, 256, stream=stream0)
            buf229 = reinterpret_tensor(buf213, (32768, 256), (256, 1), 0); del buf213  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf228, (32768, 256), (256, 1), 0), permute_217, out=buf229)
            del permute_217
            buf232 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_34.run(buf231, buf232, 256, 128, stream=stream0)
            buf234 = reinterpret_tensor(buf228, (128, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_35.run(buf229, buf234, 8388608, stream=stream0)
            buf235 = reinterpret_tensor(buf145, (512, 1, 64), (64, 64, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_222, reinterpret_tensor(buf234, (512, 256, 64), (16384, 64, 1), 0), out=buf235)
            del permute_222
            buf236 = reinterpret_tensor(buf190, (512, 256, 1), (256, 1, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf234, (512, 256, 64), (16384, 64, 1), 0), permute_223, out=buf236)
            del permute_223
            buf237 = reinterpret_tensor(bmm_34, (128, 4, 256, 1), (1024, 256, 1, 1), 0); del bmm_34  # reuse
            # Topologically Sorted Source Nodes: [matmul_16, attn_17], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36.run(buf237, buf236, 131072, stream=stream0)
            buf238 = reinterpret_tensor(buf188, (512, 64, 1), (64, 1, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [matmul_16, attn_17], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_224, reinterpret_tensor(buf237, (512, 256, 1), (256, 1, 0), 0), out=buf238)
            del permute_224
            buf240 = reinterpret_tensor(buf198, (128, 1, 512), (512, 512, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_37.run(buf238, buf235, buf240, 65536, stream=stream0)
            buf242 = reinterpret_tensor(buf238, (128, 256), (256, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf195, reinterpret_tensor(buf240, (128, 512), (512, 1), 0), permute_232, alpha=1, beta=1, out=buf242)
            del permute_232
            buf233 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf230, buf233, 65536, stream=stream0)
            del buf230
            buf241 = reinterpret_tensor(buf236, (512, 256), (256, 1), 0); del buf236  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (512, 128), (1, 512), 0), addmm_3, out=buf241)
            buf243 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf241, buf243, 131072, stream=stream0)
            del buf241
            buf239 = reinterpret_tensor(buf234, (512, 256, 64), (16384, 64, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [matmul_16, attn_17], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf237, (512, 256, 1), (256, 1, 0), 0), permute_225, out=buf239)
            del permute_225
            buf244 = buf229; del buf229  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_39.run(buf239, buf244, 8388608, stream=stream0)
            buf245 = reinterpret_tensor(buf240, (256, 256), (256, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf244, (256, 32768), (1, 256), 0), view_190, out=buf245)
            del view_190
            buf246 = reinterpret_tensor(buf239, (32768, 256), (256, 1), 0); del buf239  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf244, permute_236, out=buf246)
            del buf244
            del permute_236
            buf248 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_transpose_view_40.run(buf248, buf224, buf246, mul_237, 8388608, stream=stream0)
            del mul_237
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf251 = torch.ops.aten.convolution_backward.default(buf248, convert_element_type_336, convert_element_type_335, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_335
            del convert_element_type_336
            buf252 = buf251[0]
            assert_size_stride(buf252, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf252, 16, 'torch.ops.aten.convolution_backward.default')
            buf253 = buf251[1]
            assert_size_stride(buf253, (256, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf253, 16, 'torch.ops.aten.convolution_backward.default')
            del buf251
            buf247 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf245, buf247, 65536, stream=stream0)
            buf249 = reinterpret_tensor(buf231, (256, 128), (1, 256), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf248, buf249, 32768, 256, stream=stream0)
            buf255 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf249, buf255, 256, 128, stream=stream0)
            buf254 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf253, buf254, 65536, stream=stream0)
            buf256 = reinterpret_tensor(buf249, (128, 256, 1, 1), (256, 1, 32768, 32768), 0); del buf249  # reuse
            buf257 = reinterpret_tensor(buf246, (128, 256, 256), (65536, 256, 1), 0); del buf246  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_26.run(buf252, mul_240, buf256, buf257, 32768, 256, stream=stream0)
            buf259 = reinterpret_tensor(buf211, (128, 256, 12), (3072, 12, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf257, permute_243, out=buf259)
            del permute_243
            buf258 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_180, buf257, out=buf258)
            buf261 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf258, permute_245, out=buf261)
            del permute_245
            buf260 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf258, out=buf260)
            buf262 = buf216; del buf216  # reuse
            buf263 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_9], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27.run(buf252, mul_240, buf260, convolution_20, buf262, buf263, 32768, 256, stream=stream0)
            buf264 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf262, buf264, 4096, 8, stream=stream0)
            buf265 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf263, buf265, 4096, 8, stream=stream0)
            buf267 = reinterpret_tensor(buf224, (128, 256, 16, 16), (65536, 256, 16, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [x_norm_9], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29.run(buf252, mul_240, buf260, squeeze_17, convolution_20, buf265, squeeze_16, buf264, buf267, 8388608, stream=stream0)
            del buf252
            del convolution_20
            del mul_240
            del squeeze_16
            del squeeze_17
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf269 = torch.ops.aten.convolution_backward.default(buf267, add_44, convert_element_type_327, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_44
            del convert_element_type_327
            buf273 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_30.run(buf267, buf273, 256, 32768, stream=stream0)
            buf270 = buf269[0]
            assert_size_stride(buf270, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf270, 16, 'torch.ops.aten.convolution_backward.default')
            buf271 = buf269[1]
            assert_size_stride(buf271, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf271, 16, 'torch.ops.aten.convolution_backward.default')
            del buf269
            buf272 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(buf271, buf272, 589824, stream=stream0)
            del buf271
            buf274 = reinterpret_tensor(buf267, (128, 256, 256), (65536, 256, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_32.run(buf270, getitem_71, buf274, 8388608, stream=stream0)
            del getitem_71
            buf276 = reinterpret_tensor(buf253, (256, 256), (256, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf274, (256, 32768), (1, 256), 0), view_179, out=buf276)
            del view_179
            buf277 = reinterpret_tensor(buf263, (1, 256, 128), (32768, 1, 256), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_33.run(buf274, buf277, 32768, 256, stream=stream0)
            buf275 = reinterpret_tensor(buf260, (32768, 256), (256, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf274, (32768, 256), (256, 1), 0), permute_248, out=buf275)
            del permute_248
            buf278 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_34.run(buf277, buf278, 256, 128, stream=stream0)
            buf280 = reinterpret_tensor(buf274, (128, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf274  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_35.run(buf275, buf280, 8388608, stream=stream0)
            buf281 = reinterpret_tensor(buf195, (512, 1, 64), (64, 64, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_253, reinterpret_tensor(buf280, (512, 256, 64), (16384, 64, 1), 0), out=buf281)
            del permute_253
            buf282 = reinterpret_tensor(buf237, (512, 256, 1), (256, 1, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf280, (512, 256, 64), (16384, 64, 1), 0), permute_254, out=buf282)
            del permute_254
            buf283 = reinterpret_tensor(bmm_30, (128, 4, 256, 1), (1024, 256, 1, 1), 0); del bmm_30  # reuse
            # Topologically Sorted Source Nodes: [matmul_14, attn_15], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36.run(buf283, buf282, 131072, stream=stream0)
            buf284 = reinterpret_tensor(buf235, (512, 64, 1), (64, 1, 1), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [matmul_14, attn_15], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_255, reinterpret_tensor(buf283, (512, 256, 1), (256, 1, 0), 0), out=buf284)
            del permute_255
            buf286 = reinterpret_tensor(buf245, (128, 1, 512), (512, 512, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_37.run(buf284, buf281, buf286, 65536, stream=stream0)
            buf288 = reinterpret_tensor(buf284, (128, 256), (256, 1), 0); del buf284  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf242, reinterpret_tensor(buf286, (128, 512), (512, 1), 0), permute_263, alpha=1, beta=1, out=buf288)
            del permute_263
            buf279 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf276, buf279, 65536, stream=stream0)
            del buf276
            buf287 = reinterpret_tensor(buf282, (512, 256), (256, 1), 0); del buf282  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf286, (512, 128), (1, 512), 0), addmm_3, out=buf287)
            buf289 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf287, buf289, 131072, stream=stream0)
            del buf287
            buf285 = reinterpret_tensor(buf280, (512, 256, 64), (16384, 64, 1), 0); del buf280  # reuse
            # Topologically Sorted Source Nodes: [matmul_14, attn_15], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf283, (512, 256, 1), (256, 1, 0), 0), permute_256, out=buf285)
            del permute_256
            buf290 = buf275; del buf275  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_39.run(buf285, buf290, 8388608, stream=stream0)
            buf291 = reinterpret_tensor(buf286, (256, 256), (256, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf290, (256, 32768), (1, 256), 0), view_167, out=buf291)
            del view_167
            buf292 = reinterpret_tensor(buf285, (32768, 256), (256, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf290, permute_267, out=buf292)
            del buf290
            del permute_267
            buf294 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_transpose_view_40.run(buf294, buf270, buf292, mul_259, 8388608, stream=stream0)
            del mul_259
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf297 = torch.ops.aten.convolution_backward.default(buf294, convert_element_type_306, convert_element_type_305, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_305
            del convert_element_type_306
            buf298 = buf297[0]
            assert_size_stride(buf298, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf298, 16, 'torch.ops.aten.convolution_backward.default')
            buf299 = buf297[1]
            assert_size_stride(buf299, (256, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf299, 16, 'torch.ops.aten.convolution_backward.default')
            del buf297
            buf293 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf291, buf293, 65536, stream=stream0)
            buf295 = reinterpret_tensor(buf277, (256, 128), (1, 256), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf294, buf295, 32768, 256, stream=stream0)
            buf301 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf295, buf301, 256, 128, stream=stream0)
            buf300 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf299, buf300, 65536, stream=stream0)
            buf302 = reinterpret_tensor(buf295, (128, 256, 1, 1), (256, 1, 32768, 32768), 0); del buf295  # reuse
            buf303 = reinterpret_tensor(buf292, (128, 256, 256), (65536, 256, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_26.run(buf298, mul_262, buf302, buf303, 32768, 256, stream=stream0)
            buf305 = reinterpret_tensor(buf258, (128, 256, 12), (3072, 12, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf303, permute_274, out=buf305)
            del permute_274
            buf304 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_180, buf303, out=buf304)
            buf307 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf304, permute_276, out=buf307)
            del permute_276
            buf306 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf304, out=buf306)
            buf309 = buf262; del buf262  # reuse
            buf310 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_8], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27.run(buf298, mul_262, buf306, convolution_18, buf309, buf310, 32768, 256, stream=stream0)
            buf311 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf309, buf311, 4096, 8, stream=stream0)
            buf312 = buf264; del buf264  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf310, buf312, 4096, 8, stream=stream0)
            buf314 = reinterpret_tensor(buf270, (128, 256, 16, 16), (65536, 256, 16, 1), 0); del buf270  # reuse
            # Topologically Sorted Source Nodes: [x_norm_8], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29.run(buf298, mul_262, buf306, squeeze_15, convolution_18, buf312, squeeze_14, buf311, buf314, 8388608, stream=stream0)
            del buf298
            del convolution_18
            del mul_262
            del squeeze_14
            del squeeze_15
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf316 = torch.ops.aten.convolution_backward.default(buf314, add_39, convert_element_type_297, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_39
            del convert_element_type_297
            buf320 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_30.run(buf314, buf320, 256, 32768, stream=stream0)
            buf317 = buf316[0]
            assert_size_stride(buf317, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf317, 16, 'torch.ops.aten.convolution_backward.default')
            buf318 = buf316[1]
            assert_size_stride(buf318, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf318, 16, 'torch.ops.aten.convolution_backward.default')
            del buf316
            buf319 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(buf318, buf319, 589824, stream=stream0)
            del buf318
            buf321 = reinterpret_tensor(buf314, (128, 256, 256), (65536, 256, 1), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_32.run(buf317, getitem_62, buf321, 8388608, stream=stream0)
            del getitem_62
            buf323 = reinterpret_tensor(buf299, (256, 256), (256, 1), 0); del buf299  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf321, (256, 32768), (1, 256), 0), view_156, out=buf323)
            del view_156
            buf324 = reinterpret_tensor(buf310, (1, 256, 128), (32768, 1, 256), 0); del buf310  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_33.run(buf321, buf324, 32768, 256, stream=stream0)
            buf322 = reinterpret_tensor(buf306, (32768, 256), (256, 1), 0); del buf306  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf321, (32768, 256), (256, 1), 0), permute_279, out=buf322)
            del permute_279
            buf325 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_34.run(buf324, buf325, 256, 128, stream=stream0)
            buf327 = reinterpret_tensor(buf321, (128, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf321  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_35.run(buf322, buf327, 8388608, stream=stream0)
            buf328 = reinterpret_tensor(buf242, (512, 1, 64), (64, 64, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_284, reinterpret_tensor(buf327, (512, 256, 64), (16384, 64, 1), 0), out=buf328)
            del permute_284
            buf329 = reinterpret_tensor(buf283, (512, 256, 1), (256, 1, 1), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf327, (512, 256, 64), (16384, 64, 1), 0), permute_285, out=buf329)
            del permute_285
            buf330 = reinterpret_tensor(bmm_26, (128, 4, 256, 1), (1024, 256, 1, 1), 0); del bmm_26  # reuse
            # Topologically Sorted Source Nodes: [matmul_12, attn_13], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36.run(buf330, buf329, 131072, stream=stream0)
            buf331 = reinterpret_tensor(buf281, (512, 64, 1), (64, 1, 1), 0); del buf281  # reuse
            # Topologically Sorted Source Nodes: [matmul_12, attn_13], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_286, reinterpret_tensor(buf330, (512, 256, 1), (256, 1, 0), 0), out=buf331)
            del permute_286
            buf333 = reinterpret_tensor(buf291, (128, 1, 512), (512, 512, 1), 0); del buf291  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_37.run(buf331, buf328, buf333, 65536, stream=stream0)
            buf335 = reinterpret_tensor(buf331, (128, 256), (256, 1), 0); del buf331  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf288, reinterpret_tensor(buf333, (128, 512), (512, 1), 0), permute_294, alpha=1, beta=1, out=buf335)
            del permute_294
            buf326 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf323, buf326, 65536, stream=stream0)
            del buf323
            buf334 = reinterpret_tensor(buf329, (512, 256), (256, 1), 0); del buf329  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf333, (512, 128), (1, 512), 0), addmm_3, out=buf334)
            buf336 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf334, buf336, 131072, stream=stream0)
            del buf334
            buf332 = reinterpret_tensor(buf327, (512, 256, 64), (16384, 64, 1), 0); del buf327  # reuse
            # Topologically Sorted Source Nodes: [matmul_12, attn_13], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf330, (512, 256, 1), (256, 1, 0), 0), permute_287, out=buf332)
            del permute_287
            buf337 = buf322; del buf322  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_39.run(buf332, buf337, 8388608, stream=stream0)
            buf338 = reinterpret_tensor(buf333, (256, 256), (256, 1), 0); del buf333  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf337, (256, 32768), (1, 256), 0), view_144, out=buf338)
            del view_144
            buf339 = reinterpret_tensor(buf332, (32768, 256), (256, 1), 0); del buf332  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf337, permute_298, out=buf339)
            del buf337
            del permute_298
            buf341 = buf294; del buf294  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_transpose_view_40.run(buf341, buf317, buf339, mul_281, 8388608, stream=stream0)
            del mul_281
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf344 = torch.ops.aten.convolution_backward.default(buf341, convert_element_type_276, convert_element_type_275, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_275
            del convert_element_type_276
            buf345 = buf344[0]
            assert_size_stride(buf345, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf345, 16, 'torch.ops.aten.convolution_backward.default')
            buf346 = buf344[1]
            assert_size_stride(buf346, (256, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf346, 16, 'torch.ops.aten.convolution_backward.default')
            del buf344
            buf340 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf338, buf340, 65536, stream=stream0)
            buf342 = reinterpret_tensor(buf324, (256, 128), (1, 256), 0); del buf324  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf341, buf342, 32768, 256, stream=stream0)
            buf348 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf342, buf348, 256, 128, stream=stream0)
            buf347 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf346, buf347, 65536, stream=stream0)
            buf349 = reinterpret_tensor(buf342, (128, 256, 1, 1), (256, 1, 32768, 32768), 0); del buf342  # reuse
            buf350 = reinterpret_tensor(buf339, (128, 256, 256), (65536, 256, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_26.run(buf345, mul_284, buf349, buf350, 32768, 256, stream=stream0)
            buf352 = reinterpret_tensor(buf304, (128, 256, 12), (3072, 12, 1), 0); del buf304  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf350, permute_305, out=buf352)
            del permute_305
            buf351 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_180, buf350, out=buf351)
            buf354 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf351, permute_307, out=buf354)
            del permute_307
            buf353 = buf350; del buf350  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf351, out=buf353)
            buf355 = buf309; del buf309  # reuse
            buf356 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_7], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27.run(buf345, mul_284, buf353, convolution_16, buf355, buf356, 32768, 256, stream=stream0)
            buf357 = buf312; del buf312  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf355, buf357, 4096, 8, stream=stream0)
            buf358 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf356, buf358, 4096, 8, stream=stream0)
            buf360 = reinterpret_tensor(buf317, (128, 256, 16, 16), (65536, 256, 16, 1), 0); del buf317  # reuse
            # Topologically Sorted Source Nodes: [x_norm_7], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29.run(buf345, mul_284, buf353, squeeze_13, convolution_16, buf358, squeeze_12, buf357, buf360, 8388608, stream=stream0)
            del buf345
            del convolution_16
            del mul_284
            del squeeze_12
            del squeeze_13
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf362 = torch.ops.aten.convolution_backward.default(buf360, add_34, convert_element_type_267, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_34
            del convert_element_type_267
            buf366 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_30.run(buf360, buf366, 256, 32768, stream=stream0)
            buf363 = buf362[0]
            assert_size_stride(buf363, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf363, 16, 'torch.ops.aten.convolution_backward.default')
            buf364 = buf362[1]
            assert_size_stride(buf364, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf364, 16, 'torch.ops.aten.convolution_backward.default')
            del buf362
            buf365 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(buf364, buf365, 589824, stream=stream0)
            del buf364
            buf367 = reinterpret_tensor(buf360, (128, 256, 256), (65536, 256, 1), 0); del buf360  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_32.run(buf363, getitem_53, buf367, 8388608, stream=stream0)
            del getitem_53
            buf369 = reinterpret_tensor(buf346, (256, 256), (256, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf367, (256, 32768), (1, 256), 0), view_133, out=buf369)
            del view_133
            buf370 = reinterpret_tensor(buf356, (1, 256, 128), (32768, 1, 256), 0); del buf356  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_33.run(buf367, buf370, 32768, 256, stream=stream0)
            buf368 = reinterpret_tensor(buf353, (32768, 256), (256, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf367, (32768, 256), (256, 1), 0), permute_310, out=buf368)
            del permute_310
            buf371 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_34.run(buf370, buf371, 256, 128, stream=stream0)
            buf373 = reinterpret_tensor(buf367, (128, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf367  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_35.run(buf368, buf373, 8388608, stream=stream0)
            buf374 = reinterpret_tensor(buf288, (512, 1, 64), (64, 64, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_315, reinterpret_tensor(buf373, (512, 256, 64), (16384, 64, 1), 0), out=buf374)
            del permute_315
            buf375 = reinterpret_tensor(buf330, (512, 256, 1), (256, 1, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf373, (512, 256, 64), (16384, 64, 1), 0), permute_316, out=buf375)
            del permute_316
            buf376 = reinterpret_tensor(bmm_22, (128, 4, 256, 1), (1024, 256, 1, 1), 0); del bmm_22  # reuse
            # Topologically Sorted Source Nodes: [matmul_10, attn_11], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36.run(buf376, buf375, 131072, stream=stream0)
            buf377 = reinterpret_tensor(buf328, (512, 64, 1), (64, 1, 1), 0); del buf328  # reuse
            # Topologically Sorted Source Nodes: [matmul_10, attn_11], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_317, reinterpret_tensor(buf376, (512, 256, 1), (256, 1, 0), 0), out=buf377)
            del permute_317
            buf379 = reinterpret_tensor(buf338, (128, 1, 512), (512, 512, 1), 0); del buf338  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_37.run(buf377, buf374, buf379, 65536, stream=stream0)
            buf381 = reinterpret_tensor(buf377, (128, 256), (256, 1), 0); del buf377  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf335, reinterpret_tensor(buf379, (128, 512), (512, 1), 0), permute_325, alpha=1, beta=1, out=buf381)
            del permute_325
            buf372 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf369, buf372, 65536, stream=stream0)
            buf380 = reinterpret_tensor(buf375, (512, 256), (256, 1), 0); del buf375  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf379, (512, 128), (1, 512), 0), addmm_3, out=buf380)
            buf382 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf380, buf382, 131072, stream=stream0)
            del buf380
            buf378 = reinterpret_tensor(buf373, (512, 256, 64), (16384, 64, 1), 0); del buf373  # reuse
            # Topologically Sorted Source Nodes: [matmul_10, attn_11], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf376, (512, 256, 1), (256, 1, 0), 0), permute_318, out=buf378)
            del permute_318
            buf383 = buf368; del buf368  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_39.run(buf378, buf383, 8388608, stream=stream0)
            buf384 = reinterpret_tensor(buf379, (256, 256), (256, 1), 0); del buf379  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf383, (256, 32768), (1, 256), 0), view_121, out=buf384)
            del view_121
            buf385 = reinterpret_tensor(buf378, (32768, 256), (256, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf383, permute_329, out=buf385)
            del buf383
            del permute_329
            buf387 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_41.run(buf387, buf363, buf385, 8388608, stream=stream0)
            del buf363
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf401 = torch.ops.aten.convolution_backward.default(buf387, convert_element_type_236, convert_element_type_235, [256], [2, 2], [1, 1], [1, 1], True, [0, 0], 1, [True, True, False])
            del convert_element_type_235
            del convert_element_type_236
            buf402 = buf401[0]
            assert_size_stride(buf402, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf402, 16, 'torch.ops.aten.convolution_backward.default')
            buf403 = buf401[1]
            assert_size_stride(buf403, (512, 256, 4, 4), (4096, 1, 1024, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf403, 16, 'torch.ops.aten.convolution_backward.default')
            del buf401
            buf386 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf384, buf386, 65536, stream=stream0)
            buf399 = reinterpret_tensor(buf370, (256, 128), (1, 256), 0); del buf370  # reuse
            buf389 = buf335; del buf335  # reuse
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.mul, aten.sigmoid, aten.sum, aten._to_copy, aten.squeeze, aten.sigmoid_backward, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_mul_sigmoid_sigmoid_backward_squeeze_sum_42.run(buf387, convert_element_type_152, addmm_17, buf399, buf389, 32768, 256, stream=stream0)
            buf405 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf399, buf405, 256, 128, stream=stream0)
            buf390 = reinterpret_tensor(buf374, (128, 256), (256, 1), 0); del buf374  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf389, permute_332, out=buf390)
            del permute_332
            buf394 = buf390; del buf390  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf394, addmm_16, 32768, stream=stream0)
            del addmm_16
            buf397 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
            extern_kernels.addmm(buf381, buf394, permute_337, alpha=1, beta=1, out=buf397)
            del permute_337
            buf392 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf389, buf392, 256, 128, stream=stream0)
            buf391 = buf384; del buf384  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf389, (256, 128), (1, 256), 0), convert_element_type_243, out=buf391)
            del convert_element_type_243
            buf396 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf394, buf396, 256, 128, stream=stream0)
            buf395 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf394, (256, 128), (1, 256), 0), addmm_3, out=buf395)
            buf393 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf391, buf393, 65536, stream=stream0)
            buf398 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf395, buf398, 65536, stream=stream0)
            buf408 = reinterpret_tensor(buf399, (512, 64), (1, 512), 0); del buf399  # reuse
            buf410 = reinterpret_tensor(buf355, (512, 64), (1, 512), 0); del buf355  # reuse
            # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._to_copy, aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_43.run(buf402, add_30, getitem_49, rsqrt_7, buf408, buf410, 32768, 128, stream=stream0)
            buf412 = reinterpret_tensor(buf402, (128, 64, 512), (32768, 512, 1), 0); del buf402  # reuse
            # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._to_copy, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_44.run(buf412, primals_76, add_30, getitem_49, rsqrt_7, 8192, 512, stream=stream0)
            del add_30
            del getitem_49
            del primals_76
            del rsqrt_7
            buf414 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf412, (512, 8192), (1, 512), 0), view_117, out=buf414)
            del view_117
            buf409 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._to_copy, aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45.run(buf408, buf409, 512, 64, stream=stream0)
            buf411 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.view, aten.transpose, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45.run(buf410, buf411, 512, 64, stream=stream0)
            buf415 = reinterpret_tensor(buf410, (1, 512, 64), (32768, 1, 512), 0); del buf410  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_46.run(buf412, buf415, 32768, 128, stream=stream0)
            buf416 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45.run(buf415, buf416, 512, 64, stream=stream0)
            buf417 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf414, buf417, 262144, stream=stream0)
            buf404 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_48.run(buf403, buf404, 2097152, stream=stream0)
            del buf403
            buf413 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf412, (8192, 512), (512, 1), 0), permute_342, out=buf413)
            del permute_342
            buf418 = empty_strided_cuda((128, 8, 64, 64), (32768, 4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_49.run(buf413, buf418, 4194304, stream=stream0)
            buf419 = reinterpret_tensor(buf413, (1024, 64, 64), (4096, 64, 1), 0); del buf413  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_347, reinterpret_tensor(buf418, (1024, 64, 64), (4096, 64, 1), 0), out=buf419)
            del permute_347
            buf420 = empty_strided_cuda((1024, 64, 64), (4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf418, (1024, 64, 64), (4096, 64, 1), 0), permute_348, out=buf420)
            del permute_348
            buf422 = reinterpret_tensor(bmm_20, (128, 8, 64, 64), (32768, 4096, 64, 1), 0); del bmm_20  # reuse
            # Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_sub_view_50.run(buf422, buf420, amax_default_5, sum_5, 65536, 64, stream=stream0)
            del amax_default_5
            del sum_5
            buf423 = buf420; del buf420  # reuse
            # Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_349, reinterpret_tensor(buf422, (1024, 64, 64), (4096, 64, 1), 0), out=buf423)
            del permute_349
            buf424 = reinterpret_tensor(buf418, (1024, 64, 64), (4096, 64, 1), 0); del buf418  # reuse
            # Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf422, (1024, 64, 64), (4096, 64, 1), 0), permute_350, out=buf424)
            del buf422
            del permute_350
            buf425 = empty_strided_cuda((128, 64, 3, 8, 64), (98304, 1536, 512, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_clone_permute_select_backward_transpose_view_51.run(buf419, buf423, buf424, buf425, 8192, 1536, stream=stream0)
            del buf419
            buf427 = empty_strided_cuda((1536, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf425, (1536, 8192), (1, 1536), 0), view_107, out=buf427)
            del view_107
            buf428 = empty_strided_cuda((1, 1536, 24), (36864, 1, 1536), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone, aten._unsafe_view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_52.run(buf425, buf428, 36864, 342, stream=stream0)
            buf426 = reinterpret_tensor(buf424, (8192, 512), (512, 1), 0); del buf424  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf425, (8192, 1536), (1536, 1), 0), permute_353, out=buf426)
            del buf425
            del permute_353
            buf429 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.permute, aten.clone, aten._unsafe_view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_permute_select_backward_sum_transpose_view_53.run(buf428, buf429, 1536, 24, stream=stream0)
            del buf428
            buf437 = reinterpret_tensor(buf412, (128, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf412  # reuse
            # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.view, aten._to_copy, aten.native_layer_norm_backward, aten.silu, aten.transpose, aten.native_layer_norm, aten.add, aten.sigmoid, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_sigmoid_silu_sub_transpose_view_54.run(buf437, buf426, primals_70, add_27, getitem_47, rsqrt_6, 8192, 512, stream=stream0)
            del primals_70
            buf433 = reinterpret_tensor(buf415, (512, 64), (1, 512), 0); del buf415  # reuse
            buf435 = buf408; del buf408  # reuse
            # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.view, aten._to_copy, aten.silu, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_silu_transpose_view_55.run(buf426, add_27, getitem_47, rsqrt_6, buf433, buf435, 32768, 128, stream=stream0)
            del add_27
            del getitem_47
            del rsqrt_6
            buf434 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.view, aten._to_copy, aten.silu, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45.run(buf433, buf434, 512, 64, stream=stream0)
            buf436 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten._to_copy, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45.run(buf435, buf436, 512, 64, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf440 = torch.ops.aten.convolution_backward.default(buf437, convert_element_type_212, convert_element_type_211, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_211
            del convert_element_type_212
            buf441 = buf440[0]
            assert_size_stride(buf441, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf441, 16, 'torch.ops.aten.convolution_backward.default')
            buf442 = buf440[1]
            assert_size_stride(buf442, (512, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf442, 16, 'torch.ops.aten.convolution_backward.default')
            del buf440
            buf438 = buf435; del buf435  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_56.run(buf437, buf438, 32768, 128, stream=stream0)
            buf444 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_convolution_backward_57.run(buf438, buf444, 512, 64, stream=stream0)
            buf443 = empty_strided_cuda((512, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf442, buf443, 262144, stream=stream0)
            buf430 = empty_strided_cuda((1536, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_58.run(buf427, buf430, 786432, stream=stream0)
            buf445 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 65536, 65536), torch.float32)
            buf446 = reinterpret_tensor(buf426, (128, 512, 64), (32768, 64, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_mul_sum_view_59.run(buf441, mul_329, buf445, buf446, 65536, 64, stream=stream0)
            buf447 = empty_strided_cuda((128, 12, 64), (768, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_361, buf446, out=buf447)
            buf448 = reinterpret_tensor(buf427, (128, 512, 12), (6144, 12, 1), 0); del buf427  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf446, permute_362, out=buf448)
            del permute_362
            buf450 = empty_strided_cuda((128, 12, 512), (6144, 512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf447, permute_364, out=buf450)
            del permute_364
            buf449 = buf446; del buf446  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_363, buf447, out=buf449)
            buf451 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            buf452 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_5], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_clone_mul_native_group_norm_backward_view_60.run(buf441, mul_329, buf449, convolution_13, buf451, buf452, 65536, 64, stream=stream0)
            buf453 = buf358; del buf358  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_61.run(buf451, buf453, 4096, 16, stream=stream0)
            buf454 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_61.run(buf452, buf454, 4096, 16, stream=stream0)
            buf456 = reinterpret_tensor(buf423, (128, 512, 8, 8), (32768, 64, 8, 1), 0); del buf423  # reuse
            # Topologically Sorted Source Nodes: [x_norm_5], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_62.run(buf441, mul_329, buf449, squeeze_11, convolution_13, buf454, squeeze_10, buf453, buf456, 4194304, stream=stream0)
            del buf441
            del convolution_13
            del mul_329
            del squeeze_10
            del squeeze_11
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf458 = torch.ops.aten.convolution_backward.default(buf456, add_23, convert_element_type_203, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_23
            del convert_element_type_203
            buf462 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_63.run(buf456, buf462, 512, 8192, stream=stream0)
            buf459 = buf458[0]
            assert_size_stride(buf459, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf459, 16, 'torch.ops.aten.convolution_backward.default')
            buf460 = buf458[1]
            assert_size_stride(buf460, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf460, 16, 'torch.ops.aten.convolution_backward.default')
            del buf458
            buf463 = reinterpret_tensor(buf456, (128, 64, 512), (32768, 512, 1), 0); del buf456  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_64.run(buf459, getitem_40, buf463, 4194304, stream=stream0)
            del getitem_40
            buf465 = reinterpret_tensor(buf442, (512, 512), (512, 1), 0); del buf442  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf463, (512, 8192), (1, 512), 0), view_96, out=buf465)
            del view_96
            buf466 = reinterpret_tensor(buf438, (1, 512, 64), (32768, 1, 512), 0); del buf438  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_46.run(buf463, buf466, 32768, 128, stream=stream0)
            buf464 = reinterpret_tensor(buf449, (8192, 512), (512, 1), 0); del buf449  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf463, (8192, 512), (512, 1), 0), permute_367, out=buf464)
            del permute_367
            buf467 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45.run(buf466, buf467, 512, 64, stream=stream0)
            buf469 = reinterpret_tensor(buf463, (128, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf463  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_49.run(buf464, buf469, 4194304, stream=stream0)
            buf471 = reinterpret_tensor(buf395, (1024, 64, 1), (64, 1, 1), 0); del buf395  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf469, (1024, 64, 64), (4096, 64, 1), 0), permute_373, out=buf471)
            del permute_373
            buf470 = reinterpret_tensor(buf391, (1024, 1, 64), (64, 64, 1), 0); del buf391  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_372, reinterpret_tensor(buf469, (1024, 64, 64), (4096, 64, 1), 0), out=buf470)
            del permute_372
            buf472 = reinterpret_tensor(bmm_16, (128, 8, 64, 1), (512, 64, 1, 1), 0); del bmm_16  # reuse
            # Topologically Sorted Source Nodes: [matmul_6, attn_7], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_65.run(buf472, buf471, 65536, stream=stream0)
            buf473 = buf471; del buf471  # reuse
            # Topologically Sorted Source Nodes: [matmul_6, attn_7], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_374, reinterpret_tensor(buf472, (1024, 64, 1), (64, 1, 0), 0), out=buf473)
            del permute_374
            buf475 = reinterpret_tensor(buf376, (128, 1, 1024), (1024, 1024, 1), 0); del buf376  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_66.run(buf473, buf470, buf475, 131072, stream=stream0)
            del buf470
            buf477 = buf394; del buf394  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf397, reinterpret_tensor(buf475, (128, 1024), (1024, 1), 0), permute_382, alpha=1, beta=1, out=buf477)
            del permute_382
            buf476 = reinterpret_tensor(buf414, (1024, 256), (256, 1), 0); del buf414  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf475, (1024, 128), (1, 1024), 0), addmm_3, out=buf476)
            buf468 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf465, buf468, 262144, stream=stream0)
            buf478 = empty_strided_cuda((1024, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf476, buf478, 262144, stream=stream0)
            buf461 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_67.run(buf460, buf461, 2359296, stream=stream0)
            del buf460
            buf474 = reinterpret_tensor(buf469, (1024, 64, 64), (4096, 64, 1), 0); del buf469  # reuse
            # Topologically Sorted Source Nodes: [matmul_6, attn_7], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf472, (1024, 64, 1), (64, 1, 0), 0), permute_375, out=buf474)
            del permute_375
            buf479 = buf464; del buf464  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_68.run(buf474, buf479, 4194304, stream=stream0)
            buf480 = reinterpret_tensor(buf476, (512, 512), (512, 1), 0); del buf476  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf479, (512, 8192), (1, 512), 0), view_84, out=buf480)
            del view_84
            buf481 = reinterpret_tensor(buf474, (8192, 512), (512, 1), 0); del buf474  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf479, permute_386, out=buf481)
            del buf479
            del permute_386
            buf483 = buf437; del buf437  # reuse
            # Topologically Sorted Source Nodes: [add_17], Original ATen: [aten.fill, aten.add, aten.view, aten.transpose, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_69.run(buf483, buf459, buf481, convolution_12, convolution_10, 4194304, stream=stream0)
            del convolution_12
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf486 = torch.ops.aten.convolution_backward.default(buf483, convert_element_type_182, convert_element_type_181, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_181
            del convert_element_type_182
            buf487 = buf486[0]
            assert_size_stride(buf487, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf487, 16, 'torch.ops.aten.convolution_backward.default')
            buf488 = buf486[1]
            assert_size_stride(buf488, (512, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf488, 16, 'torch.ops.aten.convolution_backward.default')
            del buf486
            buf484 = reinterpret_tensor(buf466, (512, 64), (1, 512), 0); del buf466  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_56.run(buf483, buf484, 32768, 128, stream=stream0)
            buf490 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_convolution_backward_57.run(buf484, buf490, 512, 64, stream=stream0)
            buf482 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf480, buf482, 262144, stream=stream0)
            buf489 = empty_strided_cuda((512, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf488, buf489, 262144, stream=stream0)
            buf491 = reinterpret_tensor(buf452, (128, 512, 1, 1), (512, 1, 65536, 65536), 0); del buf452  # reuse
            buf492 = reinterpret_tensor(buf481, (128, 512, 64), (32768, 64, 1), 0); del buf481  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_mul_sum_view_59.run(buf487, mul_351, buf491, buf492, 65536, 64, stream=stream0)
            buf493 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_361, buf492, out=buf493)
            del permute_361
            buf494 = empty_strided_cuda((128, 512, 12), (6144, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf492, permute_393, out=buf494)
            del permute_393
            buf496 = empty_strided_cuda((128, 12, 512), (6144, 512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf493, permute_395, out=buf496)
            del permute_395
            buf497 = empty_strided_cuda((128, 12800), (12800, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.clone, aten._unsafe_view, aten.cat, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_70.run(buf448, buf450, buf445, buf494, buf496, buf491, buf497, 1638400, stream=stream0)
            del buf448
            del buf450
            del buf494
            del buf496
            buf696 = buf397; del buf397  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf497, permute_498, out=buf696)
            del permute_498
            buf700 = buf696; del buf696  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf700, addmm_8, 32768, stream=stream0)
            del addmm_8
            buf702 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf700, buf702, 256, 128, stream=stream0)
            buf698 = empty_strided_cuda((1, 12800), (12800, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_71.run(buf497, buf698, 12800, 128, stream=stream0)
            buf701 = reinterpret_tensor(buf472, (256, 256), (256, 1), 0); del buf472  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf700, (256, 128), (1, 256), 0), addmm_3, out=buf701)
            buf704 = reinterpret_tensor(buf491, (256, 256), (256, 1), 0); del buf491  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf701, buf704, 65536, stream=stream0)
            buf697 = empty_strided_cuda((12800, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf497, (12800, 128), (1, 12800), 0), convert_element_type_56, out=buf697)
            del convert_element_type_56
            buf699 = empty_strided_cuda((12800, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_72.run(buf697, buf699, 3276800, stream=stream0)
            del buf697
            buf495 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_363, buf493, out=buf495)
            del buf493
            del permute_363
            buf498 = reinterpret_tensor(buf445, (128, 512), (512, 1), 0); del buf445  # reuse
            buf499 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [x_norm_4], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_clone_mul_native_group_norm_backward_view_60.run(buf487, mul_351, buf495, convolution_11, buf498, buf499, 65536, 64, stream=stream0)
            buf500 = buf454; del buf454  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_61.run(buf498, buf500, 4096, 16, stream=stream0)
            buf501 = buf453; del buf453  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_61.run(buf499, buf501, 4096, 16, stream=stream0)
            buf503 = reinterpret_tensor(buf459, (128, 512, 8, 8), (32768, 64, 8, 1), 0); del buf459  # reuse
            # Topologically Sorted Source Nodes: [x_norm_4], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_62.run(buf487, mul_351, buf495, squeeze_9, convolution_11, buf501, squeeze_8, buf500, buf503, 4194304, stream=stream0)
            del buf487
            del convolution_11
            del mul_351
            del squeeze_8
            del squeeze_9
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf505 = torch.ops.aten.convolution_backward.default(buf503, add_18, convert_element_type_173, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_18
            del convert_element_type_173
            buf509 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_63.run(buf503, buf509, 512, 8192, stream=stream0)
            buf506 = buf505[0]
            assert_size_stride(buf506, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf506, 16, 'torch.ops.aten.convolution_backward.default')
            buf507 = buf505[1]
            assert_size_stride(buf507, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf507, 16, 'torch.ops.aten.convolution_backward.default')
            del buf505
            buf510 = reinterpret_tensor(buf503, (128, 64, 512), (32768, 512, 1), 0); del buf503  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_64.run(buf506, getitem_31, buf510, 4194304, stream=stream0)
            del getitem_31
            buf512 = reinterpret_tensor(buf488, (512, 512), (512, 1), 0); del buf488  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf510, (512, 8192), (1, 512), 0), view_73, out=buf512)
            del view_73
            buf513 = reinterpret_tensor(buf484, (1, 512, 64), (32768, 1, 512), 0); del buf484  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_46.run(buf510, buf513, 32768, 128, stream=stream0)
            buf511 = reinterpret_tensor(buf495, (8192, 512), (512, 1), 0); del buf495  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf510, (8192, 512), (512, 1), 0), permute_398, out=buf511)
            del permute_398
            buf514 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_view_45.run(buf513, buf514, 512, 64, stream=stream0)
            buf516 = reinterpret_tensor(buf510, (128, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf510  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_49.run(buf511, buf516, 4194304, stream=stream0)
            buf518 = reinterpret_tensor(buf701, (1024, 64, 1), (64, 1, 1), 0); del buf701  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf516, (1024, 64, 64), (4096, 64, 1), 0), permute_404, out=buf518)
            del permute_404
            buf517 = reinterpret_tensor(buf473, (1024, 1, 64), (64, 64, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_403, reinterpret_tensor(buf516, (1024, 64, 64), (4096, 64, 1), 0), out=buf517)
            del permute_403
            buf519 = reinterpret_tensor(bmm_12, (128, 8, 64, 1), (512, 64, 1, 1), 0); del bmm_12  # reuse
            # Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_65.run(buf519, buf518, 65536, stream=stream0)
            buf520 = buf518; del buf518  # reuse
            # Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_405, reinterpret_tensor(buf519, (1024, 64, 1), (64, 1, 0), 0), out=buf520)
            del permute_405
            buf522 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_66.run(buf520, buf517, buf522, 131072, stream=stream0)
            del buf517
            del buf520
            buf524 = buf389; del buf389  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf477, reinterpret_tensor(buf522, (128, 1024), (1024, 1), 0), permute_413, alpha=1, beta=1, out=buf524)
            del permute_413
            buf523 = reinterpret_tensor(buf480, (1024, 256), (256, 1), 0); del buf480  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf522, (1024, 128), (1, 1024), 0), addmm_3, out=buf523)
            buf515 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf512, buf515, 262144, stream=stream0)
            buf525 = empty_strided_cuda((1024, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf523, buf525, 262144, stream=stream0)
            buf508 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_67.run(buf507, buf508, 2359296, stream=stream0)
            del buf507
            buf521 = reinterpret_tensor(buf516, (1024, 64, 64), (4096, 64, 1), 0); del buf516  # reuse
            # Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf519, (1024, 64, 1), (64, 1, 0), 0), permute_406, out=buf521)
            del permute_406
            buf526 = buf511; del buf511  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_68.run(buf521, buf526, 4194304, stream=stream0)
            buf527 = reinterpret_tensor(buf523, (512, 512), (512, 1), 0); del buf523  # reuse
            # Topologically Sorted Source Nodes: [view_30, q_6, q_7], Original ATen: [aten.t, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (512, 8192), (1, 512), 0), reinterpret_tensor(convolution_10, (8192, 512), (512, 1), 0), out=buf527)
            del convolution_10
            buf528 = reinterpret_tensor(buf521, (8192, 512), (512, 1), 0); del buf521  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf526, permute_417, out=buf528)
            del buf526
            del permute_417
            buf530 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_73.run(buf530, buf506, buf528, 4194304, stream=stream0)
            del buf506
            del buf528
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf533 = torch.ops.aten.convolution_backward.default(buf530, convert_element_type_152, convert_element_type_154, [512], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_152
            del convert_element_type_154
            buf531 = reinterpret_tensor(buf513, (512, 64), (1, 512), 0); del buf513  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_56.run(buf530, buf531, 32768, 128, stream=stream0)
            del buf530
            buf537 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_convolution_backward_57.run(buf531, buf537, 512, 64, stream=stream0)
            buf534 = buf533[0]
            assert_size_stride(buf534, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf534, 16, 'torch.ops.aten.convolution_backward.default')
            buf538 = buf387; del buf387  # reuse
            # Topologically Sorted Source Nodes: [input_27, unsqueeze_5, gate], Original ATen: [aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_unsqueeze_74.run(buf538, addmm_17, buf534, mul_370, 8388608, stream=stream0)
            del addmm_17
            del mul_370
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf541 = torch.ops.aten.convolution_backward.default(buf538, convert_element_type_150, convert_element_type_149, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_149
            del convert_element_type_150
            buf542 = buf541[0]
            assert_size_stride(buf542, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf542, 16, 'torch.ops.aten.convolution_backward.default')
            buf535 = buf533[1]
            assert_size_stride(buf535, (512, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf535, 16, 'torch.ops.aten.convolution_backward.default')
            del buf533
            buf543 = buf541[1]
            assert_size_stride(buf543, (256, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf543, 16, 'torch.ops.aten.convolution_backward.default')
            del buf541
            buf539 = reinterpret_tensor(buf531, (256, 128), (1, 256), 0); del buf531  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf538, buf539, 32768, 256, stream=stream0)
            buf545 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf539, buf545, 256, 128, stream=stream0)
            buf544 = reinterpret_tensor(buf499, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf499  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf543, buf544, 65536, stream=stream0)
            buf529 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf527, buf529, 262144, stream=stream0)
            buf536 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_75.run(buf535, buf536, 1179648, stream=stream0)
            del buf535
            buf546 = reinterpret_tensor(buf539, (128, 256, 1, 1), (256, 1, 32768, 32768), 0); del buf539  # reuse
            buf547 = reinterpret_tensor(buf534, (128, 256, 256), (65536, 256, 1), 0); del buf534  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_26.run(buf542, mul_373, buf546, buf547, 32768, 256, stream=stream0)
            buf549 = reinterpret_tensor(buf351, (128, 256, 12), (3072, 12, 1), 0); del buf351  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf547, permute_424, out=buf549)
            del permute_424
            buf548 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_180, buf547, out=buf548)
            buf551 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf548, permute_426, out=buf551)
            del permute_426
            buf550 = buf547; del buf547  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf548, out=buf550)
            buf553 = reinterpret_tensor(buf433, (128, 256), (256, 1), 0); del buf433  # reuse
            buf554 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_3], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27.run(buf542, mul_373, buf550, convolution_8, buf553, buf554, 32768, 256, stream=stream0)
            buf555 = buf501; del buf501  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf553, buf555, 4096, 8, stream=stream0)
            del buf553
            buf556 = buf500; del buf500  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf554, buf556, 4096, 8, stream=stream0)
            buf558 = reinterpret_tensor(buf385, (128, 256, 16, 16), (65536, 256, 16, 1), 0); del buf385  # reuse
            # Topologically Sorted Source Nodes: [x_norm_3], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29.run(buf542, mul_373, buf550, squeeze_7, convolution_8, buf556, squeeze_6, buf555, buf558, 8388608, stream=stream0)
            del buf542
            del convolution_8
            del mul_373
            del squeeze_6
            del squeeze_7
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf560 = torch.ops.aten.convolution_backward.default(buf558, add_13, convert_element_type_141, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_13
            del convert_element_type_141
            buf564 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_30.run(buf558, buf564, 256, 32768, stream=stream0)
            buf561 = buf560[0]
            assert_size_stride(buf561, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf561, 16, 'torch.ops.aten.convolution_backward.default')
            buf562 = buf560[1]
            assert_size_stride(buf562, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf562, 16, 'torch.ops.aten.convolution_backward.default')
            del buf560
            buf563 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(buf562, buf563, 589824, stream=stream0)
            del buf562
            buf565 = reinterpret_tensor(buf558, (128, 256, 256), (65536, 256, 1), 0); del buf558  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_32.run(buf561, getitem_22, buf565, 8388608, stream=stream0)
            del getitem_22
            buf567 = reinterpret_tensor(buf543, (256, 256), (256, 1), 0); del buf543  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf565, (256, 32768), (1, 256), 0), view_50, out=buf567)
            del view_50
            buf568 = reinterpret_tensor(buf554, (1, 256, 128), (32768, 1, 256), 0); del buf554  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_33.run(buf565, buf568, 32768, 256, stream=stream0)
            buf566 = reinterpret_tensor(buf550, (32768, 256), (256, 1), 0); del buf550  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf565, (32768, 256), (256, 1), 0), permute_429, out=buf566)
            del permute_429
            buf569 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_34.run(buf568, buf569, 256, 128, stream=stream0)
            buf571 = reinterpret_tensor(buf565, (128, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf565  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_35.run(buf566, buf571, 8388608, stream=stream0)
            buf572 = reinterpret_tensor(buf477, (512, 1, 64), (64, 64, 1), 0); del buf477  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_434, reinterpret_tensor(buf571, (512, 256, 64), (16384, 64, 1), 0), out=buf572)
            del permute_434
            buf573 = reinterpret_tensor(buf522, (512, 256, 1), (256, 1, 1), 0); del buf522  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf571, (512, 256, 64), (16384, 64, 1), 0), permute_435, out=buf573)
            del permute_435
            buf574 = reinterpret_tensor(bmm_8, (128, 4, 256, 1), (1024, 256, 1, 1), 0); del bmm_8  # reuse
            # Topologically Sorted Source Nodes: [matmul_2, attn_3], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36.run(buf574, buf573, 131072, stream=stream0)
            buf575 = reinterpret_tensor(buf381, (512, 64, 1), (64, 1, 1), 0); del buf381  # reuse
            # Topologically Sorted Source Nodes: [matmul_2, attn_3], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_436, reinterpret_tensor(buf574, (512, 256, 1), (256, 1, 0), 0), out=buf575)
            del permute_436
            buf577 = reinterpret_tensor(buf519, (128, 1, 512), (512, 512, 1), 0); del buf519  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_37.run(buf575, buf572, buf577, 65536, stream=stream0)
            buf579 = reinterpret_tensor(buf575, (128, 256), (256, 1), 0); del buf575  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf524, reinterpret_tensor(buf577, (128, 512), (512, 1), 0), permute_444, alpha=1, beta=1, out=buf579)
            del permute_444
            buf570 = reinterpret_tensor(buf498, (256, 256), (256, 1), 0); del buf498  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf567, buf570, 65536, stream=stream0)
            del buf567
            buf578 = reinterpret_tensor(buf573, (512, 256), (256, 1), 0); del buf573  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf577, (512, 128), (1, 512), 0), addmm_3, out=buf578)
            buf580 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf578, buf580, 131072, stream=stream0)
            del buf578
            buf576 = reinterpret_tensor(buf571, (512, 256, 64), (16384, 64, 1), 0); del buf571  # reuse
            # Topologically Sorted Source Nodes: [matmul_2, attn_3], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf574, (512, 256, 1), (256, 1, 0), 0), permute_437, out=buf576)
            del permute_437
            buf581 = buf566; del buf566  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_39.run(buf576, buf581, 8388608, stream=stream0)
            buf582 = reinterpret_tensor(buf577, (256, 256), (256, 1), 0); del buf577  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf581, (256, 32768), (1, 256), 0), view_38, out=buf582)
            del view_38
            buf583 = reinterpret_tensor(buf576, (32768, 256), (256, 1), 0); del buf576  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf581, permute_448, out=buf583)
            del buf581
            del permute_448
            buf585 = buf538; del buf538  # reuse
            # Topologically Sorted Source Nodes: [add_9], Original ATen: [aten.fill, aten.add, aten.view, aten.transpose, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_transpose_view_76.run(buf585, buf561, buf583, convolution_7, convolution_5, 8388608, stream=stream0)
            del convolution_7
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf588 = torch.ops.aten.convolution_backward.default(buf585, convert_element_type_120, convert_element_type_119, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_119
            del convert_element_type_120
            buf589 = buf588[0]
            assert_size_stride(buf589, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf589, 16, 'torch.ops.aten.convolution_backward.default')
            buf590 = buf588[1]
            assert_size_stride(buf590, (256, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf590, 16, 'torch.ops.aten.convolution_backward.default')
            del buf588
            buf584 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf582, buf584, 65536, stream=stream0)
            buf586 = reinterpret_tensor(buf568, (256, 128), (1, 256), 0); del buf568  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf585, buf586, 32768, 256, stream=stream0)
            buf592 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf586, buf592, 256, 128, stream=stream0)
            buf591 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf590, buf591, 65536, stream=stream0)
            buf593 = reinterpret_tensor(buf586, (128, 256, 1, 1), (256, 1, 32768, 32768), 0); del buf586  # reuse
            buf594 = reinterpret_tensor(buf583, (128, 256, 256), (65536, 256, 1), 0); del buf583  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_26.run(buf589, mul_395, buf593, buf594, 32768, 256, stream=stream0)
            buf595 = buf548; del buf548  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_180, buf594, out=buf595)
            del permute_180
            buf596 = empty_strided_cuda((128, 256, 12), (3072, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf594, permute_455, out=buf596)
            del permute_455
            buf598 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf595, permute_457, out=buf598)
            del permute_457
            buf215 = empty_strided_cuda((128, 6400), (6400, 1), torch.bfloat16)
            buf308 = buf215; del buf215  # reuse
            buf552 = buf308; del buf308  # reuse
            buf599 = buf552; del buf552  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.clone, aten._unsafe_view, aten.cat, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_77.run(buf599, buf166, buf168, buf163, buf212, buf214, buf209, buf259, buf261, buf256, buf305, buf307, buf302, buf352, buf354, buf349, buf549, buf551, buf546, buf596, buf598, buf593, 819200, stream=stream0)
            del buf163
            del buf166
            del buf168
            del buf209
            del buf212
            del buf214
            del buf256
            del buf259
            del buf261
            del buf302
            del buf305
            del buf307
            del buf349
            del buf352
            del buf354
            del buf549
            del buf551
            del buf596
            del buf598
            buf705 = buf524; del buf524  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf599, permute_507, out=buf705)
            del permute_507
            buf709 = buf705; del buf705  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf709, addmm_6, 32768, stream=stream0)
            del addmm_6
            buf711 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf709, buf711, 256, 128, stream=stream0)
            buf707 = empty_strided_cuda((1, 6400), (6400, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_78.run(buf599, buf707, 6400, 128, stream=stream0)
            buf710 = reinterpret_tensor(buf590, (256, 256), (256, 1), 0); del buf590  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf709, (256, 128), (1, 256), 0), addmm_3, out=buf710)
            buf713 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf710, buf713, 65536, stream=stream0)
            buf706 = reinterpret_tensor(buf497, (6400, 256), (256, 1), 0); del buf497  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf599, (6400, 128), (1, 6400), 0), convert_element_type_44, out=buf706)
            del convert_element_type_44
            buf708 = empty_strided_cuda((6400, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_79.run(buf706, buf708, 1638400, stream=stream0)
            del buf706
            buf597 = buf594; del buf594  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_182, buf595, out=buf597)
            del buf595
            del permute_182
            buf600 = reinterpret_tensor(buf593, (128, 256), (256, 1), 0); del buf593  # reuse
            buf601 = reinterpret_tensor(buf546, (128, 256), (256, 1), 0); del buf546  # reuse
            # Topologically Sorted Source Nodes: [x_norm_2], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_27.run(buf589, mul_395, buf597, convolution_6, buf600, buf601, 32768, 256, stream=stream0)
            buf602 = buf556; del buf556  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf600, buf602, 4096, 8, stream=stream0)
            del buf600
            buf603 = buf555; del buf555  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_backward_28.run(buf601, buf603, 4096, 8, stream=stream0)
            buf605 = reinterpret_tensor(buf561, (128, 256, 16, 16), (65536, 256, 16, 1), 0); del buf561  # reuse
            # Topologically Sorted Source Nodes: [x_norm_2], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_29.run(buf589, mul_395, buf597, squeeze_5, convolution_6, buf603, squeeze_4, buf602, buf605, 8388608, stream=stream0)
            del buf589
            del buf602
            del convolution_6
            del mul_395
            del squeeze_4
            del squeeze_5
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf607 = torch.ops.aten.convolution_backward.default(buf605, add_8, convert_element_type_111, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_8
            del convert_element_type_111
            buf611 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_30.run(buf605, buf611, 256, 32768, stream=stream0)
            buf608 = buf607[0]
            assert_size_stride(buf608, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf608, 16, 'torch.ops.aten.convolution_backward.default')
            buf609 = buf607[1]
            assert_size_stride(buf609, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf609, 16, 'torch.ops.aten.convolution_backward.default')
            del buf607
            buf610 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(buf609, buf610, 589824, stream=stream0)
            del buf609
            buf612 = reinterpret_tensor(buf605, (128, 256, 256), (65536, 256, 1), 0); del buf605  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_dropout_backward_transpose_view_32.run(buf608, getitem_13, buf612, 8388608, stream=stream0)
            del getitem_13
            buf614 = buf710; del buf710  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf612, (256, 32768), (1, 256), 0), view_27, out=buf614)
            del view_27
            buf615 = reinterpret_tensor(buf601, (1, 256, 128), (32768, 1, 256), 0); del buf601  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_33.run(buf612, buf615, 32768, 256, stream=stream0)
            buf613 = reinterpret_tensor(buf597, (32768, 256), (256, 1), 0); del buf597  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf612, (32768, 256), (256, 1), 0), permute_460, out=buf613)
            del permute_460
            buf616 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.native_dropout_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_transpose_view_34.run(buf615, buf616, 256, 128, stream=stream0)
            buf618 = reinterpret_tensor(buf612, (128, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf612  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_35.run(buf613, buf618, 8388608, stream=stream0)
            buf619 = buf572; del buf572  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(permute_465, reinterpret_tensor(buf618, (512, 256, 64), (16384, 64, 1), 0), out=buf619)
            del permute_465
            buf620 = reinterpret_tensor(buf574, (512, 256, 1), (256, 1, 1), 0); del buf574  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf618, (512, 256, 64), (16384, 64, 1), 0), permute_466, out=buf620)
            del permute_466
            buf621 = reinterpret_tensor(bmm_4, (128, 4, 256, 1), (1024, 256, 1, 1), 0); del bmm_4  # reuse
            # Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__softmax_backward_data__to_copy_amax_mul_sub_view_36.run(buf621, buf620, 131072, stream=stream0)
            buf622 = empty_strided_cuda((512, 64, 1), (64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(permute_467, reinterpret_tensor(buf621, (512, 256, 1), (256, 1, 0), 0), out=buf622)
            del permute_467
            buf624 = reinterpret_tensor(buf582, (128, 1, 512), (512, 512, 1), 0); del buf582  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_37.run(buf622, buf619, buf624, 65536, stream=stream0)
            del buf619
            buf626 = reinterpret_tensor(buf622, (128, 256), (256, 1), 0); del buf622  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.addmm]
            extern_kernels.addmm(buf579, reinterpret_tensor(buf624, (128, 512), (512, 1), 0), permute_475, alpha=1, beta=1, out=buf626)
            del permute_475
            buf703 = buf579; del buf579  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
            extern_kernels.addmm(buf626, buf700, permute_503, alpha=1, beta=1, out=buf703)
            del buf626
            del permute_503
            buf712 = buf700; del buf700  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
            extern_kernels.addmm(buf703, buf709, permute_512, alpha=1, beta=1, out=buf712)
            del permute_512
            buf617 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf614, buf617, 65536, stream=stream0)
            buf625 = reinterpret_tensor(buf620, (512, 256), (256, 1), 0); del buf620  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.cat, aten.squeeze, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf624, (512, 128), (1, 512), 0), addmm_3, out=buf625)
            buf627 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf625, buf627, 131072, stream=stream0)
            buf623 = reinterpret_tensor(buf618, (512, 256, 64), (16384, 64, 1), 0); del buf618  # reuse
            # Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten._softmax_backward_data, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf621, (512, 256, 1), (256, 1, 0), 0), permute_468, out=buf623)
            del permute_468
            buf628 = buf613; del buf613  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_39.run(buf623, buf628, 8388608, stream=stream0)
            buf629 = reinterpret_tensor(buf624, (256, 256), (256, 1), 0); del buf624  # reuse
            # Topologically Sorted Source Nodes: [view_10, q, q_1], Original ATen: [aten.t, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf628, (256, 32768), (1, 256), 0), reinterpret_tensor(convolution_5, (32768, 256), (256, 1), 0), out=buf629)
            del convolution_5
            buf630 = reinterpret_tensor(buf623, (32768, 256), (256, 1), 0); del buf623  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf628, permute_479, out=buf630)
            del buf628
            del permute_479
            buf632 = buf585; del buf585  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_41.run(buf632, buf608, buf630, 8388608, stream=stream0)
            del buf608
            del buf630
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf635 = torch.ops.aten.convolution_backward.default(buf632, convert_element_type_90, convert_element_type_92, [256], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_90
            del convert_element_type_92
            buf633 = reinterpret_tensor(buf615, (256, 128), (1, 256), 0); del buf615  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_23.run(buf632, buf633, 32768, 256, stream=stream0)
            del buf632
            buf639 = empty_strided_cuda((256, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_24.run(buf633, buf639, 256, 128, stream=stream0)
            del buf633
            buf636 = buf635[0]
            assert_size_stride(buf636, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf636, 16, 'torch.ops.aten.convolution_backward.default')
            buf640 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [input_41, unsqueeze_12, gate_1], Original ATen: [aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_unsqueeze_80.run(buf640, addmm_24, buf636, mul_414, 16384, 1024, stream=stream0)
            del addmm_24
            del mul_414
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf642 = torch.ops.aten.convolution_backward.default(buf640, convert_element_type_88, convert_element_type_87, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_87
            del convert_element_type_88
            buf643 = buf642[0]
            assert_size_stride(buf643, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf643, 16, 'torch.ops.aten.convolution_backward.default')
            buf637 = buf635[1]
            assert_size_stride(buf637, (256, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf637, 16, 'torch.ops.aten.convolution_backward.default')
            del buf635
            buf644 = buf642[1]
            assert_size_stride(buf644, (128, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf644, 16, 'torch.ops.aten.convolution_backward.default')
            del buf642
            buf646 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf640, buf646, 128, 131072, stream=stream0)
            buf645 = reinterpret_tensor(buf127, (128, 128, 1, 1), (128, 1, 128, 128), 0); del buf127  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf644, buf645, 16384, stream=stream0)
            del buf644
            buf631 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf629, buf631, 65536, stream=stream0)
            buf638 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_81.run(buf637, buf638, 294912, stream=stream0)
            del buf637
            buf647 = reinterpret_tensor(buf126, (128, 128, 1, 1), (128, 1, 16384, 16384), 0); del buf126  # reuse
            buf648 = reinterpret_tensor(buf636, (128, 128, 1024), (131072, 1024, 1), 0); del buf636  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_10.run(buf643, mul_417, buf647, buf648, 16384, 1024, stream=stream0)
            buf650 = empty_strided_cuda((128, 128, 12), (1536, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf648, permute_486, out=buf650)
            del permute_486
            buf649 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_131, buf648, out=buf649)
            buf652 = empty_strided_cuda((128, 12, 128), (1536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf649, permute_488, out=buf652)
            del permute_488
            buf651 = buf648; del buf648  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf649, out=buf651)
            buf654 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            buf655 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_1], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11.run(buf643, mul_417, buf651, convolution_3, buf654, buf655, 16384, 1024, stream=stream0)
            buf656 = buf603; del buf603  # reuse
            buf657 = squeeze_2; del squeeze_2  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_backward_12.run(buf657, buf655, buf654, squeeze_3, buf656, 4096, stream=stream0)
            buf658 = reinterpret_tensor(buf651, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf651  # reuse
            # Topologically Sorted Source Nodes: [x_norm_1], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13.run(buf658, buf643, mul_417, squeeze_3, convolution_3, buf656, buf657, 16384, 1024, stream=stream0)
            del buf643
            del buf656
            del convolution_3
            del mul_417
            del squeeze_3
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf660 = torch.ops.aten.convolution_backward.default(buf658, convert_element_type_77, convert_element_type_79, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_77
            del convert_element_type_79
            buf664 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf658, buf664, 128, 131072, stream=stream0)
            del buf658
            buf661 = buf660[0]
            assert_size_stride(buf661, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf661, 16, 'torch.ops.aten.convolution_backward.default')
            buf665 = buf640; del buf640  # reuse
            # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.fill, aten.add, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_82.run(buf665, buf661, convolution_2, convolution, 131072, 128, stream=stream0)
            del convolution_2
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf667 = torch.ops.aten.convolution_backward.default(buf665, convert_element_type_75, convert_element_type_74, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_74
            del convert_element_type_75
            buf668 = buf667[0]
            assert_size_stride(buf668, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf668, 16, 'torch.ops.aten.convolution_backward.default')
            buf662 = buf660[1]
            assert_size_stride(buf662, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf662, 16, 'torch.ops.aten.convolution_backward.default')
            del buf660
            buf669 = buf667[1]
            assert_size_stride(buf669, (128, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf669, 16, 'torch.ops.aten.convolution_backward.default')
            del buf667
            buf671 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf665, buf671, 128, 131072, stream=stream0)
            buf670 = reinterpret_tensor(buf655, (128, 128, 1, 1), (128, 1, 128, 128), 0); del buf655  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf669, buf670, 16384, stream=stream0)
            del buf669
            buf663 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(buf662, buf663, 147456, stream=stream0)
            del buf662
            buf672 = reinterpret_tensor(buf654, (128, 128, 1, 1), (128, 1, 16384, 16384), 0); del buf654  # reuse
            buf673 = reinterpret_tensor(buf661, (128, 128, 1024), (131072, 1024, 1), 0); del buf661  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_10.run(buf668, mul_435, buf672, buf673, 16384, 1024, stream=stream0)
            buf675 = empty_strided_cuda((128, 128, 12), (1536, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf673, permute_494, out=buf675)
            del permute_494
            buf674 = buf649; del buf649  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_131, buf673, out=buf674)
            del permute_131
            buf677 = empty_strided_cuda((128, 12, 128), (1536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(buf674, permute_496, out=buf677)
            del permute_496
            buf52 = empty_strided_cuda((128, 3200), (3200, 1), torch.bfloat16)
            buf101 = buf52; del buf52  # reuse
            buf653 = buf101; del buf101  # reuse
            buf678 = buf653; del buf653  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.clone, aten._unsafe_view, aten.cat, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_83.run(buf678, buf25, buf27, buf22, buf49, buf51, buf46, buf74, buf76, buf71, buf98, buf100, buf95, buf123, buf125, buf120, buf650, buf652, buf647, buf675, buf677, buf672, 409600, stream=stream0)
            del buf100
            del buf120
            del buf123
            del buf125
            del buf22
            del buf25
            del buf27
            del buf46
            del buf49
            del buf51
            del buf647
            del buf650
            del buf652
            del buf672
            del buf675
            del buf677
            del buf74
            del buf76
            del buf98
            buf714 = buf709; del buf709  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf678, permute_516, out=buf714)
            del permute_516
            buf718 = buf714; del buf714  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_18.run(buf718, addmm_4, 32768, stream=stream0)
            del addmm_4
            buf721 = buf703; del buf703  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
            extern_kernels.addmm(buf712, buf718, permute_521, alpha=1, beta=1, out=buf721)
            del permute_521
            buf723 = reinterpret_tensor(buf629, (128, 512), (512, 1), 0); del buf629  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf721, permute_525, out=buf723)
            del permute_525
            buf727 = buf723; del buf723  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.fill, aten.sub, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_84.run(buf727, addmm_2, 65536, stream=stream0)
            del addmm_2
            buf728 = reinterpret_tensor(buf614, (128, 512), (512, 1), 0); del buf614  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf727, permute_530, out=buf728)
            del permute_530
            buf732 = buf712; del buf712  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.slice]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_slice_85.run(buf728, buf732, 32768, stream=stream0)
            buf733 = reinterpret_tensor(buf621, (128, 1024), (1024, 1), 0); del buf621  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf732, permute_534, out=buf733)
            del permute_534
            buf737 = buf733; del buf733  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.fill, aten.sub, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_fill_mul_sigmoid_sub_86.run(buf737, addmm, 131072, stream=stream0)
            del addmm
            buf720 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf718, buf720, 256, 128, stream=stream0)
            buf719 = reinterpret_tensor(buf728, (256, 256), (256, 1), 0); del buf728  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.fill, aten.sigmoid, aten.sub, aten.mul, aten.add, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf718, (256, 128), (1, 256), 0), addmm_3, out=buf719)
            del addmm_3
            del buf718
            buf725 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf721, buf725, 256, 128, stream=stream0)
            buf735 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_20.run(buf732, buf735, 256, 128, stream=stream0)
            buf730 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_87.run(buf727, buf730, 512, 128, stream=stream0)
            buf739 = empty_strided_cuda((1, 1024), (1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_88.run(buf737, buf739, 1024, 128, stream=stream0)
            buf716 = empty_strided_cuda((1, 3200), (3200, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_89.run(buf678, buf716, 3200, 128, stream=stream0)
            buf724 = reinterpret_tensor(buf625, (256, 512), (512, 1), 0); del buf625  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf721, (256, 128), (1, 256), 0), convert_element_type_20, out=buf724)
            del buf721
            del convert_element_type_20
            buf722 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_22.run(buf719, buf722, 65536, stream=stream0)
            del buf719
            buf734 = reinterpret_tensor(buf527, (256, 1024), (1024, 1), 0); del buf527  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf732, (256, 128), (1, 256), 0), convert_element_type_7, out=buf734)
            del buf732
            del convert_element_type_7
            buf738 = reinterpret_tensor(buf512, (1024, 256), (256, 1), 0); del buf512  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.fill, aten.sub, aten.mul, aten.add, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf737, (1024, 128), (1, 1024), 0), convert_element_type_2, out=buf738)
            del buf737
            del convert_element_type_2
            buf726 = empty_strided_cuda((256, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(buf724, buf726, 131072, stream=stream0)
            del buf724
            buf729 = buf465; del buf465  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf727, (512, 128), (1, 512), 0), convert_element_type_15, out=buf729)
            del buf727
            del convert_element_type_15
            buf731 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf729, buf731, 262144, stream=stream0)
            del buf729
            buf736 = empty_strided_cuda((256, 1024), (1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf734, buf736, 262144, stream=stream0)
            del buf734
            buf740 = empty_strided_cuda((1024, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(buf738, buf740, 262144, stream=stream0)
            del buf738
            buf715 = reinterpret_tensor(buf599, (3200, 256), (256, 1), 0); del buf599  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf678, (3200, 128), (1, 3200), 0), convert_element_type_32, out=buf715)
            del buf678
            del convert_element_type_32
            buf717 = empty_strided_cuda((3200, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_90.run(buf715, buf717, 819200, stream=stream0)
            del buf715
            buf676 = buf673; del buf673  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            extern_kernels.bmm(permute_133, buf674, out=buf676)
            del buf674
            del permute_133
            buf679 = reinterpret_tensor(buf95, (128, 128), (128, 1), 0); del buf95  # reuse
            buf680 = reinterpret_tensor(buf71, (128, 128), (128, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [x_norm], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_mul_native_group_norm_backward_view_11.run(buf668, mul_435, buf676, convolution_1, buf679, buf680, 16384, 1024, stream=stream0)
            buf681 = buf657; del buf657  # reuse
            buf682 = squeeze; del squeeze  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_backward_12.run(buf682, buf680, buf679, squeeze_1, buf681, 4096, stream=stream0)
            del buf679
            del buf680
            buf683 = reinterpret_tensor(buf676, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf676  # reuse
            # Topologically Sorted Source Nodes: [x_norm], Original ATen: [aten.native_group_norm_backward, aten._to_copy, aten.mul, aten.view, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13.run(buf683, buf668, mul_435, squeeze_1, convolution_1, buf681, buf682, 16384, 1024, stream=stream0)
            del buf668
            del buf681
            del buf682
            del convolution_1
            del mul_435
            del squeeze_1
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf685 = torch.ops.aten.convolution_backward.default(buf683, convolution, convert_element_type_66, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del convert_element_type_66
            del convolution
            buf690 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf683, buf690, 128, 131072, stream=stream0)
            del buf683
            buf686 = buf685[0]
            assert_size_stride(buf686, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf686, 16, 'torch.ops.aten.convolution_backward.default')
            buf688 = buf665; del buf665  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_91.run(buf688, buf686, 16384, 1024, stream=stream0)
            del buf686
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
            buf692 = torch.ops.aten.convolution_backward.default(buf688, convert_element_type_64, convert_element_type_63, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
            del convert_element_type_63
            del convert_element_type_64
            buf695 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_convolution_backward_8.run(buf688, buf695, 128, 131072, stream=stream0)
            del buf688
            buf687 = buf685[1]
            assert_size_stride(buf687, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf687, 16, 'torch.ops.aten.convolution_backward.default')
            del buf685
            buf693 = buf692[1]
            assert_size_stride(buf693, (128, 4, 3, 3), (36, 1, 12, 4), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf693, 16, 'torch.ops.aten.convolution_backward.default')
            del buf692
            buf694 = empty_strided_cuda((128, 4, 3, 3), (36, 1, 12, 4), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf693, buf694, 4608, stream=stream0)
            del buf693
            buf689 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(buf687, buf689, 147456, stream=stream0)
            del buf687
        return (None, buf740, reinterpret_tensor(buf739, (1024, ), (1, ), 0), buf736, reinterpret_tensor(buf735, (256, ), (1, ), 0), None, buf731, reinterpret_tensor(buf730, (512, ), (1, ), 0), buf726, reinterpret_tensor(buf725, (256, ), (1, ), 0), buf722, reinterpret_tensor(buf720, (256, ), (1, ), 0), buf717, reinterpret_tensor(buf716, (3200, ), (1, ), 0), buf713, reinterpret_tensor(buf711, (256, ), (1, ), 0), buf708, reinterpret_tensor(buf707, (6400, ), (1, ), 0), buf704, reinterpret_tensor(buf702, (256, ), (1, ), 0), buf699, reinterpret_tensor(buf698, (12800, ), (1, ), 0), buf694, buf695, None, buf689, buf690, buf670, buf671, buf663, buf664, buf645, buf646, buf638, buf639, buf631, buf627, buf617, reinterpret_tensor(buf616, (256, ), (1, ), 0), buf610, buf611, buf591, buf592, buf584, buf580, buf570, reinterpret_tensor(buf569, (256, ), (1, ), 0), buf563, buf564, buf544, buf545, buf536, buf537, buf529, buf525, buf515, reinterpret_tensor(buf514, (512, ), (1, ), 0), buf508, buf509, buf489, buf490, buf482, buf478, buf468, reinterpret_tensor(buf467, (512, ), (1, ), 0), buf461, buf462, buf443, buf444, buf434, buf436, buf430, reinterpret_tensor(buf429, (1536, ), (1, ), 0), buf417, reinterpret_tensor(buf416, (512, ), (1, ), 0), buf409, buf411, buf404, buf405, buf398, reinterpret_tensor(buf396, (256, ), (1, ), 0), buf393, reinterpret_tensor(buf392, (256, ), (1, ), 0), buf386, buf382, buf372, reinterpret_tensor(buf371, (256, ), (1, ), 0), buf365, buf366, buf347, buf348, buf340, buf336, buf326, reinterpret_tensor(buf325, (256, ), (1, ), 0), buf319, buf320, buf300, buf301, buf293, buf289, buf279, reinterpret_tensor(buf278, (256, ), (1, ), 0), buf272, buf273, buf254, buf255, buf247, buf243, buf233, reinterpret_tensor(buf232, (256, ), (1, ), 0), buf226, buf227, buf207, buf208, buf200, buf196, buf186, reinterpret_tensor(buf185, (256, ), (1, ), 0), buf179, buf180, buf161, buf162, buf153, buf154, buf148, reinterpret_tensor(buf147, (256, ), (1, ), 0), buf143, reinterpret_tensor(buf142, (128, ), (1, ), 0), buf136, buf137, buf118, buf119, buf111, buf112, buf93, buf94, buf86, buf87, buf69, buf70, buf62, buf63, buf44, buf45, buf37, buf38, buf20, buf21, reinterpret_tensor(buf13, (128, ), (1, ), 0), buf14, buf5, buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_2 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_7 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_15 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_2 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_20 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_3 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_4 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_32 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_6 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_44 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_8 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_56 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_63 = rand_strided((128, 4, 3, 3), (36, 1, 12, 4), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_64 = rand_strided((128, 4, 32, 32), (4096, 1, 128, 4), device='cuda:0', dtype=torch.bfloat16)
    convolution = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_66 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_1 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_74 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_75 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    convolution_2 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_77 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_79 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_3 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_2 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_3 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_87 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_88 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_90 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_92 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_5 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    bmm_4 = rand_strided((512, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_27 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_13 = rand_strided((128, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.bool)
    add_8 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_111 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_6 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_4 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_5 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_119 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_120 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    convolution_7 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    view_38 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_8 = rand_strided((512, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_50 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_22 = rand_strided((128, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.bool)
    add_13 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_141 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_8 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_6 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_149 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_150 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_152 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_154 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_10 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    bmm_12 = rand_strided((1024, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_73 = rand_strided((8192, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_31 = rand_strided((128, 64, 512), (32768, 512, 1), device='cuda:0', dtype=torch.bool)
    add_18 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_173 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_11 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_8 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_9 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_181 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_182 = rand_strided((128, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.bfloat16)
    convolution_12 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    view_84 = rand_strided((8192, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_16 = rand_strided((1024, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_96 = rand_strided((8192, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_40 = rand_strided((128, 64, 512), (32768, 512, 1), device='cuda:0', dtype=torch.bool)
    add_23 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_203 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_13 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_10 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_11 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_211 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_212 = rand_strided((128, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.bfloat16)
    add_27 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_47 = rand_strided((128, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((128, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((8192, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_20 = rand_strided((1024, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_5 = rand_strided((128, 8, 64, 1), (512, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_5 = rand_strided((128, 8, 64, 1), (512, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((8192, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    add_30 = rand_strided((128, 64, 512), (32768, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_49 = rand_strided((128, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((128, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_235 = rand_strided((512, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_236 = rand_strided((128, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    addmm_16 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_243 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_17 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    view_121 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_22 = rand_strided((512, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_133 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_53 = rand_strided((128, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.bool)
    add_34 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_267 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_16 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_12 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_275 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_276 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    view_144 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_26 = rand_strided((512, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_156 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_62 = rand_strided((128, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.bool)
    add_39 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_297 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_18 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_14 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_15 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_305 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_306 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    view_167 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_30 = rand_strided((512, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_179 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_71 = rand_strided((128, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.bool)
    add_44 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_327 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_20 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_16 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_17 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_335 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_336 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    view_190 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_34 = rand_strided((512, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_202 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_80 = rand_strided((128, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.bool)
    add_49 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_357 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_22 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_18 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_365 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_366 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    view_213 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm_38 = rand_strided((512, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    view_225 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_89 = rand_strided((128, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.bool)
    add_54 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_387 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_24 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_20 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_21 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_395 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_396 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_398 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_400 = rand_strided((256, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.bfloat16)
    addmm_23 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_407 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_24 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    add_59 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_414 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_27 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_22 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_23 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_422 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_423 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_425 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_427 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_29 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_24 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_435 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_436 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_438 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_440 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_31 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_26 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_27 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_448 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_449 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_451 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_453 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_33 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_28 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_29 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_461 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_462 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_464 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_466 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_35 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_30 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_474 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_475 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_79 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_121 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_480 = rand_strided((4, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_481 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_99 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_131 = rand_strided((128, 12, 128), (3200, 1, 12), device='cuda:0', dtype=torch.bfloat16)
    permute_132 = rand_strided((128, 1024, 12), (12288, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_133 = rand_strided((128, 128, 12), (3200, 12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_134 = rand_strided((128, 1024, 128), (131072, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    mul_114 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    mul_117 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_140 = rand_strided((128, 1024, 12), (12288, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_142 = rand_strided((128, 1024, 128), (131072, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    mul_132 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    mul_135 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_148 = rand_strided((128, 1024, 12), (12288, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_150 = rand_strided((128, 1024, 128), (131072, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    mul_150 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    mul_153 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((128, 1024, 12), (12288, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_158 = rand_strided((128, 1024, 128), (131072, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    mul_168 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    mul_171 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((128, 1024, 12), (12288, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_166 = rand_strided((128, 1024, 128), (131072, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_168 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_173 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_193 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    mul_196 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((128, 12, 256), (6400, 1, 12), device='cuda:0', dtype=torch.bfloat16)
    permute_181 = rand_strided((128, 256, 12), (3072, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_182 = rand_strided((128, 256, 12), (6400, 12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_183 = rand_strided((128, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_186 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_191 = rand_strided((512, 1, 256), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_192 = rand_strided((512, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_193 = rand_strided((512, 64, 256), (16384, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_194 = rand_strided((512, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_201 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_205 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_215 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    mul_218 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((128, 256, 12), (3072, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_214 = rand_strided((128, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_217 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_222 = rand_strided((512, 1, 256), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_223 = rand_strided((512, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_224 = rand_strided((512, 64, 256), (16384, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_225 = rand_strided((512, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_232 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_236 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_237 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    mul_240 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_243 = rand_strided((128, 256, 12), (3072, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_245 = rand_strided((128, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_248 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_253 = rand_strided((512, 1, 256), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_254 = rand_strided((512, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_255 = rand_strided((512, 64, 256), (16384, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_256 = rand_strided((512, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_263 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_267 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_259 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    mul_262 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((128, 256, 12), (3072, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_276 = rand_strided((128, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_279 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_284 = rand_strided((512, 1, 256), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_285 = rand_strided((512, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_286 = rand_strided((512, 64, 256), (16384, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_287 = rand_strided((512, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_294 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_298 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_281 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    mul_284 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_305 = rand_strided((128, 256, 12), (3072, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_307 = rand_strided((128, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_310 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_315 = rand_strided((512, 1, 256), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_316 = rand_strided((512, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_317 = rand_strided((512, 64, 256), (16384, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_318 = rand_strided((512, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_325 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_329 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_332 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_337 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_342 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_347 = rand_strided((1024, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_348 = rand_strided((1024, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_349 = rand_strided((1024, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_350 = rand_strided((1024, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_353 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_329 = rand_strided((128, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    permute_361 = rand_strided((128, 12, 512), (12800, 1, 12), device='cuda:0', dtype=torch.bfloat16)
    permute_362 = rand_strided((128, 64, 12), (768, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_363 = rand_strided((128, 512, 12), (12800, 12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_364 = rand_strided((128, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_367 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_372 = rand_strided((1024, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_373 = rand_strided((1024, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_374 = rand_strided((1024, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_375 = rand_strided((1024, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_382 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_386 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_351 = rand_strided((128, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((128, 64, 12), (768, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_395 = rand_strided((128, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_398 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_403 = rand_strided((1024, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_404 = rand_strided((1024, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_405 = rand_strided((1024, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_406 = rand_strided((1024, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_413 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_417 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_370 = rand_strided((128, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.bfloat16)
    mul_373 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_424 = rand_strided((128, 256, 12), (3072, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_426 = rand_strided((128, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_429 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_434 = rand_strided((512, 1, 256), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_435 = rand_strided((512, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_436 = rand_strided((512, 64, 256), (16384, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_437 = rand_strided((512, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_444 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_448 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_395 = rand_strided((128, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((128, 256, 12), (3072, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_457 = rand_strided((128, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.bfloat16)
    permute_460 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_465 = rand_strided((512, 1, 256), (256, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_466 = rand_strided((512, 64, 1), (64, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_467 = rand_strided((512, 64, 256), (16384, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_468 = rand_strided((512, 1, 64), (64, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_475 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_479 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_414 = rand_strided((128, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    mul_417 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((128, 1024, 12), (12288, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_488 = rand_strided((128, 1024, 128), (131072, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    mul_435 = rand_strided((128, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_494 = rand_strided((128, 1024, 12), (12288, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_496 = rand_strided((128, 1024, 128), (131072, 1, 1024), device='cuda:0', dtype=torch.bfloat16)
    permute_498 = rand_strided((12800, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_503 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_507 = rand_strided((6400, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_512 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_516 = rand_strided((3200, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_521 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_525 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_530 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_534 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 4, 32, 32), (4096, 1, 128, 4), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_70, primals_76, primals_150, primals_151, convert_element_type_2, addmm, convert_element_type_7, convert_element_type_15, addmm_2, convert_element_type_20, addmm_3, addmm_4, convert_element_type_32, addmm_6, convert_element_type_44, addmm_8, convert_element_type_56, convert_element_type_63, convert_element_type_64, convolution, convert_element_type_66, convolution_1, squeeze, squeeze_1, convert_element_type_74, convert_element_type_75, convolution_2, convert_element_type_77, convert_element_type_79, convolution_3, squeeze_2, squeeze_3, convert_element_type_87, convert_element_type_88, convert_element_type_90, convert_element_type_92, convolution_5, bmm_4, view_27, getitem_13, add_8, convert_element_type_111, convolution_6, squeeze_4, squeeze_5, convert_element_type_119, convert_element_type_120, convolution_7, view_38, bmm_8, view_50, getitem_22, add_13, convert_element_type_141, convolution_8, squeeze_6, squeeze_7, convert_element_type_149, convert_element_type_150, convert_element_type_152, convert_element_type_154, convolution_10, bmm_12, view_73, getitem_31, add_18, convert_element_type_173, convolution_11, squeeze_8, squeeze_9, convert_element_type_181, convert_element_type_182, convolution_12, view_84, bmm_16, view_96, getitem_40, add_23, convert_element_type_203, convolution_13, squeeze_10, squeeze_11, convert_element_type_211, convert_element_type_212, add_27, getitem_47, rsqrt_6, view_107, bmm_20, amax_default_5, sum_5, view_117, add_30, getitem_49, rsqrt_7, convert_element_type_235, convert_element_type_236, addmm_16, convert_element_type_243, addmm_17, view_121, bmm_22, view_133, getitem_53, add_34, convert_element_type_267, convolution_16, squeeze_12, squeeze_13, convert_element_type_275, convert_element_type_276, view_144, bmm_26, view_156, getitem_62, add_39, convert_element_type_297, convolution_18, squeeze_14, squeeze_15, convert_element_type_305, convert_element_type_306, view_167, bmm_30, view_179, getitem_71, add_44, convert_element_type_327, convolution_20, squeeze_16, squeeze_17, convert_element_type_335, convert_element_type_336, view_190, bmm_34, view_202, getitem_80, add_49, convert_element_type_357, convolution_22, squeeze_18, squeeze_19, convert_element_type_365, convert_element_type_366, view_213, bmm_38, view_225, getitem_89, add_54, convert_element_type_387, convolution_24, squeeze_20, squeeze_21, convert_element_type_395, convert_element_type_396, convert_element_type_398, convert_element_type_400, addmm_23, convert_element_type_407, addmm_24, add_59, convert_element_type_414, convolution_27, squeeze_22, squeeze_23, convert_element_type_422, convert_element_type_423, convert_element_type_425, convert_element_type_427, convolution_29, squeeze_24, squeeze_25, convert_element_type_435, convert_element_type_436, convert_element_type_438, convert_element_type_440, convolution_31, squeeze_26, squeeze_27, convert_element_type_448, convert_element_type_449, convert_element_type_451, convert_element_type_453, convolution_33, squeeze_28, squeeze_29, convert_element_type_461, convert_element_type_462, convert_element_type_464, convert_element_type_466, convolution_35, squeeze_30, squeeze_31, convert_element_type_474, convert_element_type_475, add_79, getitem_121, rsqrt_18, convert_element_type_480, convert_element_type_481, mul_99, permute_131, permute_132, permute_133, permute_134, mul_114, mul_117, permute_140, permute_142, mul_132, mul_135, permute_148, permute_150, mul_150, mul_153, permute_156, permute_158, mul_168, mul_171, permute_164, permute_166, permute_168, permute_173, mul_193, mul_196, permute_180, permute_181, permute_182, permute_183, permute_186, permute_191, permute_192, permute_193, permute_194, permute_201, permute_205, mul_215, mul_218, permute_212, permute_214, permute_217, permute_222, permute_223, permute_224, permute_225, permute_232, permute_236, mul_237, mul_240, permute_243, permute_245, permute_248, permute_253, permute_254, permute_255, permute_256, permute_263, permute_267, mul_259, mul_262, permute_274, permute_276, permute_279, permute_284, permute_285, permute_286, permute_287, permute_294, permute_298, mul_281, mul_284, permute_305, permute_307, permute_310, permute_315, permute_316, permute_317, permute_318, permute_325, permute_329, permute_332, permute_337, permute_342, permute_347, permute_348, permute_349, permute_350, permute_353, mul_329, permute_361, permute_362, permute_363, permute_364, permute_367, permute_372, permute_373, permute_374, permute_375, permute_382, permute_386, mul_351, permute_393, permute_395, permute_398, permute_403, permute_404, permute_405, permute_406, permute_413, permute_417, mul_370, mul_373, permute_424, permute_426, permute_429, permute_434, permute_435, permute_436, permute_437, permute_444, permute_448, mul_395, permute_455, permute_457, permute_460, permute_465, permute_466, permute_467, permute_468, permute_475, permute_479, mul_414, mul_417, permute_486, permute_488, mul_435, permute_494, permute_496, permute_498, permute_503, permute_507, permute_512, permute_516, permute_521, permute_525, permute_530, permute_534, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
