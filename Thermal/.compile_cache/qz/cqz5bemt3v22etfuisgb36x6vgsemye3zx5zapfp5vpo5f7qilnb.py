# AOT ID: ['3_inference']
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2n/c2nk2l7kb2fybbnzsr3m3xhib32ugz3vhrcsajsba23z6stjglyt.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h => convert_element_type_64
# Graph fragment:
#   %arg25_1 : Tensor "f32[1, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %convert_element_type_64 : Tensor "bf16[1, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_64
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 8192, 'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/sw/cswtrigvza676xunm5ubn62qvfgjh7i3k2asaqlus6kzgs5kqdnn.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h => convert_element_type_63
# Graph fragment:
#   %arg23_1 : Tensor "f32[128, 4, 3, 3][36, 1, 12, 4]cuda:0" = PlaceHolder[target=arg23_1]
#   %convert_element_type_63 : Tensor "bf16[128, 4, 3, 3][36, 1, 12, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg23_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_63
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/xt/cxt25i46rdheqkrmpigy6lku4u4pavdennnegexcw74652gjmybe.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h => convert_element_type_62, convert_element_type_63, convert_element_type_64, convolution
# Graph fragment:
#   %buf2 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf2]
#   %arg24_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %convert_element_type_64 : Tensor "bf16[1, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_63 : Tensor "bf16[128, 4, 3, 3][36, 1, 12, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg23_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg24_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_64, %convert_element_type_63, %convert_element_type_62, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused__to_copy_convolution_2 = async_compile.triton('triton_poi_fused__to_copy_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 786944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ce/ccebohbqts3ddy32c6bwue3gb3sziel6lpdxi6dbltjvx554pm7n.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type_66
# Graph fragment:
#   %arg26_1 : Tensor "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=arg26_1]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_66
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/5n/c5ngsgtzzo5tbmhh3ufdx6erlyt5jdfnm4l2qstqzxqettm2b2mw.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convert_element_type_65, convert_element_type_66, convolution_1
#   x_norm => clone, convert_element_type_67, var_mean, view
# Graph fragment:
#   %buf5 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %convert_element_type_66, %convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_67 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf7
triton_red_fused__to_copy_clone_convolution_native_group_norm_4 = async_compile.triton('triton_red_fused__to_copy_clone_convolution_native_group_norm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_convolution_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 262656}}
)
@triton.jit
def triton_red_fused__to_copy_clone_convolution_native_group_norm_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 4)
        r0_2 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4*x0 + 128*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/bq/cbqjhjyixheglhpn7gtd2nolya2gfx7xteny7to3wzmxzjkxong4.py
# Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   arange => iota
#   args => mul_1
#   cos => cos
#   emb => cat
#   freqs => exp
#   getitem => unsqueeze
#   getitem_1 => unsqueeze_1
#   input_1 => convert_element_type_2
#   mul => mul
#   sin => sin
#   truediv => div
# Graph fragment:
#   %arg0_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %unsqueeze : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, -9.210340371976184), kwargs = {})
#   %div : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 128), kwargs = {})
#   %exp : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 0), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %cos : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %sin : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %cat : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_5 = async_compile.triton('triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(tmp4, tmp6, 0.0)
    tmp8 = x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = -9.210340371976184
    tmp11 = tmp9 * tmp10
    tmp12 = 0.0078125
    tmp13 = tmp11 * tmp12
    tmp14 = libdevice.exp(tmp13)
    tmp15 = tmp7 * tmp14
    tmp16 = tl_math.cos(tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 256, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp24 = tl.where(tmp19, tmp23, 0.0)
    tmp25 = (-128) + x0
    tmp26 = tmp25.to(tl.float32)
    tmp27 = -9.210340371976184
    tmp28 = tmp26 * tmp27
    tmp29 = 0.0078125
    tmp30 = tmp28 * tmp29
    tmp31 = libdevice.exp(tmp30)
    tmp32 = tmp24 * tmp31
    tmp33 = tl_math.sin(tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp18, tmp35)
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qw/cqwznsx4xqdigcuoooymojhl4txhoefdphsy2277ua3yvs5wgwr4.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_1
# Graph fragment:
#   %arg1_1 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type_1 : Tensor "bf16[1024, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lq/clqvdixddxxwt7mzw2l32dro7r57cify5fj5zxj5jgqzklmcss2l.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_18, convert_element_type
#   input_2 => convert_element_type_6, convert_element_type_7, mul_2, sigmoid
# Graph fragment:
#   %mm_default_50 : Tensor "bf16[1, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_default_50]
#   %arg2_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type : Tensor "bf16[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %add_tensor_18 : Tensor "bf16[1, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_50, %convert_element_type), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[1, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_18, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[1, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[1, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_7
triton_poi_fused__to_copy_addmm_silu_7 = async_compile.triton('triton_poi_fused__to_copy_addmm_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_silu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_silu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/om/comshokyeof343f7wsz6zjhymqr6r5tckh3o56pckizqfe6unnw4.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_8
# Graph fragment:
#   %arg4_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convert_element_type_8 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_8
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vh/cvhj3beylly4dink6fxzcwkctpkvxed3vtq2y7tc2a64ovlfsryz.py
# Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   cat_1 => cat_1
#   input_4 => convert_element_type_15
#   s_emb => embedding
# Graph fragment:
#   %addmm_1 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %arg6_1 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg5_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg6_1), kwargs = {})
#   %cat_1 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%addmm_1, %embedding], -1), kwargs = {})
#   %convert_element_type_15 : Tensor "bf16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_15
triton_poi_fused__to_copy_cat_embedding_9 = async_compile.triton('triton_poi_fused__to_copy_cat_embedding_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_embedding_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_embedding_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp14 = tl.where(tmp9, tmp13, 0)
    tmp15 = tl.full([XBLOCK], 4, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert(((0 <= tl.broadcast_to(tmp18, [XBLOCK])) & (tl.broadcast_to(tmp18, [XBLOCK]) < 4)) | ~(tmp9 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp18, [XBLOCK]) < 4")
    tmp20 = tl.load(in_ptr2 + (256*tmp18 + ((-256) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp4, tmp8, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp22, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qj/cqjuvlod5ryozqpa25imbb2nqebjynnz2dsiuoi4pi4e3ya7fvhl.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_4 => add_tensor_17, convert_element_type_13
#   input_5 => convert_element_type_19, convert_element_type_20, mul_3, sigmoid_1
# Graph fragment:
#   %mm_default_49 : Tensor "bf16[1, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_49]
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type_13 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.bfloat16), kwargs = {})
#   %add_tensor_17 : Tensor "bf16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_49, %convert_element_type_13), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_17, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_19,), kwargs = {})
#   %mul_3 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %sigmoid_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_20
triton_poi_fused__to_copy_addmm_silu_10 = async_compile.triton('triton_poi_fused__to_copy_addmm_silu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_silu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_silu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dl/cdl2e6nujjordqmc5xkjsloezlcqnvft2wq7ppxv2nlosdtvcstn.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_6 => convert_element_type_22
# Graph fragment:
#   %arg9_1 : Tensor "f32[256, 512][512, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convert_element_type_22 : Tensor "bf16[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_22
triton_poi_fused__to_copy_11 = async_compile.triton('triton_poi_fused__to_copy_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hl/chlvghmmpy3hxzcmusfztkaa6pdcutimqegovn2ebwmpe2mkilsa.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_7 => convert_element_type_27
# Graph fragment:
#   %arg11_1 : Tensor "f32[256, 256][256, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convert_element_type_27 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_27
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/fs/cfsjfq34fzjbnnbkg4ub32yi55tynxq32ld42iedqxxtw4yf552e.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_7 => add_tensor_16, convert_element_type_26
#   input_8 => convert_element_type_31, convert_element_type_32, mul_4, sigmoid_2
# Graph fragment:
#   %mm_default_48 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_48]
#   %arg12_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %convert_element_type_26 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.bfloat16), kwargs = {})
#   %add_tensor_16 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_48, %convert_element_type_26), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_16, torch.float32), kwargs = {})
#   %sigmoid_2 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_31,), kwargs = {})
#   %mul_4 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, %sigmoid_2), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   return %convert_element_type_32
triton_poi_fused__to_copy_addmm_silu_13 = async_compile.triton('triton_poi_fused__to_copy_addmm_silu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_silu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_silu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/s4/cs4o2k3psbbvh2m4xn7geqxgno3cgngmj2wzwfipgcgxiwsrxs5y.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_34
# Graph fragment:
#   %arg13_1 : Tensor "f32[3200, 256][256, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convert_element_type_34 : Tensor "bf16[3200, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_34
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6553600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/f4/cf4olj5xtfyslxkd32thic7tuaagvj7yf3aryshpqodijvwqnykh.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_33
# Graph fragment:
#   %arg14_1 : Tensor "f32[3200][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %convert_element_type_33 : Tensor "bf16[3200][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_33
triton_poi_fused__to_copy_15 = async_compile.triton('triton_poi_fused__to_copy_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ud/cudajimfnxi2sykykaxqwpzyjrcwoqww54ojojdjas5n5fgnkvnq.py
# Topologically Sorted Source Nodes: [x, x_norm, x_flat, v_t_x], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
# Source node to ATen node mapping:
#   v_t_x => convert_element_type_68
#   x => convert_element_type_65, convert_element_type_66, convolution_1
#   x_flat => view_5
#   x_norm => add, clone, convert_element_type_67, mul_7, rsqrt, sub, var_mean, view, view_1
# Graph fragment:
#   %buf5 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %getitem_1 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf7]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %convert_element_type_66, %convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_67 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [1, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, 128, -1]), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.bfloat16), kwargs = {})
#   return %convert_element_type_68
triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16 = async_compile.triton('triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 262656, 'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x1 + 4096*(((x1 % 32)) // 32)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 // 4), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 // 4), ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 4096.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 1024*y0), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vr/cvrd7r4qhrpbhuujxlkxgqckoguqlvpwtanrlhcmpgdoqgsekir7.py
# Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, mixed, out, view_4, shift_1, out_1, x_1, x_2], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mixed => unsqueeze_default_30
#   out => add_1
#   out_1 => add_2
#   shift_1 => view_4
#   split => split_with_sizes
#   view_4 => view_6
#   x => convert_element_type_65, convert_element_type_66, convolution_1
#   x_1 => mul_8, sigmoid_5
#   x_2 => convert_element_type_75
#   x_flat => view_5
#   x_norm => add, clone, convert_element_type_67, mul_7, rsqrt, sub, var_mean, view, view_1
# Graph fragment:
#   %buf5 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %getitem_1 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf7]
#   %mm_default_30 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_default_30]
#   %addmm_5 : Tensor "bf16[1, 3200][3200, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %add_2 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=add_2]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %convert_element_type_66, %convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_67 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [1, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, 128, -1]), kwargs = {})
#   %unsqueeze_default_30 : Tensor "bf16[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default_30, 0), kwargs = {})
#   %add_1 : Tensor "f32[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %unsqueeze_default_30), kwargs = {})
#   %view_6 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [1, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "bf16[1, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [1, 128, 1, 1]), kwargs = {})
#   %add_2 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   %convert_element_type_75 : Tensor "bf16[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
#   return %add_2,%convert_element_type_75
triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17 = async_compile.triton('triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 787200, 'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 // 4), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 // 4), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1 + 1024*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (3072 + y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 4096.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 128*x1), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/kx/ckxy5y3f5dvgbxddqjvxfsmolpz5qh2pp6n52bazieitdzefnq7j.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_2 => convert_element_type_74
# Graph fragment:
#   %arg28_1 : Tensor "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=arg28_1]
#   %convert_element_type_74 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_74
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rd/crdic7iloomwe7xtdh4xs6tsu2zrbsxwogc4rrvyraonoqmsg5j2.py
# Topologically Sorted Source Nodes: [x_1, x_2, add_2, h_1], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_3
#   h_1 => convert_element_type_76, convert_element_type_77, mul_9, sigmoid_6
#   x_1 => mul_8, sigmoid_5
#   x_2 => convert_element_type_73, convert_element_type_74, convert_element_type_75, convolution_2
# Graph fragment:
#   %buf35 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf35]
#   %arg29_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %convolution : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution]
#   %sigmoid_5 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   %convert_element_type_75 : Tensor "bf16[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
#   %convert_element_type_74 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_75, %convert_element_type_74, %convert_element_type_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %sigmoid_6 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_76,), kwargs = {})
#   %mul_9 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_76, %sigmoid_6), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_77
triton_poi_fused__to_copy_add_convolution_silu_19 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1049088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_19(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/n3/cn3ftsgqqtf6rzkowl3vhpexd3gucls5tx23v62baikdsj7rj3i7.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_3 => convert_element_type_92
# Graph fragment:
#   %arg34_1 : Tensor "f32[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=arg34_1]
#   %convert_element_type_92 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg34_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_92
triton_poi_fused__to_copy_20 = async_compile.triton('triton_poi_fused__to_copy_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wk/cwkyhukcl6sghehb3e3rtdcwrb3zknzpl4fuzf7pke4fxvhzcpxb.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_3 => convert_element_type_91, convert_element_type_92, convolution_5
# Graph fragment:
#   %buf53 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf53]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convert_element_type_92 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg34_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_91 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg35_1, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %convert_element_type_92, %convert_element_type_91, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_5
triton_poi_fused__to_copy_convolution_21 = async_compile.triton('triton_poi_fused__to_copy_convolution_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 394240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hu/chubbtlw66fhrp7lmnn7rtpfj2sqwl4kqfx6gspjkauxua2fytn5.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   input_16 => add_tensor_15, convert_element_type_105, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
# Graph fragment:
#   %convolution_5 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %mm_default_47 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_47]
#   %arg39_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg39_1, torch.bfloat16), kwargs = {})
#   %add_tensor_15 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_47, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[1, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [1, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   return %add_8
triton_poi_fused__to_copy_add_addmm_transpose_view_22 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_transpose_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_transpose_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 525312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_transpose_view_22(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ui/cui5pvdjvapw7ud6znd5hcq6lbokgrctmosf6jy43ojspl7o66yl.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_7 => convert_element_type_111
# Graph fragment:
#   %arg40_1 : Tensor "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=arg40_1]
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_111
triton_poi_fused__to_copy_23 = async_compile.triton('triton_poi_fused__to_copy_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/f4/cf4rctq74htp5mz6tmawohsh6isw663qjtepdww7f3albmyhow6z.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_16 => add_tensor_15, convert_element_type_105, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
#   x_norm_2 => clone_4, convert_element_type_112, var_mean_2, view_30
# Graph fragment:
#   %buf67 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf67]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg39_1, torch.bfloat16), kwargs = {})
#   %add_tensor_15 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_47, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[1, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [1, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_112 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_4 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [1, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_13,%buf69
triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24 = async_compile.triton('triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 132096}}
)
@triton.jit
def triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 8)
        r0_2 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0 + 256*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ok/cokkbkphcrvfe4vrdbhxxfy4vsdeaup5emolkwemuytdtlfqi3zy.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_46
# Graph fragment:
#   %arg17_1 : Tensor "f32[6400, 256][256, 1]cuda:0" = PlaceHolder[target=arg17_1]
#   %convert_element_type_46 : Tensor "bf16[6400, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_46
triton_poi_fused__to_copy_25 = async_compile.triton('triton_poi_fused__to_copy_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13107200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rp/crpiujdql3yin4z77r2syw5tuow3ihlb35hpy6yzgw3v4j3lplqq.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_45
# Graph fragment:
#   %arg18_1 : Tensor "f32[6400][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %convert_element_type_45 : Tensor "bf16[6400][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_45
triton_poi_fused__to_copy_26 = async_compile.triton('triton_poi_fused__to_copy_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/na/cna52aany6icl22uannf3ntbktxmthbspr2qvr6qo3sc6t26nghb.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, x_flat_2, v_t_x_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_16 => add_tensor_15, convert_element_type_105, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   v_t_x_2 => convert_element_type_113
#   x_6 => add_8
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
#   x_flat_2 => view_35
#   x_norm_2 => add_9, clone_4, convert_element_type_112, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
# Graph fragment:
#   %buf67 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf67]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %getitem_13 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf69 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf69]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg39_1, torch.bfloat16), kwargs = {})
#   %add_tensor_15 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_47, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[1, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [1, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_112 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_4 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [1, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [1, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[1, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [1, 256, -1]), kwargs = {})
#   %convert_element_type_113 : Tensor "bf16[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.bfloat16), kwargs = {})
#   return %convert_element_type_113
triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 132096, 'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1 + 4096*(((x1 % 16)) // 16)), xmask & ymask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 // 8), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 // 8), ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 2048.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7p/c7pgrwze4adrug4ndi4jlwajdjm72a5qul76eoz4qr2vkpvbh3yd.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, mixed_2, out_6, view_19, shift_5, out_7, x_8, x_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
# Source node to ATen node mapping:
#   input_16 => add_tensor_15, convert_element_type_105, view_28
#   mixed_2 => unsqueeze_default_26
#   out_5 => view_29
#   out_6 => add_10
#   out_7 => add_11
#   shift_5 => view_34
#   split_2 => split_with_sizes_2
#   transpose_8 => permute_21
#   view_19 => view_36
#   x_6 => add_8
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
#   x_8 => mul_15, sigmoid_9
#   x_9 => convert_element_type_120
#   x_flat_2 => view_35
#   x_norm_2 => add_9, clone_4, convert_element_type_112, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
# Graph fragment:
#   %buf67 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf67]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %getitem_13 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf69 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf69]
#   %mm_default_26 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_26]
#   %addmm_7 : Tensor "bf16[1, 6400][6400, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %add_11 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_11]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg39_1, torch.bfloat16), kwargs = {})
#   %add_tensor_15 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_47, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[1, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [1, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_112 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_4 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [1, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [1, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[1, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [1, 256, -1]), kwargs = {})
#   %unsqueeze_default_26 : Tensor "bf16[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default_26, 0), kwargs = {})
#   %add_10 : Tensor "f32[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %unsqueeze_default_26), kwargs = {})
#   %view_36 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [1, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "bf16[1, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [1, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   return %add_11,%convert_element_type_120
triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 394752, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 // 8), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 // 8), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (6144 + y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 2048.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 256*x1), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/to/ctos7bal6adtldyqwwrzokwdy7pamb2ubn6iycmze4ubfix2ftbf.py
# Topologically Sorted Source Nodes: [x_8, x_9, add_9, h_4], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_4 => convert_element_type_121, convert_element_type_122, mul_16, sigmoid_10
#   x_8 => mul_15, sigmoid_9
#   x_9 => convert_element_type_118, convert_element_type_119, convert_element_type_120, convolution_7
# Graph fragment:
#   %buf85 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf85]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %convolution_5 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %sigmoid_9 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %convert_element_type_119 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg42_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_118 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.bfloat16), kwargs = {})
#   %convolution_7 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %convert_element_type_119, %convert_element_type_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %convert_element_type_121 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %sigmoid_10 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_121,), kwargs = {})
#   %mul_16 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_121, %sigmoid_10), kwargs = {})
#   %convert_element_type_122 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
#   return %convert_element_type_122
triton_poi_fused__to_copy_add_convolution_silu_29 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 525312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_29(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/6s/c6sy5e7a7zmushrjpbrlipntokycpyki3u7hah3atvazo5c4h6iv.py
# Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_6 => convert_element_type_154
# Graph fragment:
#   %arg52_1 : Tensor "f32[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=arg52_1]
#   %convert_element_type_154 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg52_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_154
triton_poi_fused__to_copy_30 = async_compile.triton('triton_poi_fused__to_copy_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/v2/cv2bymna2rhvkhifeqeisfh657y2f357s5kpbyvx4zmpra5w6m34.py
# Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_6 => convert_element_type_153, convert_element_type_154, convolution_10
# Graph fragment:
#   %buf114 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf114]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %convert_element_type_154 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg52_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_153 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg53_1, torch.bfloat16), kwargs = {})
#   %convolution_10 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_152, %convert_element_type_154, %convert_element_type_153, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_10
triton_poi_fused__to_copy_convolution_31 = async_compile.triton('triton_poi_fused__to_copy_convolution_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 198656}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_31(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ql/cqlxt5gzpobhiv2dpdsr2qcso4x2qgjrythggzf6zv4qcjjmyxz7.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   input_20 => add_tensor_12, convert_element_type_167, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
# Graph fragment:
#   %convolution_10 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %mm_default_44 : Tensor "bf16[64, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_44]
#   %arg57_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg57_1, torch.bfloat16), kwargs = {})
#   %add_tensor_12 : Tensor "bf16[64, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_44, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_12, [1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [1, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   return %add_18
triton_poi_fused__to_copy_add_addmm_transpose_view_32 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_transpose_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_transpose_view_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 264192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_transpose_view_32(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/my/cmyptq7wam3n76noagvkdklhpub3fvwkolv4ptwo34wdb6lg7iyl.py
# Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_15 => convert_element_type_173
# Graph fragment:
#   %arg58_1 : Tensor "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=arg58_1]
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg58_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_173
triton_poi_fused__to_copy_33 = async_compile.triton('triton_poi_fused__to_copy_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wb/cwbve3ukhanov63qww4rgaibqgf2cbokh5mqb5qzos6wgq4mz6vx.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_20 => add_tensor_12, convert_element_type_167, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
#   x_norm_4 => clone_10, convert_element_type_174, var_mean_4, view_76
# Graph fragment:
#   %buf128 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf128]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg57_1, torch.bfloat16), kwargs = {})
#   %add_tensor_12 : Tensor "bf16[64, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_44, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_12, [1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [1, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg58_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_174 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_10 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[1, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [1, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_27,%buf130
triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_34 = async_compile.triton('triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 67584}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_34(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
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
    r0_1 = (r0_index % 16)
    r0_2 = r0_index // 16
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0 + 512*r0_2), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/om/commcjjii7iuxdw5hm5wininfiizpihjftirhjgrlqkbnvbkr5cj.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_58
# Graph fragment:
#   %arg21_1 : Tensor "f32[12800, 256][256, 1]cuda:0" = PlaceHolder[target=arg21_1]
#   %convert_element_type_58 : Tensor "bf16[12800, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg21_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_58
triton_poi_fused__to_copy_35 = async_compile.triton('triton_poi_fused__to_copy_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26214400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3276800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ld/cldhb7apejexjpuekjljncwu2fc35im2bgbjpssc2kwnpqzzyuju.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_57
# Graph fragment:
#   %arg22_1 : Tensor "f32[12800][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_57 : Tensor "bf16[12800][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_57
triton_poi_fused__to_copy_36 = async_compile.triton('triton_poi_fused__to_copy_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/mi/cmi43d2dkzvftbubp7tpkeyekzx4vsglt7ke5q25khkd4evioil3.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, x_flat_4, v_t_x_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_20 => add_tensor_12, convert_element_type_167, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   v_t_x_4 => convert_element_type_175
#   x_14 => add_18
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
#   x_flat_4 => view_81
#   x_norm_4 => add_19, clone_10, convert_element_type_174, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
# Graph fragment:
#   %buf128 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf128]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %getitem_27 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf130 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf130]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg57_1, torch.bfloat16), kwargs = {})
#   %add_tensor_12 : Tensor "bf16[64, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_44, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_12, [1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [1, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg58_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_174 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_10 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[1, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [1, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[1, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[1, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [1, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[1, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [1, 512, -1]), kwargs = {})
#   %convert_element_type_175 : Tensor "bf16[1, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_81, torch.bfloat16), kwargs = {})
#   return %convert_element_type_175
triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_37 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67584, 'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x1 + 4096*(((x1 % 8)) // 8)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 // 16), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 // 16), ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 1024.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 64*y0), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ol/colvmtevojd5wjkhdym6junv6lmgeb6kcnbewzjzurxzgd4bk4xx.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, mixed_4, out_14, view_39, shift_9, out_15, x_16, x_17], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
# Source node to ATen node mapping:
#   input_20 => add_tensor_12, convert_element_type_167, view_74
#   mixed_4 => unsqueeze_default_22
#   out_13 => view_75
#   out_14 => add_20
#   out_15 => add_21
#   shift_9 => view_80
#   split_4 => split_with_sizes_4
#   transpose_24 => permute_43
#   view_39 => view_82
#   x_14 => add_18
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
#   x_16 => mul_23, sigmoid_13
#   x_17 => convert_element_type_182
#   x_flat_4 => view_81
#   x_norm_4 => add_19, clone_10, convert_element_type_174, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
# Graph fragment:
#   %buf128 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf128]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %getitem_27 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf130 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf130]
#   %mm_default_22 : Tensor "bf16[512, 64][64, 1]cuda:0" = PlaceHolder[target=mm_default_22]
#   %addmm_9 : Tensor "bf16[1, 12800][12800, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %add_21 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=add_21]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg57_1, torch.bfloat16), kwargs = {})
#   %add_tensor_12 : Tensor "bf16[64, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_44, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_12, [1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [1, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg58_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_174 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_10 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[1, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [1, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[1, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[1, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [1, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[1, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [1, 512, -1]), kwargs = {})
#   %unsqueeze_default_22 : Tensor "bf16[1, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default_22, 0), kwargs = {})
#   %add_20 : Tensor "f32[1, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %unsqueeze_default_22), kwargs = {})
#   %view_82 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [1, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "bf16[1, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [1, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convert_element_type_182 : Tensor "bf16[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
#   return %add_21,%convert_element_type_182
triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_38 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 199680, 'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 // 16), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 // 16), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1 + 64*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (12288 + y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 1024.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 512*x1), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/jg/cjg5bailurbhugtukckbu47iwkrbyekihetsh2ir26bdp2o7nbqa.py
# Topologically Sorted Source Nodes: [x_16, x_17, add_17, h_7], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_7 => convert_element_type_183, convert_element_type_184, mul_24, sigmoid_14
#   x_16 => mul_23, sigmoid_13
#   x_17 => convert_element_type_180, convert_element_type_181, convert_element_type_182, convolution_12
# Graph fragment:
#   %buf146 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf146]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %convolution_10 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %sigmoid_13 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convert_element_type_182 : Tensor "bf16[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
#   %convert_element_type_181 : Tensor "bf16[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg60_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_180 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg61_1, torch.bfloat16), kwargs = {})
#   %convolution_12 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_182, %convert_element_type_181, %convert_element_type_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %convert_element_type_183 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.float32), kwargs = {})
#   %sigmoid_14 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_183,), kwargs = {})
#   %mul_24 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_183, %sigmoid_14), kwargs = {})
#   %convert_element_type_184 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
#   return %convert_element_type_184
triton_poi_fused__to_copy_add_convolution_silu_39 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 264192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_39(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/es/cesdfqgob3andsbwep4dzeiq27vxu3xvgf4ppmiuky77wumjnmk7.py
# Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_21 => add_27
#   h_8 => convert_element_type_213, convert_element_type_214, mul_28, sigmoid_16
#   qkv => convert_element_type_218
#   view_50 => view_106
#   x_20 => mul_27, sigmoid_15
#   x_21 => convert_element_type_210, convert_element_type_211, convert_element_type_212, convolution_14
#   x_flat_6 => permute_56
#   x_norm_6 => add_28, add_29, convert_element_type_215, mul_29, mul_30, rsqrt_6, sub_10, var_mean_6
# Graph fragment:
#   %buf170 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf170]
#   %arg69_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %convert_element_type_184 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convert_element_type_184]
#   %getitem_39 : Tensor "f32[1, 64, 1][64, 1, 64]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf172 : Tensor "f32[1, 64, 1][64, 1, 64]cuda:0" = PlaceHolder[target=buf172]
#   %arg70_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg70_1]
#   %arg71_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg71_1]
#   %sigmoid_15 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_26,), kwargs = {})
#   %mul_27 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %sigmoid_15), kwargs = {})
#   %convert_element_type_212 : Tensor "bf16[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.bfloat16), kwargs = {})
#   %convert_element_type_211 : Tensor "bf16[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg68_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_210 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg69_1, torch.bfloat16), kwargs = {})
#   %convolution_14 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_212, %convert_element_type_211, %convert_element_type_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_27 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convert_element_type_184), kwargs = {})
#   %convert_element_type_213 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_28 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [1, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_215 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_56, torch.float32), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_215, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_215, %getitem_39), kwargs = {})
#   %add_28 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_29 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_6), kwargs = {})
#   %mul_30 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %arg70_1), kwargs = {})
#   %add_29 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %arg71_1), kwargs = {})
#   %convert_element_type_218 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_29, torch.bfloat16), kwargs = {})
#   return %getitem_39,%buf172,%convert_element_type_218
triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_40 = async_compile.triton('triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 268288}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp27 = tmp10 - tmp20
    tmp28 = 512.0
    tmp29 = (tmp26 / tmp28)
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 512*x0), tmp38, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/oj/cojt7am4safnhwionw5vlme6p4j5kcj53cqmsrlcd3k4u27rzdm2.py
# Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   qkv => convert_element_type_217
# Graph fragment:
#   %arg72_1 : Tensor "f32[1536, 512][512, 1]cuda:0" = PlaceHolder[target=arg72_1]
#   %convert_element_type_217 : Tensor "bf16[1536, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg72_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_217
triton_poi_fused__to_copy_41 = async_compile.triton('triton_poi_fused__to_copy_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vn/cvnegwzby3f2fenr2bxiztpvxyjx2zsiijb6nu2ufbwxoao2g6wb.py
# Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   qkv => convert_element_type_216
# Graph fragment:
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %convert_element_type_216 : Tensor "bf16[1536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg73_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_216
triton_poi_fused__to_copy_42 = async_compile.triton('triton_poi_fused__to_copy_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/c6/cc6c4a4qovuhuxu7shcjz22pvlnzzitr62bde6n5ch6rlksf4fcm.py
# Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.addmm, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_21 => add_27
#   h_8 => convert_element_type_213, convert_element_type_214, mul_28, sigmoid_16
#   h_9 => convert_element_type_236
#   out_22 => add_tensor_9, convert_element_type_228, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, convert_element_type_233, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   view_50 => view_106
#   x_20 => mul_27, sigmoid_15
#   x_21 => convert_element_type_210, convert_element_type_211, convert_element_type_212, convolution_14
#   x_flat_6 => permute_56
# Graph fragment:
#   %buf170 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf170]
#   %arg69_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %convert_element_type_184 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convert_element_type_184]
#   %mm_default_41 : Tensor "bf16[64, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_41]
#   %arg75_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %convert_element_type_233 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_233]
#   %getitem_41 : Tensor "f32[1, 64, 1][64, 1, 64]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf188 : Tensor "f32[1, 64, 1][64, 1, 64]cuda:0" = PlaceHolder[target=buf188]
#   %arg76_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg76_1]
#   %arg77_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %sigmoid_15 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_26,), kwargs = {})
#   %mul_27 : Tensor "f32[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %sigmoid_15), kwargs = {})
#   %convert_element_type_212 : Tensor "bf16[1, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.bfloat16), kwargs = {})
#   %convert_element_type_211 : Tensor "bf16[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg68_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_210 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg69_1, torch.bfloat16), kwargs = {})
#   %convolution_14 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_212, %convert_element_type_211, %convert_element_type_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_27 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convert_element_type_184), kwargs = {})
#   %convert_element_type_213 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_28 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [1, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_228 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg75_1, torch.bfloat16), kwargs = {})
#   %add_tensor_9 : Tensor "bf16[64, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_41, %convert_element_type_228), kwargs = {})
#   %view_118 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_9, [1, 64, 512]), kwargs = {})
#   %add_30 : Tensor "bf16[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %convert_element_type_233 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_233, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [1, 512, 8, 8]), kwargs = {})
#   %convert_element_type_236 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_119, torch.bfloat16), kwargs = {})
#   return %convert_element_type_233,%getitem_41,%buf188,%convert_element_type_236
triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_43 = async_compile.triton('triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 335872}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp9 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = (tmp22 / tmp24)
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = (tmp31 / tmp33)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 512*x0), tmp43, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/nn/cnngtpkpf6k6dlzi5ffjf5krw3dlrj5yzxgvjsesaubeurgwq3ji.py
# Topologically Sorted Source Nodes: [h_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_9 => convert_element_type_235
# Graph fragment:
#   %arg78_1 : Tensor "f32[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0" = PlaceHolder[target=arg78_1]
#   %convert_element_type_235 : Tensor "bf16[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg78_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_235
triton_poi_fused__to_copy_44 = async_compile.triton('triton_poi_fused__to_copy_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/on/conkjxbmkugamkq25imoibnxlbdnxbp24lu4r4wgfij4bvmyg5ie.py
# Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   gate => unsqueeze_7
#   h_10 => add_33
#   h_16_gated => mul_35
#   h_9 => convert_element_type_234, convert_element_type_235, convert_element_type_236, convolution_15
#   input_26 => add_tensor_7, convert_element_type_244
#   input_27 => sigmoid_18
#   out_24 => add_31, add_32, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   unsqueeze_4 => unsqueeze_6
# Graph fragment:
#   %buf194 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf194]
#   %arg79_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %convert_element_type_152 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convert_element_type_152]
#   %mm_default_39 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_39]
#   %arg83_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_233, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [1, 512, 8, 8]), kwargs = {})
#   %convert_element_type_236 : Tensor "bf16[1, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_119, torch.bfloat16), kwargs = {})
#   %convert_element_type_235 : Tensor "bf16[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg78_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_234 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg79_1, torch.bfloat16), kwargs = {})
#   %convolution_15 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_236, %convert_element_type_235, %convert_element_type_234, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %convert_element_type_244 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg83_1, torch.bfloat16), kwargs = {})
#   %add_tensor_7 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_39, %convert_element_type_244), kwargs = {})
#   %sigmoid_18 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_7,), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_18, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %mul_35 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_152, %unsqueeze_7), kwargs = {})
#   %add_33 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %mul_35), kwargs = {})
#   return %add_33
triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_45 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 262144, 'x': 264704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp4 * tmp9
    tmp11 = tmp3 + tmp10
    tl.store(out_ptr0 + (y0 + 256*x1), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/js/cjstya2hlk5eyqgfsu75vpdsueeiew7fktigla7baqds2mcjfrh2.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   input_28 => add_tensor_6, convert_element_type_261, view_134
#   out_27 => view_135
#   transpose_44 => permute_74
#   x_22 => add_34
# Graph fragment:
#   %add_33 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_default_38 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_38]
#   %arg87_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %convert_element_type_261 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg87_1, torch.bfloat16), kwargs = {})
#   %add_tensor_6 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_38, %convert_element_type_261), kwargs = {})
#   %view_134 : Tensor "bf16[1, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_6, [1, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "bf16[1, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [1, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   return %add_34
triton_poi_fused__to_copy_add_addmm_transpose_view_46 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_transpose_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_transpose_view_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 131072, 'x': 394240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_transpose_view_46(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(in_out_ptr0 + (x1 + 256*y0), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/5i/c5ijymo3l4illczxkddacufagbhng5s6cuqv3ogwcvlvzgpuiweh.py
# Topologically Sorted Source Nodes: [x_24, x_25, add_27, h_11], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_27 => add_38
#   h_11 => convert_element_type_277, convert_element_type_278, mul_39, sigmoid_20
#   x_24 => mul_38, sigmoid_19
#   x_25 => convert_element_type_274, convert_element_type_275, convert_element_type_276, convolution_17
# Graph fragment:
#   %buf225 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf225]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %sigmoid_19 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convert_element_type_276 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_38, torch.bfloat16), kwargs = {})
#   %convert_element_type_275 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg90_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_274 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg91_1, torch.bfloat16), kwargs = {})
#   %convolution_17 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_276, %convert_element_type_275, %convert_element_type_274, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %convert_element_type_277 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_38, torch.float32), kwargs = {})
#   %sigmoid_20 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_277,), kwargs = {})
#   %mul_39 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_277, %sigmoid_20), kwargs = {})
#   %convert_element_type_278 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.bfloat16), kwargs = {})
#   return %convert_element_type_278
triton_poi_fused__to_copy_add_convolution_silu_47 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 131072, 'x': 394240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_47(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_out_ptr0 + (x1 + 256*y0), xmask & ymask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (y0 + 256*x1), xmask & ymask).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x1 + 256*y0), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/jx/cjxmxsrongff7opdpptloewkpjd6abzxjsqvv6mubghxtie65qbf.py
# Topologically Sorted Source Nodes: [h_16], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_16 => convert_element_type_400
# Graph fragment:
#   %arg124_1 : Tensor "f32[256, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=arg124_1]
#   %convert_element_type_400 : Tensor "bf16[256, 128, 4, 4][2048, 1, 512, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg124_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_400
triton_poi_fused__to_copy_48 = async_compile.triton('triton_poi_fused__to_copy_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rm/crmdqdtkm44h5en2qq2thfro47cvxfknnpioi5jrn5i4y5vnzslf.py
# Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_40 => convert_element_type_409
# Graph fragment:
#   %arg128_1 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=arg128_1]
#   %convert_element_type_409 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg128_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_409
triton_poi_fused__to_copy_49 = async_compile.triton('triton_poi_fused__to_copy_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ha/chacwuyzgd4yshq6dlch6c4plyqg3ydyntwzyvq2c3n4c4aykrbl.py
# Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   add_43 => add_58
#   gate_1 => unsqueeze_14
#   h_15 => convert_element_type_397, convert_element_type_398, mul_55, sigmoid_28
#   h_16 => convert_element_type_399, convert_element_type_400, convolution_26
#   h_17 => add_59
#   h_32_gated => mul_57
#   input_40 => add_tensor, convert_element_type_408
#   input_41 => sigmoid_30
#   unsqueeze_11 => unsqueeze_13
#   x_40 => mul_54, sigmoid_27
#   x_41 => convert_element_type_394, convert_element_type_395, convert_element_type_396, convolution_25
# Graph fragment:
#   %buf330 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf330]
#   %arg125_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %convert_element_type_90 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_90]
#   %mm_default_32 : Tensor "bf16[1, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_32]
#   %arg129_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg129_1]
#   %sigmoid_27 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convert_element_type_396 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_54, torch.bfloat16), kwargs = {})
#   %convert_element_type_395 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg122_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_394 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg123_1, torch.bfloat16), kwargs = {})
#   %convolution_25 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_396, %convert_element_type_395, %convert_element_type_394, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convert_element_type_368), kwargs = {})
#   %convert_element_type_397 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_58, torch.float32), kwargs = {})
#   %sigmoid_28 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_397,), kwargs = {})
#   %mul_55 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_397, %sigmoid_28), kwargs = {})
#   %convert_element_type_398 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.bfloat16), kwargs = {})
#   %convert_element_type_400 : Tensor "bf16[256, 128, 4, 4][2048, 1, 512, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg124_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_399 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg125_1, torch.bfloat16), kwargs = {})
#   %convolution_26 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_398, %convert_element_type_400, %convert_element_type_399, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %convert_element_type_408 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg129_1, torch.bfloat16), kwargs = {})
#   %add_tensor : Tensor "bf16[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_32, %convert_element_type_408), kwargs = {})
#   %sigmoid_30 : Tensor "bf16[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor,), kwargs = {})
#   %unsqueeze_13 : Tensor "bf16[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_30, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_57 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_90, %unsqueeze_14), kwargs = {})
#   %add_59 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %mul_57), kwargs = {})
#   return %add_59
triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_50 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1049856}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp4 * tmp9
    tmp11 = tmp3 + tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/zv/czvescwfjs7ft6ypm3poi75wt3uz3qqbenkw4uvuazkduek55hcu.py
# Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_59 => add_79
#   h_22 => convert_element_type_476, mul_72, sigmoid_40
#   input_42 => clone_35, var_mean_18, view_270
#   x_55 => mul_71, sigmoid_39
#   x_56 => convert_element_type_473, convert_element_type_474, convert_element_type_475, convolution_36
# Graph fragment:
#   %buf400 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf400]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %convert_element_type_464 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_464]
#   %sigmoid_39 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convert_element_type_475 : Tensor "bf16[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_71, torch.bfloat16), kwargs = {})
#   %convert_element_type_474 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg148_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_473 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg149_1, torch.bfloat16), kwargs = {})
#   %convolution_36 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_475, %convert_element_type_474, %convert_element_type_473, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %convert_element_type_464), kwargs = {})
#   %convert_element_type_476 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_72 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_35 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_72,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [1, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_103,%buf402
triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_51 = async_compile.triton('triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 524800}}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_51(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 4)
        r0_2 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4*x0 + 128*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 4*x0 + 128*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.sigmoid(tmp6)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/nc/cncmkaywqpffmaedb53jwzk4ukjcsuu66is5xjfad22qjltiedfa.py
# Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42, input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_59 => add_79
#   h_22 => convert_element_type_476, mul_72, sigmoid_40
#   input_42 => add_80, add_81, clone_35, mul_73, mul_74, rsqrt_18, sub_28, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, var_mean_18, view_270, view_271
#   input_43 => mul_75, sigmoid_41
#   input_44 => convert_element_type_481
#   x_55 => mul_71, sigmoid_39
#   x_56 => convert_element_type_473, convert_element_type_474, convert_element_type_475, convolution_36
# Graph fragment:
#   %buf400 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf400]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %convert_element_type_464 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_464]
#   %getitem_103 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_103]
#   %buf402 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf402]
#   %arg150_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg150_1]
#   %arg151_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg151_1]
#   %add_81 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_81]
#   %sigmoid_39 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convert_element_type_475 : Tensor "bf16[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_71, torch.bfloat16), kwargs = {})
#   %convert_element_type_474 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg148_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_473 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg149_1, torch.bfloat16), kwargs = {})
#   %convolution_36 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_475, %convert_element_type_474, %convert_element_type_473, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "bf16[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %convert_element_type_464), kwargs = {})
#   %convert_element_type_476 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_72 : Tensor "f32[1, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_35 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_72,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [1, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %getitem_103), kwargs = {})
#   %add_80 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_102, 1e-05), kwargs = {})
#   %rsqrt_18 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   %mul_73 : Tensor "f32[1, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_18), kwargs = {})
#   %view_271 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_73, [1, 128, 32, 32]), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, 0), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, 2), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 3), kwargs = {})
#   %mul_74 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_271, %unsqueeze_17), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg151_1, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 3), kwargs = {})
#   %add_81 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_20), kwargs = {})
#   %sigmoid_41 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.bfloat16), kwargs = {})
#   return %add_81,%convert_element_type_481
triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_52 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1050112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0 // 4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0 // 4), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 4096.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/bs/cbs5mx7pdeaqea5u3x4c2e3xoae2q6lsyjpncroip4lcc2vzsfl4.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_75, sigmoid_41
#   input_44 => convert_element_type_479, convert_element_type_480, convert_element_type_481, convolution_37
# Graph fragment:
#   %buf407 : Tensor "bf16[1, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf407]
#   %arg153_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %sigmoid_41 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[1, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.bfloat16), kwargs = {})
#   %convert_element_type_480 : Tensor "bf16[4, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg152_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg153_1, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[1, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_37
triton_poi_fused__to_copy_convolution_silu_53 = async_compile.triton('triton_poi_fused__to_copy_convolution_silu_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_silu_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_silu_53(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x2), tmp3, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, ), (1, ))
        assert_size_stride(arg1_1, (1024, 256), (256, 1))
        assert_size_stride(arg2_1, (1024, ), (1, ))
        assert_size_stride(arg3_1, (256, 1024), (1024, 1))
        assert_size_stride(arg4_1, (256, ), (1, ))
        assert_size_stride(arg5_1, (4, 256), (256, 1))
        assert_size_stride(arg6_1, (1, ), (1, ))
        assert_size_stride(arg7_1, (512, 512), (512, 1))
        assert_size_stride(arg8_1, (512, ), (1, ))
        assert_size_stride(arg9_1, (256, 512), (512, 1))
        assert_size_stride(arg10_1, (256, ), (1, ))
        assert_size_stride(arg11_1, (256, 256), (256, 1))
        assert_size_stride(arg12_1, (256, ), (1, ))
        assert_size_stride(arg13_1, (3200, 256), (256, 1))
        assert_size_stride(arg14_1, (3200, ), (1, ))
        assert_size_stride(arg15_1, (256, 256), (256, 1))
        assert_size_stride(arg16_1, (256, ), (1, ))
        assert_size_stride(arg17_1, (6400, 256), (256, 1))
        assert_size_stride(arg18_1, (6400, ), (1, ))
        assert_size_stride(arg19_1, (256, 256), (256, 1))
        assert_size_stride(arg20_1, (256, ), (1, ))
        assert_size_stride(arg21_1, (12800, 256), (256, 1))
        assert_size_stride(arg22_1, (12800, ), (1, ))
        assert_size_stride(arg23_1, (128, 4, 3, 3), (36, 1, 12, 4))
        assert_size_stride(arg24_1, (128, ), (1, ))
        assert_size_stride(arg25_1, (1, 4, 32, 32), (4096, 1024, 32, 1))
        assert_size_stride(arg26_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg27_1, (128, ), (1, ))
        assert_size_stride(arg28_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg29_1, (128, ), (1, ))
        assert_size_stride(arg30_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg31_1, (128, ), (1, ))
        assert_size_stride(arg32_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg33_1, (128, ), (1, ))
        assert_size_stride(arg34_1, (256, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg35_1, (256, ), (1, ))
        assert_size_stride(arg36_1, (256, 256), (256, 1))
        assert_size_stride(arg37_1, (512, 256), (256, 1))
        assert_size_stride(arg38_1, (256, 256), (256, 1))
        assert_size_stride(arg39_1, (256, ), (1, ))
        assert_size_stride(arg40_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg41_1, (256, ), (1, ))
        assert_size_stride(arg42_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg43_1, (256, ), (1, ))
        assert_size_stride(arg44_1, (256, 256), (256, 1))
        assert_size_stride(arg45_1, (512, 256), (256, 1))
        assert_size_stride(arg46_1, (256, 256), (256, 1))
        assert_size_stride(arg47_1, (256, ), (1, ))
        assert_size_stride(arg48_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg49_1, (256, ), (1, ))
        assert_size_stride(arg50_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg51_1, (256, ), (1, ))
        assert_size_stride(arg52_1, (512, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg53_1, (512, ), (1, ))
        assert_size_stride(arg54_1, (512, 512), (512, 1))
        assert_size_stride(arg55_1, (1024, 256), (256, 1))
        assert_size_stride(arg56_1, (512, 512), (512, 1))
        assert_size_stride(arg57_1, (512, ), (1, ))
        assert_size_stride(arg58_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(arg59_1, (512, ), (1, ))
        assert_size_stride(arg60_1, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(arg61_1, (512, ), (1, ))
        assert_size_stride(arg62_1, (512, 512), (512, 1))
        assert_size_stride(arg63_1, (1024, 256), (256, 1))
        assert_size_stride(arg64_1, (512, 512), (512, 1))
        assert_size_stride(arg65_1, (512, ), (1, ))
        assert_size_stride(arg66_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(arg67_1, (512, ), (1, ))
        assert_size_stride(arg68_1, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(arg69_1, (512, ), (1, ))
        assert_size_stride(arg70_1, (512, ), (1, ))
        assert_size_stride(arg71_1, (512, ), (1, ))
        assert_size_stride(arg72_1, (1536, 512), (512, 1))
        assert_size_stride(arg73_1, (1536, ), (1, ))
        assert_size_stride(arg74_1, (512, 512), (512, 1))
        assert_size_stride(arg75_1, (512, ), (1, ))
        assert_size_stride(arg76_1, (512, ), (1, ))
        assert_size_stride(arg77_1, (512, ), (1, ))
        assert_size_stride(arg78_1, (512, 256, 4, 4), (4096, 1, 1024, 256))
        assert_size_stride(arg79_1, (256, ), (1, ))
        assert_size_stride(arg80_1, (256, 256), (256, 1))
        assert_size_stride(arg81_1, (256, ), (1, ))
        assert_size_stride(arg82_1, (256, 256), (256, 1))
        assert_size_stride(arg83_1, (256, ), (1, ))
        assert_size_stride(arg84_1, (256, 256), (256, 1))
        assert_size_stride(arg85_1, (512, 256), (256, 1))
        assert_size_stride(arg86_1, (256, 256), (256, 1))
        assert_size_stride(arg87_1, (256, ), (1, ))
        assert_size_stride(arg88_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg89_1, (256, ), (1, ))
        assert_size_stride(arg90_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg91_1, (256, ), (1, ))
        assert_size_stride(arg92_1, (256, 256), (256, 1))
        assert_size_stride(arg93_1, (512, 256), (256, 1))
        assert_size_stride(arg94_1, (256, 256), (256, 1))
        assert_size_stride(arg95_1, (256, ), (1, ))
        assert_size_stride(arg96_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg97_1, (256, ), (1, ))
        assert_size_stride(arg98_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg99_1, (256, ), (1, ))
        assert_size_stride(arg100_1, (256, 256), (256, 1))
        assert_size_stride(arg101_1, (512, 256), (256, 1))
        assert_size_stride(arg102_1, (256, 256), (256, 1))
        assert_size_stride(arg103_1, (256, ), (1, ))
        assert_size_stride(arg104_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg105_1, (256, ), (1, ))
        assert_size_stride(arg106_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg107_1, (256, ), (1, ))
        assert_size_stride(arg108_1, (256, 256), (256, 1))
        assert_size_stride(arg109_1, (512, 256), (256, 1))
        assert_size_stride(arg110_1, (256, 256), (256, 1))
        assert_size_stride(arg111_1, (256, ), (1, ))
        assert_size_stride(arg112_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg113_1, (256, ), (1, ))
        assert_size_stride(arg114_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg115_1, (256, ), (1, ))
        assert_size_stride(arg116_1, (256, 256), (256, 1))
        assert_size_stride(arg117_1, (512, 256), (256, 1))
        assert_size_stride(arg118_1, (256, 256), (256, 1))
        assert_size_stride(arg119_1, (256, ), (1, ))
        assert_size_stride(arg120_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg121_1, (256, ), (1, ))
        assert_size_stride(arg122_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg123_1, (256, ), (1, ))
        assert_size_stride(arg124_1, (256, 128, 4, 4), (2048, 1, 512, 128))
        assert_size_stride(arg125_1, (128, ), (1, ))
        assert_size_stride(arg126_1, (256, 256), (256, 1))
        assert_size_stride(arg127_1, (256, ), (1, ))
        assert_size_stride(arg128_1, (128, 256), (256, 1))
        assert_size_stride(arg129_1, (128, ), (1, ))
        assert_size_stride(arg130_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg131_1, (128, ), (1, ))
        assert_size_stride(arg132_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg133_1, (128, ), (1, ))
        assert_size_stride(arg134_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg135_1, (128, ), (1, ))
        assert_size_stride(arg136_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg137_1, (128, ), (1, ))
        assert_size_stride(arg138_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg139_1, (128, ), (1, ))
        assert_size_stride(arg140_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg141_1, (128, ), (1, ))
        assert_size_stride(arg142_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg143_1, (128, ), (1, ))
        assert_size_stride(arg144_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg145_1, (128, ), (1, ))
        assert_size_stride(arg146_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg147_1, (128, ), (1, ))
        assert_size_stride(arg148_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg149_1, (128, ), (1, ))
        assert_size_stride(arg150_1, (128, ), (1, ))
        assert_size_stride(arg151_1, (128, ), (1, ))
        assert_size_stride(arg152_1, (4, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg153_1, (4, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 4, 32, 32), (4096, 1, 128, 4), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg25_1, buf0, 4, 1024, stream=stream0)
            del arg25_1
            buf1 = empty_strided_cuda((128, 4, 3, 3), (36, 1, 12, 4), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg23_1, buf1, 4608, stream=stream0)
            del arg23_1
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(buf3, arg24_1, 131072, stream=stream0)
            del arg24_1
            buf4 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(arg26_1, buf4, 147456, stream=stream0)
            del arg26_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf5, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf6 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
            buf7 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_4.run(buf5, arg27_1, buf6, buf7, 32, 4096, stream=stream0)
            buf9 = empty_strided_cuda((1, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_5.run(arg0_1, buf9, 256, stream=stream0)
            del arg0_1
            buf10 = empty_strided_cuda((1024, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg1_1, buf10, 262144, stream=stream0)
            del arg1_1
            buf11 = empty_strided_cuda((1, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf9, reinterpret_tensor(buf10, (256, 1024), (1, 256), 0), out=buf11)
            buf12 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_7.run(buf12, arg2_1, 1024, stream=stream0)
            del arg2_1
            buf13 = reinterpret_tensor(buf10, (256, 1024), (1024, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg3_1, buf13, 262144, stream=stream0)
            del arg3_1
            buf14 = reinterpret_tensor(buf9, (256, ), (1, ), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(arg4_1, buf14, 256, stream=stream0)
            del arg4_1
            buf15 = empty_strided_cuda((1, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_3, input_1, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf14, buf12, reinterpret_tensor(buf13, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf15)
            buf16 = empty_strided_cuda((1, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_embedding_9.run(buf15, arg6_1, arg5_1, buf16, 512, stream=stream0)
            del arg5_1
            del arg6_1
            buf17 = reinterpret_tensor(buf13, (512, 512), (512, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg7_1, buf17, 262144, stream=stream0)
            del arg7_1
            buf18 = empty_strided_cuda((1, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf16, reinterpret_tensor(buf17, (512, 512), (1, 512), 0), out=buf18)
            del buf16
            buf19 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_10.run(buf19, arg8_1, 512, stream=stream0)
            del arg8_1
            buf20 = empty_strided_cuda((256, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg9_1, buf20, 131072, stream=stream0)
            del arg9_1
            buf21 = reinterpret_tensor(buf15, (256, ), (1, ), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(arg10_1, buf21, 256, stream=stream0)
            del arg10_1
            buf22 = reinterpret_tensor(buf14, (1, 256), (256, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [input_6, input_4, input_5], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf21, buf19, reinterpret_tensor(buf20, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf22)
            buf23 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg11_1, buf23, 65536, stream=stream0)
            del arg11_1
            buf24 = reinterpret_tensor(buf21, (1, 256), (256, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf23, (256, 256), (1, 256), 0), out=buf24)
            buf25 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_13.run(buf25, arg12_1, 256, stream=stream0)
            del arg12_1
            buf26 = empty_strided_cuda((3200, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_14.run(arg13_1, buf26, 819200, stream=stream0)
            del arg13_1
            buf27 = empty_strided_cuda((3200, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(arg14_1, buf27, 3200, stream=stream0)
            del arg14_1
            buf28 = empty_strided_cuda((1, 3200), (3200, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9, input_7, input_8], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf27, buf25, reinterpret_tensor(buf26, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf28)
            del buf26
            del buf27
            buf29 = reinterpret_tensor(buf20, (1, 128, 1024), (131072, 1024, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [x, x_norm, x_flat, v_t_x], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16.run(buf5, arg27_1, buf6, buf7, buf29, 128, 1024, stream=stream0)
            buf30 = empty_strided_cuda((12, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, x_norm, split, v_1, transpose, v_t_x, x_flat], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (12, 128), (1, 12), 1536), reinterpret_tensor(buf29, (128, 1024), (1024, 1), 0), out=buf30)
            buf31 = reinterpret_tensor(buf29, (128, 1024), (1024, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed, v_t_x], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 12), (12, 1), 0), buf30, out=buf31)
            buf33 = empty_strided_cuda((1, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, mixed, out, view_4, shift_1, out_1, x_1, x_2], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17.run(buf5, arg27_1, buf6, buf7, buf31, buf28, buf33, 128, 1024, stream=stream0)
            del arg27_1
            del buf31
            del buf5
            buf34 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg28_1, buf34, 16384, stream=stream0)
            del arg28_1
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf35 = extern_kernels.convolution(buf33, buf34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf35, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf36 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [x_1, x_2, add_2, h_1], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_19.run(buf36, arg29_1, buf3, 131072, stream=stream0)
            del arg29_1
            buf37 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(arg30_1, buf37, 147456, stream=stream0)
            del arg30_1
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.convolution]
            buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf38, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf37
            buf39 = buf7; del buf7  # reuse
            buf40 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_4.run(buf38, arg31_1, buf39, buf40, 32, 4096, stream=stream0)
            buf42 = reinterpret_tensor(buf3, (512, 256), (256, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg37_1, buf42, 131072, stream=stream0)
            del arg37_1
            buf43 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf42, (256, 512), (1, 256), 0), out=buf43)
            buf44 = reinterpret_tensor(buf42, (1, 128, 1024), (131072, 1024, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1, x_flat_1, v_t_x_1], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16.run(buf38, arg31_1, buf39, buf40, buf44, 128, 1024, stream=stream0)
            buf45 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, v_3, transpose_1, v_t_x_1, x_flat_1], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (12, 128), (1, 12), 1536), reinterpret_tensor(buf44, (128, 1024), (1024, 1), 0), out=buf45)
            buf46 = reinterpret_tensor(buf44, (128, 1024), (1024, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [split_1, u_3, mixed_1, v_t_x_1], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 12), (12, 1), 0), buf45, out=buf46)
            del buf45
            buf48 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, mixed_1, out_2, view_9, shift_3, out_3, x_4, x_5], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17.run(buf38, arg31_1, buf39, buf40, buf46, buf28, buf48, 128, 1024, stream=stream0)
            del arg31_1
            del buf38
            del buf46
            buf49 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg32_1, buf49, 16384, stream=stream0)
            del arg32_1
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf50, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf48
            del buf49
            buf51 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5, add_5, h_2], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_19.run(buf51, arg33_1, buf36, 131072, stream=stream0)
            del arg33_1
            buf52 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_20.run(arg34_1, buf52, 294912, stream=stream0)
            del arg34_1
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            buf53 = extern_kernels.convolution(buf51, buf52, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf53, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf52
            buf54 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_21.run(buf54, arg35_1, 65536, stream=stream0)
            del arg35_1
            buf55 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [q_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg36_1, buf55, 65536, stream=stream0)
            del arg36_1
            buf56 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf54, (256, 256), (256, 1), 0), reinterpret_tensor(buf55, (256, 256), (1, 256), 0), out=buf56)
            # Topologically Sorted Source Nodes: [kv, chunk, q_1, view_11, q_2, view_12, k_1, view_13, v_5], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf57 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf56, (1, 4, 256, 64), (65536, 64, 256, 1), 0), reinterpret_tensor(buf43, (1, 4, 1, 64), (512, 64, 512, 1), 0), reinterpret_tensor(buf43, (1, 4, 1, 64), (512, 64, 512, 1), 256), scale=0.125)
            buf58 = buf57[0]
            assert_size_stride(buf58, (1, 4, 256, 64), (65536, 64, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf58, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf57
            buf63 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg38_1, buf63, 65536, stream=stream0)
            del arg38_1
            buf64 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, out_4, input_16], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf58, (256, 256), (256, 1), 0), reinterpret_tensor(buf63, (256, 256), (1, 256), 0), out=buf64)
            del buf58
            buf65 = reinterpret_tensor(buf64, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_22.run(buf65, buf54, arg39_1, 65536, stream=stream0)
            del arg39_1
            buf66 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg40_1, buf66, 589824, stream=stream0)
            del arg40_1
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf67 = extern_kernels.convolution(buf65, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf67, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf68 = buf40; del buf40  # reuse
            buf69 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24.run(buf67, arg41_1, buf68, buf69, 32, 2048, stream=stream0)
            buf71 = reinterpret_tensor(buf65, (256, 256), (256, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg15_1, buf71, 65536, stream=stream0)
            del arg15_1
            buf72 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf71, (256, 256), (1, 256), 0), out=buf72)
            buf73 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_13.run(buf73, arg16_1, 256, stream=stream0)
            del arg16_1
            buf74 = empty_strided_cuda((6400, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(arg17_1, buf74, 1638400, stream=stream0)
            del arg17_1
            buf75 = empty_strided_cuda((6400, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_26.run(arg18_1, buf75, 6400, stream=stream0)
            del arg18_1
            buf76 = empty_strided_cuda((1, 6400), (6400, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12, input_10, input_11], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf75, buf73, reinterpret_tensor(buf74, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf76)
            del buf74
            del buf75
            buf77 = reinterpret_tensor(buf36, (512, 256), (256, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg45_1, buf77, 131072, stream=stream0)
            del arg45_1
            buf78 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf77, (256, 512), (1, 256), 0), out=buf78)
            del buf77
            buf79 = reinterpret_tensor(buf71, (1, 256, 256), (65536, 256, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, x_flat_2, v_t_x_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27.run(buf67, arg41_1, buf68, buf69, buf79, 256, 256, stream=stream0)
            buf80 = empty_strided_cuda((12, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, v_7, transpose_9, v_t_x_2, x_flat_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (12, 256), (1, 12), 3072), reinterpret_tensor(buf79, (256, 256), (256, 1), 0), out=buf80)
            buf81 = reinterpret_tensor(buf79, (256, 256), (256, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_2, v_t_x_2], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (256, 12), (12, 1), 0), buf80, out=buf81)
            buf83 = reinterpret_tensor(buf63, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, mixed_2, out_6, view_19, shift_5, out_7, x_8, x_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf67, arg41_1, buf68, buf69, buf81, buf76, buf83, 256, 256, stream=stream0)
            del arg41_1
            del buf67
            buf84 = reinterpret_tensor(buf81, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg42_1, buf84, 65536, stream=stream0)
            del arg42_1
            # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf85 = extern_kernels.convolution(buf83, buf84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf85, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf83
            buf86 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [x_8, x_9, add_9, h_4], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_29.run(buf86, arg43_1, buf54, 65536, stream=stream0)
            del arg43_1
            buf87 = reinterpret_tensor(buf54, (256, 256), (256, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [q_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg44_1, buf87, 65536, stream=stream0)
            del arg44_1
            buf88 = reinterpret_tensor(buf84, (256, 256), (256, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf86, (256, 256), (256, 1), 0), reinterpret_tensor(buf87, (256, 256), (1, 256), 0), out=buf88)
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, q_4, view_21, q_5, view_22, k_3, view_23, v_9], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf89 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf88, (1, 4, 256, 64), (65536, 64, 256, 1), 0), reinterpret_tensor(buf78, (1, 4, 1, 64), (512, 64, 512, 1), 0), reinterpret_tensor(buf78, (1, 4, 1, 64), (512, 64, 512, 1), 256), scale=0.125)
            del buf78
            buf90 = buf89[0]
            assert_size_stride(buf90, (1, 4, 256, 64), (65536, 64, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf90, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf89
            buf95 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg46_1, buf95, 65536, stream=stream0)
            del arg46_1
            buf96 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [transpose_15, out_8, input_18], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf90, (256, 256), (256, 1), 0), reinterpret_tensor(buf95, (256, 256), (1, 256), 0), out=buf96)
            del buf90
            buf97 = reinterpret_tensor(buf96, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_22.run(buf97, buf86, arg47_1, 65536, stream=stream0)
            del arg47_1
            buf98 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg48_1, buf98, 589824, stream=stream0)
            del arg48_1
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf99 = extern_kernels.convolution(buf97, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf99, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf98
            buf100 = buf69; del buf69  # reuse
            buf101 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24.run(buf99, arg49_1, buf100, buf101, 32, 2048, stream=stream0)
            buf103 = reinterpret_tensor(buf17, (1024, 256), (256, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg55_1, buf103, 262144, stream=stream0)
            del arg55_1
            buf104 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf103, (256, 1024), (1, 256), 0), out=buf104)
            buf105 = reinterpret_tensor(buf97, (1, 256, 256), (65536, 256, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3, x_flat_3, v_t_x_3], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27.run(buf99, arg49_1, buf100, buf101, buf105, 256, 256, stream=stream0)
            buf106 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, v_11, transpose_17, v_t_x_3, x_flat_3], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (12, 256), (1, 12), 3072), reinterpret_tensor(buf105, (256, 256), (256, 1), 0), out=buf106)
            buf107 = reinterpret_tensor(buf105, (256, 256), (256, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [split_3, u_7, mixed_3, v_t_x_3], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (256, 12), (12, 1), 0), buf106, out=buf107)
            del buf106
            buf109 = reinterpret_tensor(buf95, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, mixed_3, out_10, view_29, shift_7, out_11, x_12, x_13], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf99, arg49_1, buf100, buf101, buf107, buf76, buf109, 256, 256, stream=stream0)
            del arg49_1
            del buf107
            buf110 = reinterpret_tensor(buf99, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg50_1, buf110, 65536, stream=stream0)
            del arg50_1
            # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf111 = extern_kernels.convolution(buf109, buf110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf111, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf109
            del buf110
            buf112 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13, add_13, h_5], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_29.run(buf112, arg51_1, buf86, 65536, stream=stream0)
            del arg51_1
            buf113 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg52_1, buf113, 1179648, stream=stream0)
            del arg52_1
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            buf114 = extern_kernels.convolution(buf112, buf113, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf114, (1, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf113
            buf115 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_31.run(buf115, arg53_1, 32768, stream=stream0)
            del arg53_1
            buf116 = reinterpret_tensor(buf103, (512, 512), (512, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [q_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg54_1, buf116, 262144, stream=stream0)
            del arg54_1
            buf117 = empty_strided_cuda((64, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_6, view_30, q_6, q_7], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf115, (64, 512), (512, 1), 0), reinterpret_tensor(buf116, (512, 512), (1, 512), 0), out=buf117)
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, q_7, view_31, q_8, view_32, k_5, view_33, v_13], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf118 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf117, (1, 8, 64, 64), (32768, 64, 512, 1), 0), reinterpret_tensor(buf104, (1, 8, 1, 64), (1024, 64, 1024, 1), 0), reinterpret_tensor(buf104, (1, 8, 1, 64), (1024, 64, 1024, 1), 512), scale=0.125)
            del buf104
            buf119 = buf118[0]
            assert_size_stride(buf119, (1, 8, 64, 64), (32768, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf119, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf118
            buf124 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg56_1, buf124, 262144, stream=stream0)
            del arg56_1
            buf125 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [transpose_23, out_12, input_20], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf119, (64, 512), (512, 1), 0), reinterpret_tensor(buf124, (512, 512), (1, 512), 0), out=buf125)
            del buf119
            del buf124
            buf126 = reinterpret_tensor(buf125, (1, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_32.run(buf126, buf115, arg57_1, 32768, stream=stream0)
            del arg57_1
            buf127 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_33.run(arg58_1, buf127, 2359296, stream=stream0)
            del arg58_1
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf128 = extern_kernels.convolution(buf126, buf127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf128, (1, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf126
            del buf127
            buf129 = buf101; del buf101  # reuse
            buf130 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_34.run(buf128, arg59_1, buf129, buf130, 32, 1024, stream=stream0)
            buf132 = reinterpret_tensor(buf86, (256, 256), (256, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg19_1, buf132, 65536, stream=stream0)
            del arg19_1
            buf133 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf132, (256, 256), (1, 256), 0), out=buf133)
            del buf132
            buf134 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_13.run(buf134, arg20_1, 256, stream=stream0)
            del arg20_1
            buf135 = empty_strided_cuda((12800, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_35.run(arg21_1, buf135, 3276800, stream=stream0)
            del arg21_1
            buf136 = empty_strided_cuda((12800, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(arg22_1, buf136, 12800, stream=stream0)
            del arg22_1
            buf137 = empty_strided_cuda((1, 12800), (12800, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15, input_13, input_14], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf136, buf134, reinterpret_tensor(buf135, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf137)
            del buf135
            del buf136
            buf138 = empty_strided_cuda((1024, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg63_1, buf138, 262144, stream=stream0)
            del arg63_1
            buf139 = empty_strided_cuda((1, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf138, (256, 1024), (1, 256), 0), out=buf139)
            buf140 = empty_strided_cuda((1, 512, 64), (32768, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, x_flat_4, v_t_x_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_37.run(buf128, arg59_1, buf129, buf130, buf140, 512, 64, stream=stream0)
            buf141 = empty_strided_cuda((12, 64), (64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, v_15, transpose_25, v_t_x_4, x_flat_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf137, (12, 512), (1, 12), 6144), reinterpret_tensor(buf140, (512, 64), (64, 1), 0), out=buf141)
            buf142 = reinterpret_tensor(buf140, (512, 64), (64, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [split_4, u_9, mixed_4, v_t_x_4], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf137, (512, 12), (12, 1), 0), buf141, out=buf142)
            buf144 = empty_strided_cuda((1, 512, 8, 8), (32768, 1, 4096, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, mixed_4, out_14, view_39, shift_9, out_15, x_16, x_17], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_38.run(buf128, arg59_1, buf129, buf130, buf142, buf137, buf144, 512, 64, stream=stream0)
            del arg59_1
            del buf128
            del buf142
            buf145 = reinterpret_tensor(buf138, (512, 512, 1, 1), (512, 1, 1, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg60_1, buf145, 262144, stream=stream0)
            del arg60_1
            # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf146 = extern_kernels.convolution(buf144, buf145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf146, (1, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf144
            buf147 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, add_17, h_7], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_39.run(buf147, arg61_1, buf115, 32768, stream=stream0)
            del arg61_1
            buf148 = reinterpret_tensor(buf145, (512, 512), (512, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [q_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg62_1, buf148, 262144, stream=stream0)
            del arg62_1
            buf149 = reinterpret_tensor(buf115, (64, 512), (512, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf147, (64, 512), (512, 1), 0), reinterpret_tensor(buf148, (512, 512), (1, 512), 0), out=buf149)
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, q_10, view_41, q_11, view_42, k_7, view_43, v_17], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf150 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf149, (1, 8, 64, 64), (32768, 64, 512, 1), 0), reinterpret_tensor(buf139, (1, 8, 1, 64), (1024, 64, 1024, 1), 0), reinterpret_tensor(buf139, (1, 8, 1, 64), (1024, 64, 1024, 1), 512), scale=0.125)
            del buf139
            buf151 = buf150[0]
            assert_size_stride(buf151, (1, 8, 64, 64), (32768, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf151, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf150
            buf156 = buf148; del buf148  # reuse
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg64_1, buf156, 262144, stream=stream0)
            del arg64_1
            buf157 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [transpose_31, out_16, input_22], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf151, (64, 512), (512, 1), 0), reinterpret_tensor(buf156, (512, 512), (1, 512), 0), out=buf157)
            buf158 = reinterpret_tensor(buf157, (1, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_32.run(buf158, buf147, arg65_1, 32768, stream=stream0)
            del arg65_1
            buf159 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_33.run(arg66_1, buf159, 2359296, stream=stream0)
            del arg66_1
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf160, (1, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf159
            buf161 = buf130; del buf130  # reuse
            buf162 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_34.run(buf160, arg67_1, buf161, buf162, 32, 1024, stream=stream0)
            buf164 = reinterpret_tensor(buf158, (1, 512, 64), (32768, 64, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5, x_flat_5, v_t_x_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_37.run(buf160, arg67_1, buf161, buf162, buf164, 512, 64, stream=stream0)
            buf165 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, v_19, transpose_33, v_t_x_5, x_flat_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf137, (12, 512), (1, 12), 6144), reinterpret_tensor(buf164, (512, 64), (64, 1), 0), out=buf165)
            buf166 = reinterpret_tensor(buf164, (512, 64), (64, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [split_5, u_11, mixed_5, v_t_x_5], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf137, (512, 12), (12, 1), 0), buf165, out=buf166)
            del buf165
            buf168 = reinterpret_tensor(buf151, (1, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, mixed_5, out_18, view_49, shift_11, out_19, x_20, x_21], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_38.run(buf160, arg67_1, buf161, buf162, buf166, buf137, buf168, 512, 64, stream=stream0)
            del arg67_1
            del buf137
            del buf160
            del buf166
            buf169 = reinterpret_tensor(buf156, (512, 512, 1, 1), (512, 1, 1, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg68_1, buf169, 262144, stream=stream0)
            del arg68_1
            # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf170 = extern_kernels.convolution(buf168, buf169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf170, (1, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf174 = reinterpret_tensor(buf168, (1, 64, 512), (32768, 512, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_40.run(buf170, arg69_1, buf147, arg70_1, arg71_1, buf174, 64, 512, stream=stream0)
            del arg70_1
            del arg71_1
            buf175 = empty_strided_cuda((1536, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_41.run(arg72_1, buf175, 786432, stream=stream0)
            del arg72_1
            buf176 = empty_strided_cuda((1536, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(arg73_1, buf176, 1536, stream=stream0)
            del arg73_1
            buf177 = empty_strided_cuda((64, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.addmm(buf176, reinterpret_tensor(buf174, (64, 512), (512, 1), 0), reinterpret_tensor(buf175, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf177)
            del buf175
            del buf176
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, k_8, v_20], Original ATen: [aten.view, aten.permute, aten.select, aten._scaled_dot_product_flash_attention]
            buf178 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf177, (1, 8, 64, 64), (0, 64, 1536, 1), 0), reinterpret_tensor(buf177, (1, 8, 64, 64), (0, 64, 1536, 1), 512), reinterpret_tensor(buf177, (1, 8, 64, 64), (0, 64, 1536, 1), 1024), scale=0.125)
            del buf177
            buf179 = buf178[0]
            assert_size_stride(buf179, (1, 8, 64, 64), (32768, 64, 512, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf179, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf178
            buf184 = reinterpret_tensor(buf169, (512, 512), (512, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(arg74_1, buf184, 262144, stream=stream0)
            del arg74_1
            buf185 = reinterpret_tensor(buf174, (64, 512), (512, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [transpose_36, out_21, out_22], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf179, (64, 512), (512, 1), 0), reinterpret_tensor(buf184, (512, 512), (1, 512), 0), out=buf185)
            del buf184
            buf192 = reinterpret_tensor(buf179, (1, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.addmm, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_43.run(buf170, arg69_1, buf147, buf185, arg75_1, arg76_1, arg77_1, buf192, 64, 512, stream=stream0)
            del arg69_1
            del arg75_1
            del arg76_1
            del arg77_1
            del buf147
            del buf170
            del buf185
            buf190 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg85_1, buf190, 131072, stream=stream0)
            del arg85_1
            buf191 = empty_strided_cuda((1, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf190, (256, 512), (1, 256), 0), out=buf191)
            buf193 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_44.run(arg78_1, buf193, 2097152, stream=stream0)
            del arg78_1
            # Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution]
            buf194 = extern_kernels.convolution(buf192, buf193, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf194, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf193
            buf195 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg80_1, buf195, 65536, stream=stream0)
            del arg80_1
            buf196 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf195, (256, 256), (1, 256), 0), out=buf196)
            buf197 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_13.run(buf197, arg81_1, 256, stream=stream0)
            del arg81_1
            buf198 = buf195; del buf195  # reuse
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg82_1, buf198, 65536, stream=stream0)
            del arg82_1
            buf199 = empty_strided_cuda((1, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.mm(buf197, reinterpret_tensor(buf198, (256, 256), (1, 256), 0), out=buf199)
            del buf197
            buf200 = reinterpret_tensor(buf198, (1, 256, 16, 16), (65536, 256, 16, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_45.run(buf194, arg79_1, buf112, buf199, arg83_1, buf200, 256, 256, stream=stream0)
            del arg79_1
            del arg83_1
            buf201 = reinterpret_tensor(buf194, (256, 256), (256, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [q_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg84_1, buf201, 65536, stream=stream0)
            del arg84_1
            buf202 = reinterpret_tensor(buf112, (256, 256), (256, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf200, (256, 256), (1, 256), 0), reinterpret_tensor(buf201, (256, 256), (1, 256), 0), out=buf202)
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, q_14, view_53, q_15, view_54, k_10, view_55, v_22], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf203 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf202, (1, 4, 256, 64), (65536, 64, 256, 1), 0), reinterpret_tensor(buf191, (1, 4, 1, 64), (512, 64, 512, 1), 0), reinterpret_tensor(buf191, (1, 4, 1, 64), (512, 64, 512, 1), 256), scale=0.125)
            buf204 = buf203[0]
            assert_size_stride(buf204, (1, 4, 256, 64), (65536, 64, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf204, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf203
            buf209 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg86_1, buf209, 65536, stream=stream0)
            del arg86_1
            buf210 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [transpose_43, out_26, input_28], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf204, (256, 256), (256, 1), 0), reinterpret_tensor(buf209, (256, 256), (1, 256), 0), out=buf210)
            del buf204
            buf211 = reinterpret_tensor(buf210, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_46.run(buf211, buf200, arg87_1, 256, 256, stream=stream0)
            del arg87_1
            buf212 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg88_1, buf212, 589824, stream=stream0)
            del arg88_1
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf213, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf214 = buf162; del buf162  # reuse
            buf215 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24.run(buf213, arg89_1, buf214, buf215, 32, 2048, stream=stream0)
            buf217 = buf190; del buf190  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg93_1, buf217, 131072, stream=stream0)
            del arg93_1
            buf218 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf217, (256, 512), (1, 256), 0), out=buf218)
            buf219 = reinterpret_tensor(buf211, (1, 256, 256), (65536, 256, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, x_flat_7, v_t_x_6], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27.run(buf213, arg89_1, buf214, buf215, buf219, 256, 256, stream=stream0)
            buf220 = empty_strided_cuda((12, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, v_24, transpose_45, v_t_x_6, x_flat_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (12, 256), (1, 12), 3072), reinterpret_tensor(buf219, (256, 256), (256, 1), 0), out=buf220)
            buf221 = reinterpret_tensor(buf219, (256, 256), (256, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [split_6, u_13, mixed_6, v_t_x_6], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (256, 12), (12, 1), 0), buf220, out=buf221)
            buf223 = reinterpret_tensor(buf209, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, mixed_6, out_28, view_61, shift_13, out_29, x_24, x_25], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf213, arg89_1, buf214, buf215, buf221, buf76, buf223, 256, 256, stream=stream0)
            del arg89_1
            del buf213
            buf224 = reinterpret_tensor(buf221, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg90_1, buf224, 65536, stream=stream0)
            del arg90_1
            # Topologically Sorted Source Nodes: [x_24, x_25], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf225 = extern_kernels.convolution(buf223, buf224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf225, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf223
            buf226 = buf225; del buf225  # reuse
            # Topologically Sorted Source Nodes: [x_24, x_25, add_27, h_11], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_47.run(buf226, arg91_1, buf200, 256, 256, stream=stream0)
            del arg91_1
            buf227 = reinterpret_tensor(buf200, (256, 256), (256, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [q_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg92_1, buf227, 65536, stream=stream0)
            del arg92_1
            buf228 = reinterpret_tensor(buf224, (256, 256), (256, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf226, (256, 256), (256, 1), 0), reinterpret_tensor(buf227, (256, 256), (1, 256), 0), out=buf228)
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, q_17, view_63, q_18, view_64, k_12, view_65, v_26], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf229 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf228, (1, 4, 256, 64), (65536, 64, 256, 1), 0), reinterpret_tensor(buf218, (1, 4, 1, 64), (512, 64, 512, 1), 0), reinterpret_tensor(buf218, (1, 4, 1, 64), (512, 64, 512, 1), 256), scale=0.125)
            buf230 = buf229[0]
            assert_size_stride(buf230, (1, 4, 256, 64), (65536, 64, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf230, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf229
            buf235 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg94_1, buf235, 65536, stream=stream0)
            del arg94_1
            buf236 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [transpose_51, out_30, input_30], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf230, (256, 256), (256, 1), 0), reinterpret_tensor(buf235, (256, 256), (1, 256), 0), out=buf236)
            del buf230
            buf237 = reinterpret_tensor(buf236, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf236  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_22.run(buf237, buf226, arg95_1, 65536, stream=stream0)
            del arg95_1
            buf238 = buf212; del buf212  # reuse
            # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg96_1, buf238, 589824, stream=stream0)
            del arg96_1
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf239 = extern_kernels.convolution(buf237, buf238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf239, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf240 = buf215; del buf215  # reuse
            buf241 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24.run(buf239, arg97_1, buf240, buf241, 32, 2048, stream=stream0)
            buf243 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg101_1, buf243, 131072, stream=stream0)
            del arg101_1
            buf244 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf243, (256, 512), (1, 256), 0), out=buf244)
            buf245 = reinterpret_tensor(buf237, (1, 256, 256), (65536, 256, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8, x_flat_8, v_t_x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27.run(buf239, arg97_1, buf240, buf241, buf245, 256, 256, stream=stream0)
            buf246 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, v_28, transpose_53, v_t_x_7, x_flat_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (12, 256), (1, 12), 3072), reinterpret_tensor(buf245, (256, 256), (256, 1), 0), out=buf246)
            buf247 = reinterpret_tensor(buf245, (256, 256), (256, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [split_7, u_15, mixed_7, v_t_x_7], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (256, 12), (12, 1), 0), buf246, out=buf247)
            buf249 = reinterpret_tensor(buf235, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, mixed_7, out_32, view_71, shift_15, out_33, x_28, x_29], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf239, arg97_1, buf240, buf241, buf247, buf76, buf249, 256, 256, stream=stream0)
            del arg97_1
            del buf239
            buf250 = reinterpret_tensor(buf247, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg98_1, buf250, 65536, stream=stream0)
            del arg98_1
            # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf251 = extern_kernels.convolution(buf249, buf250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf251, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf249
            buf252 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, add_31, h_12], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_29.run(buf252, arg99_1, buf226, 65536, stream=stream0)
            del arg99_1
            buf253 = reinterpret_tensor(buf226, (256, 256), (256, 1), 0); del buf226  # reuse
            # Topologically Sorted Source Nodes: [q_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg100_1, buf253, 65536, stream=stream0)
            del arg100_1
            buf254 = reinterpret_tensor(buf250, (256, 256), (256, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf252, (256, 256), (256, 1), 0), reinterpret_tensor(buf253, (256, 256), (1, 256), 0), out=buf254)
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, q_20, view_73, q_21, view_74, k_14, view_75, v_30], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf255 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf254, (1, 4, 256, 64), (65536, 64, 256, 1), 0), reinterpret_tensor(buf244, (1, 4, 1, 64), (512, 64, 512, 1), 0), reinterpret_tensor(buf244, (1, 4, 1, 64), (512, 64, 512, 1), 256), scale=0.125)
            buf256 = buf255[0]
            assert_size_stride(buf256, (1, 4, 256, 64), (65536, 64, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf256, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf255
            buf261 = buf254; del buf254  # reuse
            # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg102_1, buf261, 65536, stream=stream0)
            del arg102_1
            buf262 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [transpose_59, out_34, input_32], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf256, (256, 256), (256, 1), 0), reinterpret_tensor(buf261, (256, 256), (1, 256), 0), out=buf262)
            del buf256
            buf263 = reinterpret_tensor(buf262, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf262  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_22.run(buf263, buf252, arg103_1, 65536, stream=stream0)
            del arg103_1
            buf264 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg104_1, buf264, 589824, stream=stream0)
            del arg104_1
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf265 = extern_kernels.convolution(buf263, buf264, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf265, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf266 = buf241; del buf241  # reuse
            buf267 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24.run(buf265, arg105_1, buf266, buf267, 32, 2048, stream=stream0)
            buf269 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg109_1, buf269, 131072, stream=stream0)
            del arg109_1
            buf270 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf269, (256, 512), (1, 256), 0), out=buf270)
            buf271 = reinterpret_tensor(buf263, (1, 256, 256), (65536, 256, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9, x_flat_9, v_t_x_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27.run(buf265, arg105_1, buf266, buf267, buf271, 256, 256, stream=stream0)
            buf272 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, v_32, transpose_61, v_t_x_8, x_flat_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (12, 256), (1, 12), 3072), reinterpret_tensor(buf271, (256, 256), (256, 1), 0), out=buf272)
            buf273 = reinterpret_tensor(buf271, (256, 256), (256, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [split_8, u_17, mixed_8, v_t_x_8], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (256, 12), (12, 1), 0), buf272, out=buf273)
            buf275 = reinterpret_tensor(buf261, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, mixed_8, out_36, view_81, shift_17, out_37, x_32, x_33], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf265, arg105_1, buf266, buf267, buf273, buf76, buf275, 256, 256, stream=stream0)
            del arg105_1
            del buf265
            buf276 = reinterpret_tensor(buf273, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf273  # reuse
            # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg106_1, buf276, 65536, stream=stream0)
            del arg106_1
            # Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf277 = extern_kernels.convolution(buf275, buf276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf277, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf275
            buf278 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [x_32, x_33, add_35, h_13], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_29.run(buf278, arg107_1, buf252, 65536, stream=stream0)
            del arg107_1
            buf279 = reinterpret_tensor(buf252, (256, 256), (256, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [q_23], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg108_1, buf279, 65536, stream=stream0)
            del arg108_1
            buf280 = reinterpret_tensor(buf276, (256, 256), (256, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf278, (256, 256), (256, 1), 0), reinterpret_tensor(buf279, (256, 256), (1, 256), 0), out=buf280)
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, q_23, view_83, q_24, view_84, k_16, view_85, v_34], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf281 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf280, (1, 4, 256, 64), (65536, 64, 256, 1), 0), reinterpret_tensor(buf270, (1, 4, 1, 64), (512, 64, 512, 1), 0), reinterpret_tensor(buf270, (1, 4, 1, 64), (512, 64, 512, 1), 256), scale=0.125)
            buf282 = buf281[0]
            assert_size_stride(buf282, (1, 4, 256, 64), (65536, 64, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf282, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf281
            buf287 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg110_1, buf287, 65536, stream=stream0)
            del arg110_1
            buf288 = buf279; del buf279  # reuse
            # Topologically Sorted Source Nodes: [transpose_67, out_38, input_34], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf282, (256, 256), (256, 1), 0), reinterpret_tensor(buf287, (256, 256), (1, 256), 0), out=buf288)
            del buf282
            buf289 = reinterpret_tensor(buf288, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_22.run(buf289, buf278, arg111_1, 65536, stream=stream0)
            del arg111_1
            buf290 = buf264; del buf264  # reuse
            # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg112_1, buf290, 589824, stream=stream0)
            del arg112_1
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf291 = extern_kernels.convolution(buf289, buf290, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf291, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf292 = buf267; del buf267  # reuse
            buf293 = buf266; del buf266  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24.run(buf291, arg113_1, buf292, buf293, 32, 2048, stream=stream0)
            buf295 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(arg117_1, buf295, 131072, stream=stream0)
            del arg117_1
            buf296 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf295, (256, 512), (1, 256), 0), out=buf296)
            buf297 = reinterpret_tensor(buf289, (1, 256, 256), (65536, 256, 1), 0); del buf289  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10, x_flat_10, v_t_x_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27.run(buf291, arg113_1, buf292, buf293, buf297, 256, 256, stream=stream0)
            buf298 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, v_36, transpose_69, v_t_x_9, x_flat_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (12, 256), (1, 12), 3072), reinterpret_tensor(buf297, (256, 256), (256, 1), 0), out=buf298)
            buf299 = reinterpret_tensor(buf297, (256, 256), (256, 1), 0); del buf297  # reuse
            # Topologically Sorted Source Nodes: [split_9, u_19, mixed_9, v_t_x_9], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (256, 12), (12, 1), 0), buf298, out=buf299)
            buf301 = reinterpret_tensor(buf287, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf287  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, mixed_9, out_40, view_91, shift_19, out_41, x_36, x_37], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf291, arg113_1, buf292, buf293, buf299, buf76, buf301, 256, 256, stream=stream0)
            del arg113_1
            del buf291
            buf302 = reinterpret_tensor(buf299, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf299  # reuse
            # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg114_1, buf302, 65536, stream=stream0)
            del arg114_1
            # Topologically Sorted Source Nodes: [x_36, x_37], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf303 = extern_kernels.convolution(buf301, buf302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf303, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf301
            buf304 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, add_39, h_14], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_29.run(buf304, arg115_1, buf278, 65536, stream=stream0)
            del arg115_1
            buf305 = reinterpret_tensor(buf278, (256, 256), (256, 1), 0); del buf278  # reuse
            # Topologically Sorted Source Nodes: [q_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg116_1, buf305, 65536, stream=stream0)
            del arg116_1
            buf306 = reinterpret_tensor(buf302, (256, 256), (256, 1), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf304, (256, 256), (256, 1), 0), reinterpret_tensor(buf305, (256, 256), (1, 256), 0), out=buf306)
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, q_26, view_93, q_27, view_94, k_18, view_95, v_38], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf307 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf306, (1, 4, 256, 64), (65536, 64, 256, 1), 0), reinterpret_tensor(buf296, (1, 4, 1, 64), (512, 64, 512, 1), 0), reinterpret_tensor(buf296, (1, 4, 1, 64), (512, 64, 512, 1), 256), scale=0.125)
            del buf296
            buf308 = buf307[0]
            assert_size_stride(buf308, (1, 4, 256, 64), (65536, 64, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf308, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf307
            buf313 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg118_1, buf313, 65536, stream=stream0)
            del arg118_1
            buf314 = buf305; del buf305  # reuse
            # Topologically Sorted Source Nodes: [transpose_75, out_42, input_36], Original ATen: [aten.transpose, aten._unsafe_view, aten.view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf308, (256, 256), (256, 1), 0), reinterpret_tensor(buf313, (256, 256), (1, 256), 0), out=buf314)
            del buf308
            buf315 = reinterpret_tensor(buf314, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_transpose_view_22.run(buf315, buf304, arg119_1, 65536, stream=stream0)
            del arg119_1
            buf316 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg120_1, buf316, 589824, stream=stream0)
            del arg120_1
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf317 = extern_kernels.convolution(buf315, buf316, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf317, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf316
            buf318 = buf293; del buf293  # reuse
            buf319 = buf292; del buf292  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_24.run(buf317, arg121_1, buf318, buf319, 32, 2048, stream=stream0)
            buf321 = reinterpret_tensor(buf315, (1, 256, 256), (65536, 256, 1), 0); del buf315  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11, x_flat_11, v_t_x_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_27.run(buf317, arg121_1, buf318, buf319, buf321, 256, 256, stream=stream0)
            buf322 = buf298; del buf298  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, v_40, transpose_77, v_t_x_10, x_flat_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (12, 256), (1, 12), 3072), reinterpret_tensor(buf321, (256, 256), (256, 1), 0), out=buf322)
            buf323 = reinterpret_tensor(buf321, (256, 256), (256, 1), 0); del buf321  # reuse
            # Topologically Sorted Source Nodes: [split_10, u_21, mixed_10, v_t_x_10], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf76, (256, 12), (12, 1), 0), buf322, out=buf323)
            del buf322
            buf325 = reinterpret_tensor(buf313, (1, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, mixed_10, out_44, view_101, shift_21, out_45, x_40, x_41], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf317, arg121_1, buf318, buf319, buf323, buf76, buf325, 256, 256, stream=stream0)
            del arg121_1
            del buf317
            del buf76
            buf326 = reinterpret_tensor(buf323, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf323  # reuse
            # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg122_1, buf326, 65536, stream=stream0)
            del arg122_1
            # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf327 = extern_kernels.convolution(buf325, buf326, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf327, (1, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf325
            del buf326
            buf328 = buf327; del buf327  # reuse
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_29.run(buf328, arg123_1, buf304, 65536, stream=stream0)
            del arg123_1
            del buf304
            buf329 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_48.run(arg124_1, buf329, 524288, stream=stream0)
            del arg124_1
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            buf330 = extern_kernels.convolution(buf328, buf329, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf330, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf329
            buf331 = reinterpret_tensor(buf328, (256, 256), (256, 1), 0); del buf328  # reuse
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg126_1, buf331, 65536, stream=stream0)
            del arg126_1
            buf332 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf22, reinterpret_tensor(buf331, (256, 256), (1, 256), 0), out=buf332)
            del buf22
            del buf331
            buf333 = buf332; del buf332  # reuse
            # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_13.run(buf333, arg127_1, 256, stream=stream0)
            del arg127_1
            buf334 = reinterpret_tensor(buf192, (128, 256), (256, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_49.run(arg128_1, buf334, 32768, stream=stream0)
            del arg128_1
            buf335 = empty_strided_cuda((1, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.mm(buf333, reinterpret_tensor(buf334, (256, 128), (1, 256), 0), out=buf335)
            del buf333
            del buf334
            buf336 = buf330; del buf330  # reuse
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_50.run(buf336, arg125_1, buf51, buf335, arg129_1, 131072, stream=stream0)
            del arg125_1
            del arg129_1
            del buf335
            buf337 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(arg130_1, buf337, 147456, stream=stream0)
            del arg130_1
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.convolution]
            buf338 = extern_kernels.convolution(buf336, buf337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf338, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf339 = buf319; del buf319  # reuse
            buf340 = buf318; del buf318  # reuse
            # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_4.run(buf338, arg131_1, buf339, buf340, 32, 4096, stream=stream0)
            buf342 = reinterpret_tensor(buf51, (1, 128, 1024), (131072, 1024, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [x_42, x_norm_12, x_flat_12, v_t_x_11], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16.run(buf338, arg131_1, buf339, buf340, buf342, 128, 1024, stream=stream0)
            buf343 = empty_strided_cuda((12, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, v_42, transpose_78, v_t_x_11, x_flat_12], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (12, 128), (1, 12), 1536), reinterpret_tensor(buf342, (128, 1024), (1024, 1), 0), out=buf343)
            buf344 = reinterpret_tensor(buf342, (128, 1024), (1024, 1), 0); del buf342  # reuse
            # Topologically Sorted Source Nodes: [split_11, u_23, mixed_11, v_t_x_11], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 12), (12, 1), 0), buf343, out=buf344)
            buf346 = reinterpret_tensor(buf295, (1, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf295  # reuse
            # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, mixed_11, out_46, view_106, shift_23, out_47, x_43, x_44], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17.run(buf338, arg131_1, buf339, buf340, buf344, buf28, buf346, 128, 1024, stream=stream0)
            del arg131_1
            del buf338
            del buf344
            buf347 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg132_1, buf347, 16384, stream=stream0)
            del arg132_1
            # Topologically Sorted Source Nodes: [x_43, x_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf348 = extern_kernels.convolution(buf346, buf347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf348, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf349 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [x_43, x_44, add_47, h_18], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_19.run(buf349, arg133_1, buf336, 131072, stream=stream0)
            del arg133_1
            buf350 = buf337; del buf337  # reuse
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(arg134_1, buf350, 147456, stream=stream0)
            del arg134_1
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy, aten.convolution]
            buf351 = extern_kernels.convolution(buf349, buf350, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf351, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf352 = buf340; del buf340  # reuse
            buf353 = buf339; del buf339  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_4.run(buf351, arg135_1, buf352, buf353, 32, 4096, stream=stream0)
            buf355 = reinterpret_tensor(buf336, (1, 128, 1024), (131072, 1024, 1), 0); del buf336  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13, x_flat_13, v_t_x_12], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16.run(buf351, arg135_1, buf352, buf353, buf355, 128, 1024, stream=stream0)
            buf356 = buf343; del buf343  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, v_44, transpose_79, v_t_x_12, x_flat_13], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (12, 128), (1, 12), 1536), reinterpret_tensor(buf355, (128, 1024), (1024, 1), 0), out=buf356)
            buf357 = reinterpret_tensor(buf355, (128, 1024), (1024, 1), 0); del buf355  # reuse
            # Topologically Sorted Source Nodes: [split_12, u_25, mixed_12, v_t_x_12], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 12), (12, 1), 0), buf356, out=buf357)
            buf359 = buf346; del buf346  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, mixed_12, out_48, view_111, shift_25, out_49, x_46, x_47], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17.run(buf351, arg135_1, buf352, buf353, buf357, buf28, buf359, 128, 1024, stream=stream0)
            del arg135_1
            del buf351
            del buf357
            buf360 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg136_1, buf360, 16384, stream=stream0)
            del arg136_1
            # Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf361 = extern_kernels.convolution(buf359, buf360, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf361, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf362 = buf361; del buf361  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47, add_50, h_19], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_19.run(buf362, arg137_1, buf349, 131072, stream=stream0)
            del arg137_1
            buf363 = buf350; del buf350  # reuse
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(arg138_1, buf363, 147456, stream=stream0)
            del arg138_1
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.convolution]
            buf364 = extern_kernels.convolution(buf362, buf363, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf364, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf365 = buf353; del buf353  # reuse
            buf366 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_4.run(buf364, arg139_1, buf365, buf366, 32, 4096, stream=stream0)
            buf368 = reinterpret_tensor(buf349, (1, 128, 1024), (131072, 1024, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14, x_flat_14, v_t_x_13], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16.run(buf364, arg139_1, buf365, buf366, buf368, 128, 1024, stream=stream0)
            buf369 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, v_46, transpose_80, v_t_x_13, x_flat_14], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (12, 128), (1, 12), 1536), reinterpret_tensor(buf368, (128, 1024), (1024, 1), 0), out=buf369)
            buf370 = reinterpret_tensor(buf368, (128, 1024), (1024, 1), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [split_13, u_27, mixed_13, v_t_x_13], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 12), (12, 1), 0), buf369, out=buf370)
            buf372 = buf359; del buf359  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, mixed_13, out_50, view_116, shift_27, out_51, x_49, x_50], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17.run(buf364, arg139_1, buf365, buf366, buf370, buf28, buf372, 128, 1024, stream=stream0)
            del arg139_1
            del buf364
            del buf370
            buf373 = buf360; del buf360  # reuse
            # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg140_1, buf373, 16384, stream=stream0)
            del arg140_1
            # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf374 = extern_kernels.convolution(buf372, buf373, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf374, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf375 = buf374; del buf374  # reuse
            # Topologically Sorted Source Nodes: [x_49, x_50, add_53, h_20], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_19.run(buf375, arg141_1, buf362, 131072, stream=stream0)
            del arg141_1
            buf376 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(arg142_1, buf376, 147456, stream=stream0)
            del arg142_1
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy, aten.convolution]
            buf377 = extern_kernels.convolution(buf375, buf376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf377, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf378 = buf366; del buf366  # reuse
            buf379 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_4.run(buf377, arg143_1, buf378, buf379, 32, 4096, stream=stream0)
            buf381 = reinterpret_tensor(buf362, (1, 128, 1024), (131072, 1024, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15, x_flat_15, v_t_x_14], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16.run(buf377, arg143_1, buf378, buf379, buf381, 128, 1024, stream=stream0)
            buf382 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, v_48, transpose_81, v_t_x_14, x_flat_15], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (12, 128), (1, 12), 1536), reinterpret_tensor(buf381, (128, 1024), (1024, 1), 0), out=buf382)
            buf383 = reinterpret_tensor(buf381, (128, 1024), (1024, 1), 0); del buf381  # reuse
            # Topologically Sorted Source Nodes: [split_14, u_29, mixed_14, v_t_x_14], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 12), (12, 1), 0), buf382, out=buf383)
            buf385 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, mixed_14, out_52, view_121, shift_29, out_53, x_52, x_53], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17.run(buf377, arg143_1, buf378, buf379, buf383, buf28, buf385, 128, 1024, stream=stream0)
            del arg143_1
            del buf377
            del buf383
            buf386 = buf373; del buf373  # reuse
            # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg144_1, buf386, 16384, stream=stream0)
            del arg144_1
            # Topologically Sorted Source Nodes: [x_52, x_53], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf387 = extern_kernels.convolution(buf385, buf386, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf387, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf388 = buf387; del buf387  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53, add_56, h_21], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_19.run(buf388, arg145_1, buf375, 131072, stream=stream0)
            del arg145_1
            buf389 = buf376; del buf376  # reuse
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(arg146_1, buf389, 147456, stream=stream0)
            del arg146_1
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.convolution]
            buf390 = extern_kernels.convolution(buf388, buf389, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf390, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf389
            buf391 = buf379; del buf379  # reuse
            buf392 = buf378; del buf378  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_4.run(buf390, arg147_1, buf391, buf392, 32, 4096, stream=stream0)
            buf394 = reinterpret_tensor(buf375, (1, 128, 1024), (131072, 1024, 1), 0); del buf375  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16, x_flat_16, v_t_x_15], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_16.run(buf390, arg147_1, buf391, buf392, buf394, 128, 1024, stream=stream0)
            buf395 = buf382; del buf382  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, v_50, transpose_82, v_t_x_15, x_flat_16], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (12, 128), (1, 12), 1536), reinterpret_tensor(buf394, (128, 1024), (1024, 1), 0), out=buf395)
            buf396 = reinterpret_tensor(buf394, (128, 1024), (1024, 1), 0); del buf394  # reuse
            # Topologically Sorted Source Nodes: [split_15, u_31, mixed_15, v_t_x_15], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 12), (12, 1), 0), buf395, out=buf396)
            del buf395
            buf398 = buf385; del buf385  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, mixed_15, out_54, view_126, shift_31, out_55, x_55, x_56], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.bmm, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_bmm_clone_convolution_native_group_norm_silu_split_with_sizes_view_17.run(buf390, arg147_1, buf391, buf392, buf396, buf28, buf398, 128, 1024, stream=stream0)
            del arg147_1
            del buf28
            del buf390
            del buf396
            buf399 = buf386; del buf386  # reuse
            # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg148_1, buf399, 16384, stream=stream0)
            del arg148_1
            # Topologically Sorted Source Nodes: [x_55, x_56], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf400 = extern_kernels.convolution(buf398, buf399, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf400, (1, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf399
            buf401 = buf392; del buf392  # reuse
            buf402 = buf391; del buf391  # reuse
            # Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_51.run(buf400, arg149_1, buf388, buf401, buf402, 32, 4096, stream=stream0)
            buf405 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42, input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_52.run(buf400, arg149_1, buf388, buf401, buf402, arg150_1, arg151_1, buf405, 131072, stream=stream0)
            del arg149_1
            del arg150_1
            del arg151_1
            del buf388
            del buf400
            del buf401
            del buf402
            buf406 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg152_1, buf406, 4608, stream=stream0)
            del arg152_1
            # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf407 = extern_kernels.convolution(buf405, buf406, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf407, (1, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
            del buf405
            del buf406
            buf408 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_53.run(buf408, arg153_1, 4096, stream=stream0)
            del arg153_1
        return (buf408, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3200, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((6400, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((6400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((12800, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((12800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 4, 3, 3), (36, 1, 12, 4), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((4, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
