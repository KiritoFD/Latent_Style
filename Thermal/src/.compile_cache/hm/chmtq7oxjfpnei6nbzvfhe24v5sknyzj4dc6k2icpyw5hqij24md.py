# AOT ID: ['2_inference']
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
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: ./.compile_cache/6g/c6g3l5mn65qjk3illjezuxqnla5ut4z4wb3phywwkpc7vaxgipha.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h => convolution
# Graph fragment:
#   %arg25_1 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %convolution : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1310720, 'x': 655360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 4*x2 + 4096*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/jk/cjkvhnramivnohq66j7t5gedqqptugkjzfowww3psdvqnfutd3i3.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h => convolution
# Graph fragment:
#   %arg23_1 : Tensor "f32[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = PlaceHolder[target=arg23_1]
#   %convolution : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 36864, 'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 4*x2 + 36*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/f3/cf3laercnvmcml4hh6d5va27ftxqla4q5n2noxuoc6og7igevzch.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h => convolution
# Graph fragment:
#   %buf2 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf2]
#   %arg24_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %convolution : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62915072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/uq/cuqwzra2d27cybu2oraxjo7qskqkexov36lrdhwscti7exzztw75.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution_1
# Graph fragment:
#   %arg26_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg26_1]
#   %convolution_1 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf4
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/xn/cxnxkrfwk5eczfm32pcv7yqhshkonijai24hzbjil3vzb3ihkmum.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution_1
#   x_norm => var_mean, view
# Graph fragment:
#   %buf5 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %convolution_1 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf7
triton_red_fused_convolution_native_group_norm_4 = async_compile.triton('triton_red_fused_convolution_native_group_norm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480, 'r0_': 20972032}}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1280
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/i3/ci3dlqkwj2w7jtczdsdeo5szpflabbqrdugzvt7puosdmpey7mba.py
# Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat]
# Source node to ATen node mapping:
#   arange => iota
#   args => mul_1
#   cos => cos
#   emb => cat
#   freqs => exp
#   getitem => unsqueeze
#   getitem_1 => unsqueeze_1
#   mul => mul
#   sin => sin
#   truediv => div
# Graph fragment:
#   %arg0_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %unsqueeze : Tensor "f32[40, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, -9.210340371976184), kwargs = {})
#   %div : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 128), kwargs = {})
#   %exp : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 0), kwargs = {})
#   %mul_1 : Tensor "f32[40, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %cos : Tensor "f32[40, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %sin : Tensor "f32[40, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %cat : Tensor "f32[40, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   return %cat
triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5 = async_compile.triton('triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 81920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = -9.210340371976184
    tmp9 = tmp7 * tmp8
    tmp10 = 0.0078125
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp13 = tmp5 * tmp12
    tmp14 = tl_math.cos(tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 256, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr0 + (x1), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = (-128) + x0
    tmp22 = tmp21.to(tl.float32)
    tmp23 = -9.210340371976184
    tmp24 = tmp22 * tmp23
    tmp25 = 0.0078125
    tmp26 = tmp24 * tmp25
    tmp27 = libdevice.exp(tmp26)
    tmp28 = tmp20 * tmp27
    tmp29 = tl_math.sin(tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp17, tmp29, tmp30)
    tmp32 = tl.where(tmp4, tmp16, tmp31)
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/3m/c3mz76cpv4hcsktkcbps2d5emrcxdkyhmni6gzaprndr2zaqk5d6.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_40
#   input_2 => mul_2, sigmoid
# Graph fragment:
#   %mm_default_39 : Tensor "f32[40, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_default_39]
#   %arg2_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add_tensor_40 : Tensor "f32[40, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_39, %arg2_1), kwargs = {})
#   %sigmoid : Tensor "f32[40, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_40,), kwargs = {})
#   %mul_2 : Tensor "f32[40, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_40, %sigmoid), kwargs = {})
#   return %mul_2
triton_poi_fused_addmm_silu_6 = async_compile.triton('triton_poi_fused_addmm_silu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 495616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/xd/cxdyfjksbnvpzsvpkchbghmyor262c2my2ana2repcgm66wvueb4.py
# Topologically Sorted Source Nodes: [s_emb], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   s_emb => embedding
# Graph fragment:
#   %arg6_1 : Tensor "i64[40][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg5_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "f32[40, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg6_1), kwargs = {})
#   return %embedding
triton_poi_fused_embedding_7 = async_compile.triton('triton_poi_fused_embedding_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x0 + 256*tmp4), xmask)
    tl.store(out_ptr0 + (x0 + 512*x1), tmp6, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/6m/c6msnuruvl5y7zpcs6wgegljcfu3vimd4lwmqtfttzr3mbcbz5qi.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_4 => add_tensor_39
#   input_5 => mul_3, sigmoid_1
# Graph fragment:
#   %mm_default_38 : Tensor "f32[40, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_38]
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %add_tensor_39 : Tensor "f32[40, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_38, %arg8_1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[40, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_39,), kwargs = {})
#   %mul_3 : Tensor "f32[40, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_39, %sigmoid_1), kwargs = {})
#   return %mul_3
triton_poi_fused_addmm_silu_8 = async_compile.triton('triton_poi_fused_addmm_silu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 247808}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/w3/cw3rzhq4ipfahzqqyxumkfqdlyqv3wrpo7mcpemhmtgcp2vicgyr.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_7 => add_tensor_38
#   input_8 => mul_4, sigmoid_2
# Graph fragment:
#   %mm_default_37 : Tensor "f32[40, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_37]
#   %arg12_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_tensor_38 : Tensor "f32[40, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_37, %arg12_1), kwargs = {})
#   %sigmoid_2 : Tensor "f32[40, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_38,), kwargs = {})
#   %mul_4 : Tensor "f32[40, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_38, %sigmoid_2), kwargs = {})
#   return %mul_4
triton_poi_fused_addmm_silu_9 = async_compile.triton('triton_poi_fused_addmm_silu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 123904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/u2/cu2icbv2c2tdsglwtttndwqva2t4qwjtcvix65k7ywlt4gy72rtd.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution_1
#   x_norm => add, mul_7, rsqrt, sub, var_mean, view
# Graph fragment:
#   %buf5 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %getitem_1 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=buf7]
#   %convolution_1 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   return %mul_7
triton_poi_fused_convolution_native_group_norm_10 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 20972032, 'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 128
    y4 = (yindex % 128)
    y5 = yindex // 4
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 128*x3 + 131072*y2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 4096.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3 + 1024*y6), tmp11, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/tw/ctwk35ym3zmliccztjuouyvm2l6rvbimlkrrqwr33in7t6wx5au2.py
# Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
# Source node to ATen node mapping:
#   out => add_1
#   out_1 => add_2
#   shift_1 => view_4
#   split => split_with_sizes
#   view_4 => view_6
#   x => convolution_1
#   x_1 => mul_8, sigmoid_5
#   x_flat => view_5
#   x_norm => add, mul_7, rsqrt, sub, var_mean, view, view_1
# Graph fragment:
#   %mul_7 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0" = PlaceHolder[target=mul_7]
#   %bmm_1 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %addmm_5 : Tensor "f32[40, 3200][3200, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %convolution_1 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [40, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [40, 128, -1]), kwargs = {})
#   %add_1 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [40, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "f32[40, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [40, 128, 1, 1]), kwargs = {})
#   %add_2 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   return %mul_8
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 41963520, 'x': 41943040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (3072 + y0 + 3200*y1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (y0 + 128*x2 + 131072*y1), tmp6, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/q5/cq5olgy7n7doyj5rupejnaou5znv3tzd2zrvn7iirlyu4zuy4qtl.py
# Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2, add_2, h_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_2 => add_3
#   h_1 => mul_9, sigmoid_6
#   out => add_1
#   out_1 => add_2
#   shift_1 => view_4
#   split => split_with_sizes
#   view_4 => view_6
#   x => convolution_1
#   x_1 => mul_8, sigmoid_5
#   x_2 => convolution_2
#   x_flat => view_5
#   x_norm => add, mul_7, rsqrt, sub, var_mean, view, view_1
# Graph fragment:
#   %buf25 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf25]
#   %arg29_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %convolution : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution]
#   %convolution_1 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [40, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [40, 128, -1]), kwargs = {})
#   %add_1 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [40, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "f32[40, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [40, 128, 1, 1]), kwargs = {})
#   %add_2 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   %convolution_2 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_8, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution), kwargs = {})
#   %sigmoid_6 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_9 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_6), kwargs = {})
#   return %mul_9
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83886592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/xv/cxv44fzgy52ofut65hpi7olsilzolkm367yf3bygg6xjmcxc5t6c.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h_3 => convolution_5
# Graph fragment:
#   %arg34_1 : Tensor "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg34_1]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf39
triton_poi_fused_convolution_13 = async_compile.triton('triton_poi_fused_convolution_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/3n/c3ncarz5isablxdvs7nh4phncjpx65c4hztp5vnnlpvmeqwm2nrr.py
# Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   q => permute_12
#   q_1 => clone
#   view_10 => view_14
# Graph fragment:
#   %buf40 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_14 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_5, [40, 256, 256]), kwargs = {})
#   %permute_12 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_12,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_convolution_transpose_view_14 = async_compile.triton('triton_poi_fused_clone_convolution_transpose_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_transpose_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31458304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_transpose_view_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/tk/ctknydrxwrokihg6vgrhaeji6wnwirhpavit7nod3eoympszcmh6.py
# Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_1
#   q_1 => view_16
#   q_2 => permute_15
#   view_11 => view_17
# Graph fragment:
#   %mm : Tensor "f32[10240, 256][256, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_16 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [40, 256, 256]), kwargs = {})
#   %view_17 : Tensor "f32[40, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [40, 256, 4, 64]), kwargs = {})
#   %permute_15 : Tensor "f32[40, 4, 256, 64][65536, 64, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "f32[40, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_clone_transpose_view_15 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31457280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 256)
    x2 = ((xindex // 16384) % 4)
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 256*x1 + 65536*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/lo/clopq3zqtreezndzntubovlcuccivr5v6mrhz7hdgz2acvtb4cxw.py
# Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   k_1 => permute_16
#   kv => unsqueeze_2
#   matmul => clone_2
#   transpose_6 => permute_18
#   view_12 => view_18
# Graph fragment:
#   %mm_1 : Tensor "f32[40, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "f32[40, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_18 : Tensor "f32[40, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_10, [40, 1, 4, 64]), kwargs = {})
#   %permute_16 : Tensor "f32[40, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %permute_18 : Tensor "f32[40, 4, 64, 1][512, 64, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_16, [0, 1, 3, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[40, 4, 64, 1][256, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused_clone_split_transpose_unsqueeze_view_16 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 122880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/gf/cgfk7ba6xfdvtsmkwwuc77xhsrwu72hmyeky7cq3hcmhqjs44ii6.py
# Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div_1, exp_1, sum_1
#   matmul => view_22
# Graph fragment:
#   %bmm_4 : Tensor "f32[160, 256, 1][256, 1, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %view_22 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_4, [40, 4, 256, 1]), kwargs = {})
#   %mul_tensor_40 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, 1), kwargs = {})
#   %amax_default_19 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_40, [-1], True), kwargs = {})
#   %sub_tensor_19 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_40, %amax_default_19), kwargs = {})
#   %mul_tensor_41 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_19, 0.125), kwargs = {})
#   %exp_1 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_41,), kwargs = {})
#   %sum_1 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[40, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_1), kwargs = {})
#   return %expand_2
triton_poi_fused__softmax_amax_mul_sub_view_17 = async_compile.triton('triton_poi_fused__softmax_amax_mul_sub_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_amax_mul_sub_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 491520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_amax_mul_sub_view_17(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 - tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.exp(tmp5)
    tmp7 = (tmp6 / tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/og/cogmg5scl3mjzj55vxojwlzrxbbo3shafwuhws4u2wiuvx3n7cbh.py
# Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   kv => unsqueeze_2
#   matmul_1 => clone_3
#   v_5 => permute_17
#   view_13 => view_19
# Graph fragment:
#   %mm_1 : Tensor "f32[40, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "f32[40, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_19 : Tensor "f32[40, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_11, [40, 1, 4, 64]), kwargs = {})
#   %permute_17 : Tensor "f32[40, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[40, 4, 1, 64][256, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_split_transpose_unsqueeze_view_18 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 122880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/2e/c2e4uybjoustwjcvybyjbtk6gmbqk5r54gg2prxfkdfmnzuwakco.py
# Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => view_25
#   out_4 => clone_4
#   transpose_7 => permute_19
# Graph fragment:
#   %bmm_5 : Tensor "f32[160, 256, 64][16384, 64, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %view_25 : Tensor "f32[40, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [40, 4, 256, 64]), kwargs = {})
#   %permute_19 : Tensor "f32[40, 256, 4, 64][65536, 64, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_25, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[40, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_transpose_view_19 = async_compile.triton('triton_poi_fused_clone_transpose_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31457280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 4)
    x2 = ((xindex // 256) % 256)
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 16384*x1 + 65536*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/dz/cdzpgxmfdbdnisturw5zkgeoi2n4pyp7hcrd2vuymeqtbm35fwsp.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_37, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
# Graph fragment:
#   %buf40 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %mm_default_36 : Tensor "f32[10240, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_36]
#   %arg39_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   return %add_8
triton_poi_fused_add_addmm_convolution_transpose_view_20 = async_compile.triton('triton_poi_fused_add_addmm_convolution_transpose_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_transpose_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41945088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_transpose_view_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/pj/cpj2zf2nbciszvqo7jdvzltmz2pbal2pq2zysrqqtm4armrdaszr.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_37, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
# Graph fragment:
#   %arg40_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg40_1]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf52
triton_poi_fused_add_addmm_convolution_transpose_view_21 = async_compile.triton('triton_poi_fused_add_addmm_convolution_transpose_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_transpose_view_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_transpose_view_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/oi/coi5t65astowuwv5yljsxhxtpj246vwfiy36a7t6nufeuhieure5.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_37, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
#   x_norm_2 => var_mean_2, view_30
# Graph fragment:
#   %buf53 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf53]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_13,%buf55
triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22 = async_compile.triton('triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480, 'r0_': 10486784}}
)
@triton.jit
def triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1280
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 256*r0_3 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/vo/cvoq7qn26e4lout62anjmvhbdtjlgmrlgblsic5avhrx2k3ckyf2.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_37, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
#   x_norm_2 => add_9, mul_14, rsqrt_2, sub_3, var_mean_2, view_30
# Graph fragment:
#   %buf53 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf53]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %getitem_13 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf55 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=buf55]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   return %mul_14
triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 10486784, 'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 256
    y4 = (yindex % 256)
    y5 = yindex // 8
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 256*x3 + 65536*y2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 2048.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3 + 256*y6), tmp11, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/ha/chatzwvvjjvtvrl4kj7zsbu2zeq5rxk5ruyfmiddpfdocccczzwu.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_37, view_28
#   out_5 => view_29
#   out_6 => add_10
#   out_7 => add_11
#   shift_5 => view_34
#   split_2 => split_with_sizes_2
#   transpose_8 => permute_21
#   view_19 => view_36
#   x_6 => add_8
#   x_7 => convolution_6
#   x_8 => mul_15, sigmoid_9
#   x_flat_2 => view_35
#   x_norm_2 => add_9, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
# Graph fragment:
#   %mul_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0" = PlaceHolder[target=mul_14]
#   %bmm_7 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %addmm_7 : Tensor "f32[40, 6400][6400, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [40, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [40, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [40, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [40, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   return %mul_15
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 21012480, 'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (6144 + y0 + 6400*y1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (y0 + 256*x2 + 65536*y1), tmp6, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/lk/clk72a2dveipcace4w6a3qwtgamltyvgmjrbq7hacmcubkeyjoqa.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   input_16 => add_tensor_37, view_28
#   out_5 => view_29
#   out_6 => add_10
#   out_7 => add_11
#   q_3 => permute_23
#   q_4 => clone_6
#   shift_5 => view_34
#   split_2 => split_with_sizes_2
#   transpose_8 => permute_21
#   view_19 => view_36
#   view_20 => view_37
#   x_6 => add_8
#   x_7 => convolution_6
#   x_8 => mul_15, sigmoid_9
#   x_9 => convolution_7
#   x_flat_2 => view_35
#   x_norm_2 => add_9, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
# Graph fragment:
#   %buf65 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf65]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf40 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [40, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [40, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [40, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [40, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %view_37 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_16, [40, 256, 256]), kwargs = {})
#   %permute_23 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_37, [0, 2, 1]), kwargs = {})
#   %clone_6 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_6
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41945088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/hc/chcwcfp3uumzi3vodmzq3wn5ygfsnu5k6f7n3staauhj7enytp64.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   input_16 => add_tensor_37, view_28
#   input_18 => add_tensor_35, view_51
#   out_5 => view_29
#   out_6 => add_10
#   out_7 => add_11
#   out_9 => view_52
#   shift_5 => view_34
#   split_2 => split_with_sizes_2
#   transpose_16 => permute_32
#   transpose_8 => permute_21
#   view_19 => view_36
#   x_10 => add_13
#   x_6 => add_8
#   x_7 => convolution_6
#   x_8 => mul_15, sigmoid_9
#   x_9 => convolution_7
#   x_flat_2 => view_35
#   x_norm_2 => add_9, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
# Graph fragment:
#   %buf65 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf65]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf40 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %mm_default_34 : Tensor "f32[10240, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg47_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [40, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [40, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [40, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [40, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %add_tensor_35 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg47_1), kwargs = {})
#   %view_51 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [40, 256, 256]), kwargs = {})
#   %permute_32 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_51, [0, 2, 1]), kwargs = {})
#   %view_52 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_32, [40, 256, 16, 16]), kwargs = {})
#   %add_13 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %view_52), kwargs = {})
#   return %add_13
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 52431872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x2), None)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/nd/cndj2hktte6tkngjrygxbu6pkafravdd4gderubyy6xlxq6jzyjy.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13, add_13, h_5], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_13 => add_17
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   h_5 => mul_20, sigmoid_12
#   input_16 => add_tensor_37, view_28
#   input_18 => add_tensor_35, view_51
#   out_10 => add_15
#   out_11 => add_16
#   out_5 => view_29
#   out_6 => add_10
#   out_7 => add_11
#   out_9 => view_52
#   shift_5 => view_34
#   shift_7 => view_57
#   split_2 => split_with_sizes_2
#   split_3 => split_with_sizes_3
#   transpose_16 => permute_32
#   transpose_8 => permute_21
#   view_19 => view_36
#   view_29 => view_59
#   x_10 => add_13
#   x_11 => convolution_8
#   x_12 => mul_19, sigmoid_11
#   x_13 => convolution_9
#   x_6 => add_8
#   x_7 => convolution_6
#   x_8 => mul_15, sigmoid_9
#   x_9 => convolution_7
#   x_flat_2 => view_35
#   x_flat_3 => view_58
#   x_norm_2 => add_9, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
#   x_norm_3 => add_14, mul_18, rsqrt_3, sub_5, var_mean_3, view_53, view_54
# Graph fragment:
#   %buf87 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf87]
#   %arg51_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %buf65 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf65]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf40 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %add_17 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_17]
#   %convolution_5 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_37 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [40, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [40, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [40, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [40, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [40, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [40, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %add_tensor_35 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg47_1), kwargs = {})
#   %view_51 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [40, 256, 256]), kwargs = {})
#   %permute_32 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_51, [0, 2, 1]), kwargs = {})
#   %view_52 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_32, [40, 256, 16, 16]), kwargs = {})
#   %add_13 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %view_52), kwargs = {})
#   %convolution_8 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_13, %arg48_1, %arg49_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_53 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_8, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_3 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_5 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_53, %getitem_20), kwargs = {})
#   %add_14 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_19, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_18 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %view_54 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_18, [40, 256, 16, 16]), kwargs = {})
#   %view_58 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_54, [40, 256, -1]), kwargs = {})
#   %add_15 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_58, %bmm_11), kwargs = {})
#   %view_59 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_15, [40, 256, 16, 16]), kwargs = {})
#   %view_57 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_23, [40, 256, 1, 1]), kwargs = {})
#   %add_16 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_59, %view_57), kwargs = {})
#   %sigmoid_11 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_16,), kwargs = {})
#   %mul_19 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sigmoid_11), kwargs = {})
#   %convolution_9 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_19, %arg50_1, %arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_17 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %mul_16), kwargs = {})
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   return %add_17,%mul_20
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 52431872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp12 * tmp13
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/i4/ci4f5xovq6mbluq2rm52nulrqrlpdi2y6qjhulqn4f7hctylwrct.py
# Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
# Graph fragment:
#   %arg52_1 : Tensor "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg52_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf90
triton_poi_fused_convolution_silu_28 = async_compile.triton('triton_poi_fused_convolution_silu_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/ej/cejzx7frsvfjwt5xbmsx33k5kpspg4kct5j737kb5xx7g5tgygjv.py
# Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   q_6 => permute_34
#   q_7 => clone_12
#   view_30 => view_60
# Graph fragment:
#   %buf91 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_60 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [40, 512, 64]), kwargs = {})
#   %permute_34 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_60, [0, 2, 1]), kwargs = {})
#   %clone_12 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_12
triton_poi_fused_clone_convolution_silu_transpose_view_29 = async_compile.triton('triton_poi_fused_clone_convolution_silu_transpose_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_silu_transpose_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15730688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_silu_transpose_view_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/bb/cbbnnkvsf6vpd5mzwu5k3oj4asekl4rezpum7zvipgggxutp3nve.py
# Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_4 => clone_13
#   q_7 => view_62
#   q_8 => permute_37
#   view_31 => view_63
# Graph fragment:
#   %mm_4 : Tensor "f32[2560, 512][512, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %view_62 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [40, 64, 512]), kwargs = {})
#   %view_63 : Tensor "f32[40, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_62, [40, 64, 8, 64]), kwargs = {})
#   %permute_37 : Tensor "f32[40, 8, 64, 64][32768, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_63, [0, 2, 1, 3]), kwargs = {})
#   %clone_13 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_13
triton_poi_fused__unsafe_view_clone_transpose_view_30 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15728640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 512*x1 + 32768*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/ql/cqldec5fycuejw4pp2tkrvte7b25r5ahj3m3y7s3rlrjnwxvatpy.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   k_5 => permute_38
#   kv_2 => unsqueeze_4
#   matmul_4 => clone_14
#   transpose_22 => permute_40
#   view_32 => view_64
# Graph fragment:
#   %mm_5 : Tensor "f32[40, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "f32[40, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_64 : Tensor "f32[40, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_24, [40, 1, 8, 64]), kwargs = {})
#   %permute_38 : Tensor "f32[40, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_64, [0, 2, 1, 3]), kwargs = {})
#   %permute_40 : Tensor "f32[40, 8, 64, 1][1024, 64, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_38, [0, 1, 3, 2]), kwargs = {})
#   %clone_14 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_14
triton_poi_fused_clone_split_transpose_unsqueeze_view_31 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 245760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/mt/cmt5kq7iw3v2xkos34w5pahoukqv6iisv6ubhms4qw7txris3ykn.py
# Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_5 => div_3, exp_3, sum_3
#   matmul_4 => view_68
# Graph fragment:
#   %bmm_12 : Tensor "f32[320, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_12]
#   %view_68 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_12, [40, 8, 64, 1]), kwargs = {})
#   %mul_tensor_36 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_68, 1), kwargs = {})
#   %amax_default_17 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_36, [-1], True), kwargs = {})
#   %sub_tensor_17 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_36, %amax_default_17), kwargs = {})
#   %mul_tensor_37 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_17, 0.125), kwargs = {})
#   %exp_3 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_37,), kwargs = {})
#   %sum_3 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [-1], True), kwargs = {})
#   %div_3 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_3, %sum_3), kwargs = {})
#   return %expand_10
triton_poi_fused__softmax_amax_mul_sub_view_32 = async_compile.triton('triton_poi_fused__softmax_amax_mul_sub_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_amax_mul_sub_view_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 245760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_amax_mul_sub_view_32(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 - tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.exp(tmp5)
    tmp7 = (tmp6 / tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/rv/crvawhcahnkgcyd4gc4gfrwfa24kkdww3rbw5tgs6hwlgen264qi.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   kv_2 => unsqueeze_4
#   matmul_5 => clone_15
#   v_13 => permute_39
#   view_33 => view_65
# Graph fragment:
#   %mm_5 : Tensor "f32[40, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "f32[40, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_65 : Tensor "f32[40, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_25, [40, 1, 8, 64]), kwargs = {})
#   %permute_39 : Tensor "f32[40, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_65, [0, 2, 1, 3]), kwargs = {})
#   %clone_15 : Tensor "f32[40, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_15
triton_poi_fused_clone_split_transpose_unsqueeze_view_33 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 245760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + 1024*x1), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/mi/cmic3nphtqd52kiblvb2dhwgfotpuh4odqx6o4f663dd7tlf7o5t.py
# Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_5 => view_71
#   out_12 => clone_16
#   transpose_23 => permute_41
# Graph fragment:
#   %bmm_13 : Tensor "f32[320, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %view_71 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_13, [40, 8, 64, 64]), kwargs = {})
#   %permute_41 : Tensor "f32[40, 64, 8, 64][32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_71, [0, 2, 1, 3]), kwargs = {})
#   %clone_16 : Tensor "f32[40, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_16
triton_poi_fused_clone_transpose_view_34 = async_compile.triton('triton_poi_fused_clone_transpose_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15728640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 8)
    x2 = ((xindex // 512) % 64)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 4096*x1 + 32768*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/ca/cca5hidlgdzq2xnxn2b66xelh7msnnwhcndvrtiteomikhfpsfy5.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_34, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
# Graph fragment:
#   %buf91 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %mm_default_33 : Tensor "f32[2560, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg57_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   return %add_18
triton_poi_fused_add_addmm_convolution_silu_transpose_view_35 = async_compile.triton('triton_poi_fused_add_addmm_convolution_silu_transpose_view_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_silu_transpose_view_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20975616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_silu_transpose_view_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/ey/ceyfluvioqzm3ocu6ifqkqalgmiw3yh3idc5bv7kkyojsdn5sr7w.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_34, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
# Graph fragment:
#   %arg58_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg58_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf103
triton_poi_fused_add_addmm_convolution_silu_transpose_view_36 = async_compile.triton('triton_poi_fused_add_addmm_convolution_silu_transpose_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_silu_transpose_view_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 18874368, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_silu_transpose_view_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/75/c755tbazkfan6djgrujiokuaplqdg7plavmahndwutcp4z4lfun3.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_34, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
#   x_norm_4 => var_mean_4, view_76
# Graph fragment:
#   %buf104 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf104]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [40, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_27,%buf106
triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37 = async_compile.triton('triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480, 'r0_': 5244928}}
)
@triton.jit
def triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1280
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
    r0_2 = (r0_index % 16)
    r0_3 = r0_index // 16
    x0 = (xindex % 32)
    x1 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 32768*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([XBLOCK, 1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp12, xmask)
    tl.store(out_ptr1 + (x4), tmp18, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/cg/ccgb5iyml7ov3hab5326envvgqrbe5dh4e2irqkq4rnnz2voiv5y.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_34, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
#   x_norm_4 => add_19, mul_22, rsqrt_4, sub_7, var_mean_4, view_76
# Graph fragment:
#   %buf104 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf104]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %getitem_27 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf106 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=buf106]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [40, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   return %mul_22
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5244928, 'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 512
    y4 = (yindex % 512)
    y5 = yindex // 16
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 512*x3 + 32768*y2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1024.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3 + 64*y6), tmp11, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/3o/c3oxk5tfa2k6feftuk6g2bhlxzjvokh2uog5qv6v5wo6d3g3zusx.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_34, view_74
#   out_13 => view_75
#   out_14 => add_20
#   out_15 => add_21
#   shift_9 => view_80
#   split_4 => split_with_sizes_4
#   transpose_24 => permute_43
#   view_39 => view_82
#   x_14 => add_18
#   x_15 => convolution_11
#   x_16 => mul_23, sigmoid_13
#   x_flat_4 => view_81
#   x_norm_4 => add_19, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
# Graph fragment:
#   %mul_22 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0" = PlaceHolder[target=mul_22]
#   %bmm_15 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0" = PlaceHolder[target=bmm_15]
#   %addmm_9 : Tensor "f32[40, 12800][12800, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [40, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [40, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [40, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [40, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[40, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [40, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   return %mul_23
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 10567680, 'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 64*y3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (12288 + y0 + 12800*y1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (y0 + 512*x2 + 32768*y1), tmp6, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/x3/cx3wvqcaeiz4bek6mazbsnr2sqadrhn3wowz6rklzbgqyabt6r75.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   input_20 => add_tensor_34, view_74
#   out_13 => view_75
#   out_14 => add_20
#   out_15 => add_21
#   q_10 => clone_18
#   q_9 => permute_45
#   shift_9 => view_80
#   split_4 => split_with_sizes_4
#   transpose_24 => permute_43
#   view_39 => view_82
#   view_40 => view_83
#   x_14 => add_18
#   x_15 => convolution_11
#   x_16 => mul_23, sigmoid_13
#   x_17 => convolution_12
#   x_flat_4 => view_81
#   x_norm_4 => add_19, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
# Graph fragment:
#   %buf116 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf116]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf91 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [40, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [40, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [40, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [40, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[40, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [40, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %view_83 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_24, [40, 512, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_83, [0, 2, 1]), kwargs = {})
#   %clone_18 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_45,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_18
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20975616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/6l/c6l6ot7izykkp3zvnabb7iyuqprsujmlv3hmxcuaqsldwbc4s3ox.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   input_20 => add_tensor_34, view_74
#   input_22 => add_tensor_32, view_97
#   out_13 => view_75
#   out_14 => add_20
#   out_15 => add_21
#   out_17 => view_98
#   shift_9 => view_80
#   split_4 => split_with_sizes_4
#   transpose_24 => permute_43
#   transpose_32 => permute_54
#   view_39 => view_82
#   x_14 => add_18
#   x_15 => convolution_11
#   x_16 => mul_23, sigmoid_13
#   x_17 => convolution_12
#   x_18 => add_23
#   x_flat_4 => view_81
#   x_norm_4 => add_19, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
# Graph fragment:
#   %buf116 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf116]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf91 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %mm_default_31 : Tensor "f32[2560, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_31]
#   %arg65_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [40, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [40, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [40, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [40, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[40, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [40, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %add_tensor_32 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_31, %arg65_1), kwargs = {})
#   %view_97 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [40, 64, 512]), kwargs = {})
#   %permute_54 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_97, [0, 2, 1]), kwargs = {})
#   %view_98 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_54, [40, 512, 8, 8]), kwargs = {})
#   %add_23 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %view_98), kwargs = {})
#   return %add_23
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26220544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x2), None)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/sy/csyczuesykn4folbk3c5hbf3ih7o4ngan3q2hdi2in6oswv27alu.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_17 => add_22
#   add_21 => add_27
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   h_8 => mul_28, sigmoid_16
#   input_20 => add_tensor_34, view_74
#   input_22 => add_tensor_32, view_97
#   out_13 => view_75
#   out_14 => add_20
#   out_15 => add_21
#   out_17 => view_98
#   out_18 => add_25
#   out_19 => add_26
#   shift_11 => view_103
#   shift_9 => view_80
#   split_4 => split_with_sizes_4
#   split_5 => split_with_sizes_5
#   transpose_24 => permute_43
#   transpose_32 => permute_54
#   view_39 => view_82
#   view_49 => view_105
#   view_50 => view_106
#   x_14 => add_18
#   x_15 => convolution_11
#   x_16 => mul_23, sigmoid_13
#   x_17 => convolution_12
#   x_18 => add_23
#   x_19 => convolution_13
#   x_20 => mul_27, sigmoid_15
#   x_21 => convolution_14
#   x_flat_4 => view_81
#   x_flat_5 => view_104
#   x_flat_6 => permute_56
#   x_norm_4 => add_19, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
#   x_norm_5 => add_24, mul_26, rsqrt_5, sub_9, var_mean_5, view_100, view_99
#   x_norm_6 => add_28, add_29, clone_24, mul_29, mul_30, rsqrt_6, sub_10, var_mean_6
# Graph fragment:
#   %buf137 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf137]
#   %arg69_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %buf116 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf116]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf91 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %add_27 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %getitem_39 : Tensor "f32[40, 64, 1][64, 1, 2560]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf140 : Tensor "f32[40, 64, 1][64, 1, 2560]cuda:0" = PlaceHolder[target=buf140]
#   %arg70_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg70_1]
#   %arg71_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg71_1]
#   %sigmoid_12 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_34 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [40, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [40, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [40, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [40, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [40, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [40, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[40, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [40, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %add_tensor_32 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_31, %arg65_1), kwargs = {})
#   %view_97 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [40, 64, 512]), kwargs = {})
#   %permute_54 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_97, [0, 2, 1]), kwargs = {})
#   %view_98 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_54, [40, 512, 8, 8]), kwargs = {})
#   %add_23 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %view_98), kwargs = {})
#   %convolution_13 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_23, %arg66_1, %arg67_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_99 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_13, [40, 32, 16, 64]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_99, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_5 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_9 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_99, %getitem_34), kwargs = {})
#   %add_24 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_33, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %mul_26 : Tensor "f32[40, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_5), kwargs = {})
#   %view_100 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_26, [40, 512, 8, 8]), kwargs = {})
#   %view_104 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_100, [40, 512, -1]), kwargs = {})
#   %add_25 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_104, %bmm_19), kwargs = {})
#   %view_105 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_25, [40, 512, 8, 8]), kwargs = {})
#   %view_103 : Tensor "f32[40, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_37, [40, 512, 1, 1]), kwargs = {})
#   %add_26 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_105, %view_103), kwargs = {})
#   %sigmoid_15 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_26,), kwargs = {})
#   %mul_27 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %sigmoid_15), kwargs = {})
#   %convolution_14 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_27, %arg68_1, %arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_27 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %mul_24), kwargs = {})
#   %sigmoid_16 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [40, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %clone_24 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_56,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_24, %getitem_39), kwargs = {})
#   %add_28 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_29 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_6), kwargs = {})
#   %mul_30 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %arg70_1), kwargs = {})
#   %add_29 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %arg71_1), kwargs = {})
#   return %add_27,%getitem_39,%buf140,%add_29
triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42 = async_compile.triton('triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 36710400}}
)
@triton.jit
def triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = (tmp21 / tmp23)
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tmp14 - tmp24
    tmp32 = 512.0
    tmp33 = (tmp30 / tmp32)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp12, xmask)
    tl.store(out_ptr2 + (r0_1 + 512*x0), tmp41, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/fq/cfqlhwuo5wq2mqcyucbxqd4sxbgpap726gu4jx45zactpdh4pwwk.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   matmul_8 => clone_25
#   q_12 => select
#   qkv => add_tensor_31, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
# Graph fragment:
#   %mm_default_30 : Tensor "f32[2560, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_30]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_31 : Tensor "f32[2560, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_30, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[40, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_31, [40, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[40, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [40, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 40, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select : Tensor "f32[40, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 0), kwargs = {})
#   %clone_25 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_16,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_25
triton_poi_fused_addmm_clone_permute_select_view_43 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15730688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_view_43(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 1536*x1 + 98304*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/he/chefm7zdoaepo7xtlmd2jxynvmipqkzv2ubtk2lrklbnmkv5zxkz.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   k_8 => select_1
#   matmul_8 => clone_26
#   qkv => add_tensor_31, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   transpose_35 => permute_59
# Graph fragment:
#   %mm_default_30 : Tensor "f32[2560, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_30]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_31 : Tensor "f32[2560, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_30, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[40, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_31, [40, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[40, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [40, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 40, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_1 : Tensor "f32[40, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 1), kwargs = {})
#   %permute_59 : Tensor "f32[40, 8, 64, 64][98304, 64, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_26 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_17,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_26
triton_poi_fused_addmm_clone_permute_select_transpose_view_44 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_transpose_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_transpose_view_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5244928, 'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_transpose_view_44(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 20480
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + 1536*x2 + 98304*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 64*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/xg/cxgf4trq4xgjlluvblkopfdxgwlvcbdpqfez6err7shmnzuzyg4y.py
# Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_9 => div_5, exp_5, sum_5
#   matmul_8 => view_112
# Graph fragment:
#   %bmm_20 : Tensor "f32[320, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %amax_default_15 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 20480]cuda:0" = PlaceHolder[target=amax_default_15]
#   %sum_5 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 20480]cuda:0" = PlaceHolder[target=sum_5]
#   %view_112 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_20, [40, 8, 64, 64]), kwargs = {})
#   %mul_tensor_32 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_112, 1), kwargs = {})
#   %amax_default_15 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_32, [-1], True), kwargs = {})
#   %sub_tensor_15 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_32, %amax_default_15), kwargs = {})
#   %mul_tensor_33 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_15, 0.125), kwargs = {})
#   %exp_5 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_33,), kwargs = {})
#   %sum_5 : Tensor "f32[40, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1], True), kwargs = {})
#   %div_5 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_5), kwargs = {})
#   return %amax_default_15,%sum_5,%expand_18
triton_per_fused__softmax_amax_mul_sub_view_45 = async_compile.triton('triton_per_fused__softmax_amax_mul_sub_view_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_amax_mul_sub_view_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 15728640}}
)
@triton.jit
def triton_per_fused__softmax_amax_mul_sub_view_45(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 20480
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp2 - tmp5
    tmp7 = 0.125
    tmp8 = tmp6 * tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = (tmp9 / tmp12)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp13, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/wh/cwhcnc6utu6j5q3ucaw6lh6rbck4gsmgn7mu5bpoxrvuwj5da564.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   out_20 => clone_27
#   qkv => add_tensor_31, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   v_20 => select_2
# Graph fragment:
#   %mm_default_30 : Tensor "f32[2560, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_30]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_31 : Tensor "f32[2560, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_30, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[40, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_31, [40, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[40, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [40, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 40, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_2 : Tensor "f32[40, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 2), kwargs = {})
#   %clone_27 : Tensor "f32[40, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_27
triton_poi_fused_addmm_clone_permute_select_view_46 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_view_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15730688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_view_46(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + 64*x2 + 1536*x1 + 98304*x3), None)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/lx/clx2epfyghdtiv4zcfwhsvckdkvoo6a45reb3ryoq54bfynxv5px.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_8 => mul_28, sigmoid_16
#   out_22 => add_tensor_30, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %add_27 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %mm_default_29 : Tensor "f32[2560, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_29]
#   %arg75_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %getitem_41 : Tensor "f32[40, 64, 1][64, 1, 2560]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf155 : Tensor "f32[40, 64, 1][64, 1, 2560]cuda:0" = PlaceHolder[target=buf155]
#   %arg76_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg76_1]
#   %arg77_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %sigmoid_16 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [40, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_30 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_29, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_30, [40, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   return %getitem_41,%buf155,%add_32
triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 20977664}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tmp6 - tmp16
    tmp24 = 512.0
    tmp25 = (tmp22 / tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp33, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/el/celylh5a2bgm4yfnlcppdob7jdg6uehhd22urbfzkh4ps3hfplfo.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
# Source node to ATen node mapping:
#   h_8 => mul_28, sigmoid_16
#   h_9 => convolution_15
#   out_22 => add_tensor_30, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %arg78_1 : Tensor "f32[512, 256, 4, 4][4096, 16, 4, 1]cuda:0" = PlaceHolder[target=arg78_1]
#   %sigmoid_16 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [40, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_30 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_29, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_30, [40, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [40, 512, 8, 8]), kwargs = {})
#   %convolution_15 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_119, %arg78_1, %arg79_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf159
triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_48 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 16777216, 'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_48(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 4096*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/e4/ce4uzogtegw63dfpfhz3j7d3d5gpyy5akyozxkwqfajrjsn7j4s6.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   gate => unsqueeze_7
#   h_10 => add_33
#   h_16_gated => mul_35
#   h_8 => mul_28, sigmoid_16
#   h_9 => convolution_15
#   input_26 => add_tensor_28
#   input_27 => sigmoid_18
#   out_22 => add_tensor_30, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   unsqueeze_4 => unsqueeze_6
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %buf160 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf160]
#   %arg79_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %mul_20 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_20]
#   %mm_default_27 : Tensor "f32[40, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_27]
#   %arg83_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %sigmoid_16 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[40, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[40, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [40, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_30 : Tensor "f32[2560, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_29, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_30, [40, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[40, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[40, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[40, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[40, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[40, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [40, 512, 8, 8]), kwargs = {})
#   %convolution_15 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_119, %arg78_1, %arg79_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_tensor_28 : Tensor "f32[40, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_27, %arg83_1), kwargs = {})
#   %sigmoid_18 : Tensor "f32[40, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_28,), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[40, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_18, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[40, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %mul_35 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_7), kwargs = {})
#   %add_33 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %mul_35), kwargs = {})
#   return %add_33
triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_49 = async_compile.triton('triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 20971520, 'x': 21014528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 256
    y0 = (yindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + 256*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp2 + tmp8
    tl.store(out_ptr0 + (y0 + 256*x2 + 65536*y1), tmp9, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/uq/cuqhi54hivg5rmei26gsz4owkrx7uslkkv5olutn6h75eekhqzbt.py
# Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
# Source node to ATen node mapping:
#   q_13 => permute_65
#   q_14 => mm_8, permute_66, view_121
#   view_52 => view_120
# Graph fragment:
#   %add_33 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %view_120 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_33, [40, 256, 256]), kwargs = {})
#   %permute_65 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_120, [0, 2, 1]), kwargs = {})
#   %view_121 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_65, [10240, 256]), kwargs = {})
#   %permute_66 : Tensor "f32[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg84_1, [1, 0]), kwargs = {})
#   %mm_8 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_121, %permute_66), kwargs = {})
#   return %buf165
triton_poi_fused_mm_t_transpose_view_50 = async_compile.triton('triton_poi_fused_mm_t_transpose_view_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_t_transpose_view_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31457280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_t_transpose_view_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 10240)
    x1 = xindex // 10240
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256*x1 + 65536*(x0 // 256) + ((x0 % 256))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/pc/cpcjy2u5ew4z76sfdcy2tzoq7v3x3dbqdvovrnc4h2d4x76mbw2g.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   input_28 => add_tensor_27, view_134
#   out_27 => view_135
#   transpose_44 => permute_74
#   x_22 => add_34
# Graph fragment:
#   %add_33 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_default_26 : Tensor "f32[10240, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_26]
#   %arg87_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %add_tensor_27 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_27, [40, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [40, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   return %add_34
triton_poi_fused_add_addmm_transpose_view_51 = async_compile.triton('triton_poi_fused_add_addmm_transpose_view_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_transpose_view_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 10485760, 'x': 31458304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_transpose_view_51(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 65536*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp4, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/y3/cy3uf4u6ikxnwufbvlx2otwwjajcl47wyqm6kvjqncbvgzzk4y5x.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   h_11 => mul_39, sigmoid_20
#   input_28 => add_tensor_27, view_134
#   out_27 => view_135
#   out_28 => add_36
#   out_29 => add_37
#   q_16 => permute_76
#   q_17 => clone_36
#   shift_13 => view_140
#   split_6 => split_with_sizes_6
#   transpose_44 => permute_74
#   view_61 => view_142
#   view_62 => view_143
#   x_22 => add_34
#   x_23 => convolution_16
#   x_24 => mul_38, sigmoid_19
#   x_25 => convolution_17
#   x_flat_7 => view_141
#   x_norm_7 => add_35, clone_35, mul_37, rsqrt_8, sub_14, var_mean_8, view_136, view_137
# Graph fragment:
#   %buf186 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf186]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %add_tensor_27 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_27, [40, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [40, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [40, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [40, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [40, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [40, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %view_143 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [40, 256, 256]), kwargs = {})
#   %permute_76 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_143, [0, 2, 1]), kwargs = {})
#   %clone_36 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_76,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_36
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_52 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 10485760, 'x': 31458304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 256*x2 + 65536*y1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2 + 256*y3), tmp6, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/j7/cj7unm5c7kaoi4s63mwga4noh7vqejyorlozgkv3gxtioq4cu4yp.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   h_11 => mul_39, sigmoid_20
#   input_28 => add_tensor_27, view_134
#   input_30 => add_tensor_26, view_157
#   out_27 => view_135
#   out_28 => add_36
#   out_29 => add_37
#   out_31 => view_158
#   shift_13 => view_140
#   split_6 => split_with_sizes_6
#   transpose_44 => permute_74
#   transpose_52 => permute_85
#   view_61 => view_142
#   x_22 => add_34
#   x_23 => convolution_16
#   x_24 => mul_38, sigmoid_19
#   x_25 => convolution_17
#   x_26 => add_39
#   x_flat_7 => view_141
#   x_norm_7 => add_35, clone_35, mul_37, rsqrt_8, sub_14, var_mean_8, view_136, view_137
# Graph fragment:
#   %buf186 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf186]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_default_25 : Tensor "f32[10240, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_25]
#   %arg95_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg95_1]
#   %add_tensor_27 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_27, [40, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [40, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [40, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [40, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [40, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [40, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %add_tensor_26 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_25, %arg95_1), kwargs = {})
#   %view_157 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_26, [40, 256, 256]), kwargs = {})
#   %permute_85 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1]), kwargs = {})
#   %view_158 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_85, [40, 256, 16, 16]), kwargs = {})
#   %add_39 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %view_158), kwargs = {})
#   return %add_39
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 10485760, 'x': 41945088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 256*x2 + 65536*y1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp10, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/cd/ccd2ciollzo2lldpb6wsat3cyj5lbvzdozp55uwqlmd6vlt7byqd.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   add_31 => add_43
#   h_11 => mul_39, sigmoid_20
#   h_12 => mul_43, sigmoid_22
#   input_28 => add_tensor_27, view_134
#   input_30 => add_tensor_26, view_157
#   out_27 => view_135
#   out_28 => add_36
#   out_29 => add_37
#   out_31 => view_158
#   out_32 => add_41
#   out_33 => add_42
#   q_19 => permute_87
#   q_20 => clone_42
#   shift_13 => view_140
#   shift_15 => view_163
#   split_6 => split_with_sizes_6
#   split_7 => split_with_sizes_7
#   transpose_44 => permute_74
#   transpose_52 => permute_85
#   view_61 => view_142
#   view_71 => view_165
#   view_72 => view_166
#   x_22 => add_34
#   x_23 => convolution_16
#   x_24 => mul_38, sigmoid_19
#   x_25 => convolution_17
#   x_26 => add_39
#   x_27 => convolution_18
#   x_28 => mul_42, sigmoid_21
#   x_29 => convolution_19
#   x_flat_7 => view_141
#   x_flat_8 => view_164
#   x_norm_7 => add_35, clone_35, mul_37, rsqrt_8, sub_14, var_mean_8, view_136, view_137
#   x_norm_8 => add_40, mul_41, rsqrt_9, sub_16, var_mean_9, view_159, view_160
# Graph fragment:
#   %buf208 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf208]
#   %arg99_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg99_1]
#   %buf186 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf186]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %add_43 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %add_tensor_27 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_27, [40, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [40, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [40, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [40, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [40, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [40, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %add_tensor_26 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_25, %arg95_1), kwargs = {})
#   %view_157 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_26, [40, 256, 256]), kwargs = {})
#   %permute_85 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1]), kwargs = {})
#   %view_158 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_85, [40, 256, 16, 16]), kwargs = {})
#   %add_39 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %view_158), kwargs = {})
#   %convolution_18 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_39, %arg96_1, %arg97_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_159 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_18, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_159, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_7 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_16 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_159, %getitem_52), kwargs = {})
#   %add_40 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_51, 1e-06), kwargs = {})
#   %rsqrt_9 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
#   %mul_41 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %rsqrt_9), kwargs = {})
#   %view_160 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_41, [40, 256, 16, 16]), kwargs = {})
#   %view_164 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [40, 256, -1]), kwargs = {})
#   %add_41 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_164, %bmm_29), kwargs = {})
#   %view_165 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_41, [40, 256, 16, 16]), kwargs = {})
#   %view_163 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_55, [40, 256, 1, 1]), kwargs = {})
#   %add_42 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_165, %view_163), kwargs = {})
#   %sigmoid_21 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_42,), kwargs = {})
#   %mul_42 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %sigmoid_21), kwargs = {})
#   %convolution_19 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_42, %arg98_1, %arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_43 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %mul_39), kwargs = {})
#   %sigmoid_22 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %view_166 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_43, [40, 256, 256]), kwargs = {})
#   %permute_87 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_166, [0, 2, 1]), kwargs = {})
#   %clone_42 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_87,), kwargs = {memory_format: torch.contiguous_format})
#   return %add_43,%clone_42
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62916608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 256)
    x2 = xindex // 65536
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1 + 256*x0 + 65536*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 + tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/os/cos2weezj62oum4o7gmupew5m6422pya6x46xmccms3j6bxy4xkn.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_12 => mul_43, sigmoid_22
#   input_32 => add_tensor_25, view_180
#   out_35 => view_181
#   transpose_60 => permute_96
#   x_30 => add_44
# Graph fragment:
#   %add_43 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %mm_default_24 : Tensor "f32[10240, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_24]
#   %arg103_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg103_1]
#   %sigmoid_22 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_25 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_25, [40, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [40, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   return %add_44
triton_poi_fused_add_addmm_silu_transpose_view_55 = async_compile.triton('triton_poi_fused_add_addmm_silu_transpose_view_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_silu_transpose_view_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41944064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_silu_transpose_view_55(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/te/cteg5tuibpbjbzolutgjl4kfyeoa4mg55rttknzpvwm7jp2jisgo.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_35 => add_48
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   input_32 => add_tensor_25, view_180
#   out_35 => view_181
#   out_36 => add_46
#   out_37 => add_47
#   q_22 => permute_98
#   q_23 => clone_48
#   shift_17 => view_186
#   split_8 => split_with_sizes_8
#   transpose_60 => permute_96
#   view_81 => view_188
#   view_82 => view_189
#   x_30 => add_44
#   x_31 => convolution_20
#   x_32 => mul_46, sigmoid_23
#   x_33 => convolution_21
#   x_flat_9 => view_187
#   x_norm_9 => add_45, mul_45, rsqrt_10, sub_18, var_mean_10, view_182, view_183
# Graph fragment:
#   %buf231 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf231]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %sigmoid_22 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_25 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_25, [40, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [40, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [40, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [40, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [40, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [40, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %view_189 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_47, [40, 256, 256]), kwargs = {})
#   %permute_98 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_189, [0, 2, 1]), kwargs = {})
#   %clone_48 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_98,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_48
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_56 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41944064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_56(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/pa/cpatld2bzaurxums7ejktw5ygxdisglypytzximpo4ba2xiggtyq.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_35 => add_48
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   input_32 => add_tensor_25, view_180
#   input_34 => add_tensor_24, view_203
#   out_35 => view_181
#   out_36 => add_46
#   out_37 => add_47
#   out_39 => view_204
#   shift_17 => view_186
#   split_8 => split_with_sizes_8
#   transpose_60 => permute_96
#   transpose_68 => permute_107
#   view_81 => view_188
#   x_30 => add_44
#   x_31 => convolution_20
#   x_32 => mul_46, sigmoid_23
#   x_33 => convolution_21
#   x_34 => add_49
#   x_flat_9 => view_187
#   x_norm_9 => add_45, mul_45, rsqrt_10, sub_18, var_mean_10, view_182, view_183
# Graph fragment:
#   %buf231 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf231]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %mm_default_23 : Tensor "f32[10240, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %arg111_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg111_1]
#   %sigmoid_22 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_25 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_25, [40, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [40, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [40, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [40, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [40, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [40, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %add_tensor_24 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %arg111_1), kwargs = {})
#   %view_203 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [40, 256, 256]), kwargs = {})
#   %permute_107 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_203, [0, 2, 1]), kwargs = {})
#   %view_204 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_107, [40, 256, 16, 16]), kwargs = {})
#   %add_49 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %view_204), kwargs = {})
#   return %add_49
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 52430848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp9 = tl.load(in_out_ptr0 + (x2), None)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/yf/cyfynr52ssfirvodnvdw5f4vrlc55zl2gzjkazcqzp5toc37si6z.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_35 => add_48
#   add_39 => add_53
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   h_14 => mul_51, sigmoid_26
#   input_32 => add_tensor_25, view_180
#   input_34 => add_tensor_24, view_203
#   out_35 => view_181
#   out_36 => add_46
#   out_37 => add_47
#   out_39 => view_204
#   out_40 => add_51
#   out_41 => add_52
#   q_25 => permute_109
#   q_26 => clone_54
#   shift_17 => view_186
#   shift_19 => view_209
#   split_8 => split_with_sizes_8
#   split_9 => split_with_sizes_9
#   transpose_60 => permute_96
#   transpose_68 => permute_107
#   view_81 => view_188
#   view_91 => view_211
#   view_92 => view_212
#   x_30 => add_44
#   x_31 => convolution_20
#   x_32 => mul_46, sigmoid_23
#   x_33 => convolution_21
#   x_34 => add_49
#   x_35 => convolution_22
#   x_36 => mul_50, sigmoid_25
#   x_37 => convolution_23
#   x_flat_10 => view_210
#   x_flat_9 => view_187
#   x_norm_10 => add_50, mul_49, rsqrt_11, sub_20, var_mean_11, view_205, view_206
#   x_norm_9 => add_45, mul_45, rsqrt_10, sub_18, var_mean_10, view_182, view_183
# Graph fragment:
#   %buf253 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf253]
#   %arg115_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg115_1]
#   %buf231 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf231]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %add_53 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_53]
#   %sigmoid_22 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_25 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_25, [40, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [40, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [40, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [40, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [40, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [40, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %add_tensor_24 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %arg111_1), kwargs = {})
#   %view_203 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [40, 256, 256]), kwargs = {})
#   %permute_107 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_203, [0, 2, 1]), kwargs = {})
#   %view_204 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_107, [40, 256, 16, 16]), kwargs = {})
#   %add_49 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %view_204), kwargs = {})
#   %convolution_22 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_49, %arg112_1, %arg113_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_205 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_22, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_205, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_9 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_20 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_205, %getitem_66), kwargs = {})
#   %add_50 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_65, 1e-06), kwargs = {})
#   %rsqrt_11 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_50,), kwargs = {})
#   %mul_49 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_11), kwargs = {})
#   %view_206 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_49, [40, 256, 16, 16]), kwargs = {})
#   %view_210 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_206, [40, 256, -1]), kwargs = {})
#   %add_51 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_210, %bmm_37), kwargs = {})
#   %view_211 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_51, [40, 256, 16, 16]), kwargs = {})
#   %view_209 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_69, [40, 256, 1, 1]), kwargs = {})
#   %add_52 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_211, %view_209), kwargs = {})
#   %sigmoid_25 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_52,), kwargs = {})
#   %mul_50 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_52, %sigmoid_25), kwargs = {})
#   %convolution_23 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_50, %arg114_1, %arg115_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_53 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_23, %mul_47), kwargs = {})
#   %sigmoid_26 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %view_212 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_51, [40, 256, 256]), kwargs = {})
#   %permute_109 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_212, [0, 2, 1]), kwargs = {})
#   %clone_54 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_109,), kwargs = {memory_format: torch.contiguous_format})
#   return %add_53,%clone_54
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 73402368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp12 * tmp13
    tl.store(in_out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/db/cdbb4kqdev273dpxx7kbmhckq7t3txj62vpsy4zbsp3dffvlapbj.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_43 => add_58
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   input_36 => add_tensor_23, view_226
#   out_43 => view_227
#   out_44 => add_56
#   out_45 => add_57
#   shift_21 => view_232
#   split_10 => split_with_sizes_10
#   transpose_76 => permute_118
#   view_101 => view_234
#   x_38 => add_54
#   x_39 => convolution_24
#   x_40 => mul_54, sigmoid_27
#   x_41 => convolution_25
#   x_flat_11 => view_233
#   x_norm_11 => add_55, mul_53, rsqrt_12, sub_22, var_mean_12, view_228, view_229
# Graph fragment:
#   %buf275 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf275]
#   %arg123_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg123_1]
#   %add_53 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_53]
#   %sigmoid_26 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_23 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [40, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [40, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [40, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [40, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [40, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [40, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   return %mul_55
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41944064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/za/czapbm2e3c5dhku4af4djvs4rlpssuxgwaxyevy7snmtoae5pkbt.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_43 => add_58
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   h_16 => convolution_26
#   input_36 => add_tensor_23, view_226
#   out_43 => view_227
#   out_44 => add_56
#   out_45 => add_57
#   shift_21 => view_232
#   split_10 => split_with_sizes_10
#   transpose_76 => permute_118
#   view_101 => view_234
#   x_38 => add_54
#   x_39 => convolution_24
#   x_40 => mul_54, sigmoid_27
#   x_41 => convolution_25
#   x_flat_11 => view_233
#   x_norm_11 => add_55, mul_53, rsqrt_12, sub_22, var_mean_12, view_228, view_229
# Graph fragment:
#   %arg124_1 : Tensor "f32[256, 128, 4, 4][2048, 16, 4, 1]cuda:0" = PlaceHolder[target=arg124_1]
#   %sigmoid_26 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_23 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [40, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [40, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [40, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [40, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [40, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [40, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   %convolution_26 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_55, %arg124_1, %arg125_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf277
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4194304, 'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 2048*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/ic/cic4mvaw5j5czvqmqz6kdjhqr6qb3f2mo5fa5fhhtd6xjivisbhc.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   add_43 => add_58
#   gate_1 => unsqueeze_14
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   h_16 => convolution_26
#   h_17 => add_59
#   h_32_gated => mul_57
#   input_36 => add_tensor_23, view_226
#   input_40 => add_tensor_21
#   input_41 => sigmoid_30
#   out_43 => view_227
#   out_44 => add_56
#   out_45 => add_57
#   shift_21 => view_232
#   split_10 => split_with_sizes_10
#   transpose_76 => permute_118
#   unsqueeze_11 => unsqueeze_13
#   view_101 => view_234
#   x_38 => add_54
#   x_39 => convolution_24
#   x_40 => mul_54, sigmoid_27
#   x_41 => convolution_25
#   x_flat_11 => view_233
#   x_norm_11 => add_55, mul_53, rsqrt_12, sub_22, var_mean_12, view_228, view_229
# Graph fragment:
#   %buf278 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf278]
#   %arg125_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %mul_12 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_12]
#   %mm_default_20 : Tensor "f32[40, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_20]
#   %arg129_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg129_1]
#   %sigmoid_26 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_23 : Tensor "f32[10240, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [40, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[40, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[40, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [40, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [40, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[40, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [40, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [40, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[40, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [40, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[40, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [40, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[40, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   %convolution_26 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_55, %arg124_1, %arg125_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_tensor_21 : Tensor "f32[40, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_20, %arg129_1), kwargs = {})
#   %sigmoid_30 : Tensor "f32[40, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_21,), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[40, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_30, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[40, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_57 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_14), kwargs = {})
#   %add_59 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %mul_57), kwargs = {})
#   return %add_59
triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_61 = async_compile.triton('triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_61', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83907584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_61(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 131072
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp2 + tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/hi/chi7mmwfgn7uwfgcd3nld3epnvfd4rxzc5bgkzax76gyr7j3ljvz.py
# Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_59 => add_79
#   h_22 => mul_72, sigmoid_40
#   input_42 => var_mean_18, view_270
#   out_54 => add_77
#   out_55 => add_78
#   shift_31 => view_267
#   split_15 => split_with_sizes_15
#   view_126 => view_269
#   x_54 => convolution_35
#   x_55 => mul_71, sigmoid_39
#   x_56 => convolution_36
#   x_flat_16 => view_268
#   x_norm_16 => add_76, mul_70, rsqrt_17, sub_27, var_mean_17, view_263, view_264
# Graph fragment:
#   %buf336 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf336]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %mul_69 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_69]
#   %convolution_35 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_69, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_263 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_35, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_263, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_15 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub_27 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_263, %getitem_98), kwargs = {})
#   %add_76 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_97, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_70 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_17), kwargs = {})
#   %view_264 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [40, 128, 32, 32]), kwargs = {})
#   %view_268 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_264, [40, 128, -1]), kwargs = {})
#   %add_77 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_268, %bmm_51), kwargs = {})
#   %view_269 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [40, 128, 32, 32]), kwargs = {})
#   %view_267 : Tensor "f32[40, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_101, [40, 128, 1, 1]), kwargs = {})
#   %add_78 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_269, %view_267), kwargs = {})
#   %sigmoid_39 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convolution_36 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_71, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %mul_69), kwargs = {})
#   %sigmoid_40 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_72 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_40), kwargs = {})
#   %view_270 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_72, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_103,%buf338
triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_62 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480, 'r0_': 41943552}}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_62(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1280
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask & xmask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tl.store(out_ptr1 + (x4), tmp12, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/zh/czhpwxv4hbmg4hbblncubixixf2p3sq3x4xcvgiv37ricv3ugdqb.py
# Topologically Sorted Source Nodes: [unsqueeze_13, s_emb_1, cat_3], Original ATen: [aten.unsqueeze, aten.expand, aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_3
#   s_emb_1 => expand_40
#   unsqueeze_13 => unsqueeze_23
# Graph fragment:
#   %arg154_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg154_1]
#   %unsqueeze_23 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg154_1, 0), kwargs = {})
#   %expand_40 : Tensor "f32[40, 256][0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_23, [40, -1]), kwargs = {})
#   %cat_3 : Tensor "f32[40, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%addmm_26, %expand_40], -1), kwargs = {})
#   return %buf353
triton_poi_fused_cat_expand_unsqueeze_63 = async_compile.triton('triton_poi_fused_cat_expand_unsqueeze_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_expand_unsqueeze_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 82944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_expand_unsqueeze_63(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + 512*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/rt/crtu26w7ci3cjokicbnzag7e5bn4qf6fjfyvbf4ynvjwsalrohxj.py
# Topologically Sorted Source Nodes: [lt, zeros_like, full_like, sigma, gt, add_noise], Original ATen: [aten.lt, aten.zeros_like, aten.full_like, aten.where, aten.gt, aten.any]
# Source node to ATen node mapping:
#   add_noise => any_1
#   full_like => full_default_4
#   gt => gt
#   lt => lt
#   sigma => where
#   zeros_like => full_default_3
# Graph fragment:
#   %arg0_1 : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %where : Tensor "f32[40][1]cuda:0" = PlaceHolder[target=where]
#   %lt : Tensor "b8[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, 0.5), kwargs = {})
#   %full_default_3 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([40], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_4 : Tensor "f32[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([40], 0.3), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[40][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%lt, %full_default_3, %full_default_4), kwargs = {})
#   %gt : Tensor "b8[40][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
#   %any_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.default](args = (%gt,), kwargs = {})
#   return %where,%any_1
triton_per_fused_any_full_like_gt_lt_where_zeros_like_64 = async_compile.triton('triton_per_fused_any_full_like_gt_lt_where_zeros_like_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_full_like_gt_lt_where_zeros_like_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 480}}
)
@triton.jit
def triton_per_fused_any_full_like_gt_lt_where_zeros_like_64(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 40
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = 0.5
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = 0.3
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5 > tmp3
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, False)
    tmp10 = triton_helpers.any(tmp9, 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp5, r0_mask)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp10, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/el/celml6bys6xymc5jjacjgtvobaakjnn6nduibcuscimj4vkxeytf.py
# Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42, x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112, x_113, add_119, h_45, input_86, input_43, input_87], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_119 => add_161
#   add_59 => add_79
#   h_22 => mul_72, sigmoid_40
#   h_45 => mul_148, sigmoid_82
#   input_42 => add_80, add_81, mul_73, mul_74, rsqrt_18, sub_28, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, var_mean_18, view_270, view_271
#   input_43 => mul_75, sigmoid_41
#   input_86 => add_162, add_163, mul_149, mul_150, rsqrt_37, sub_57, unsqueeze_37, unsqueeze_38, unsqueeze_39, unsqueeze_40, unsqueeze_41, unsqueeze_42, var_mean_37, view_542, view_543
#   input_87 => mul_151, sigmoid_83
#   out_110 => add_159
#   out_111 => add_160
#   out_54 => add_77
#   out_55 => add_78
#   shift_31 => view_267
#   shift_63 => view_539
#   split_15 => split_with_sizes_15
#   split_31 => split_with_sizes_31
#   view_126 => view_269
#   view_253 => view_541
#   x_111 => convolution_73
#   x_112 => mul_147, sigmoid_81
#   x_113 => convolution_74
#   x_54 => convolution_35
#   x_55 => mul_71, sigmoid_39
#   x_56 => convolution_36
#   x_flat_16 => view_268
#   x_flat_33 => view_540
#   x_norm_16 => add_76, mul_70, rsqrt_17, sub_27, var_mean_17, view_263, view_264
#   x_norm_33 => add_158, mul_146, rsqrt_36, sub_56, var_mean_36, view_535, view_536
# Graph fragment:
#   %buf336 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf336]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %mul_69 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_69]
#   %getitem_103 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=getitem_103]
#   %buf338 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=buf338]
#   %arg150_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg150_1]
#   %arg151_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg151_1]
#   %add_81 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_81]
#   %buf676 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf676]
#   %mul_145 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_145]
#   %getitem_207 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=getitem_207]
#   %buf678 : Tensor "f32[40, 32, 1, 1][32, 1, 1280, 1280]cuda:0" = PlaceHolder[target=buf678]
#   %add_163 : Tensor "f32[40, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_163]
#   %convolution_35 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_69, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_263 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_35, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_263, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_15 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub_27 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_263, %getitem_98), kwargs = {})
#   %add_76 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_97, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_70 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_17), kwargs = {})
#   %view_264 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [40, 128, 32, 32]), kwargs = {})
#   %view_268 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_264, [40, 128, -1]), kwargs = {})
#   %add_77 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_268, %bmm_51), kwargs = {})
#   %view_269 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [40, 128, 32, 32]), kwargs = {})
#   %view_267 : Tensor "f32[40, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_101, [40, 128, 1, 1]), kwargs = {})
#   %add_78 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_269, %view_267), kwargs = {})
#   %sigmoid_39 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convolution_36 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_71, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %mul_69), kwargs = {})
#   %sigmoid_40 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_72 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_40), kwargs = {})
#   %view_270 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_72, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %convolution_73 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_145, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_535 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_73, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean_36 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_535, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_31 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_30, [1536, 1536, 128], 1), kwargs = {})
#   %sub_56 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_535, %getitem_202), kwargs = {})
#   %add_158 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_201, 1e-06), kwargs = {})
#   %rsqrt_36 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_158,), kwargs = {})
#   %mul_146 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %rsqrt_36), kwargs = {})
#   %view_536 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_146, [40, 128, 32, 32]), kwargs = {})
#   %view_540 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_536, [40, 128, -1]), kwargs = {})
#   %add_159 : Tensor "f32[40, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_540, %bmm_103), kwargs = {})
#   %view_541 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_159, [40, 128, 32, 32]), kwargs = {})
#   %view_539 : Tensor "f32[40, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_205, [40, 128, 1, 1]), kwargs = {})
#   %add_160 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_541, %view_539), kwargs = {})
#   %sigmoid_81 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_160,), kwargs = {})
#   %mul_147 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_160, %sigmoid_81), kwargs = {})
#   %convolution_74 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_147, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_161 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_74, %mul_145), kwargs = {})
#   %sigmoid_82 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_161,), kwargs = {})
#   %mul_148 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %sigmoid_82), kwargs = {})
#   %view_542 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_148, [40, 32, 4, 1024]), kwargs = {})
#   %var_mean_37 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_542, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %getitem_103), kwargs = {})
#   %add_80 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_102, 1e-05), kwargs = {})
#   %rsqrt_18 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   %mul_73 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_18), kwargs = {})
#   %view_271 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_73, [40, 128, 32, 32]), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, 0), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, 2), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 3), kwargs = {})
#   %mul_74 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_271, %unsqueeze_17), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg151_1, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 3), kwargs = {})
#   %add_81 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_20), kwargs = {})
#   %sigmoid_41 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %sub_57 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_542, %getitem_207), kwargs = {})
#   %add_162 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_206, 1e-05), kwargs = {})
#   %rsqrt_37 : Tensor "f32[40, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_162,), kwargs = {})
#   %mul_149 : Tensor "f32[40, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %rsqrt_37), kwargs = {})
#   %view_543 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_149, [40, 128, 32, 32]), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, 0), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_37, 2), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, 3), kwargs = {})
#   %mul_150 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_543, %unsqueeze_39), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg151_1, 0), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, 2), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_41, 3), kwargs = {})
#   %add_163 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %unsqueeze_42), kwargs = {})
#   %sigmoid_83 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_151 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_83), kwargs = {})
#   return %add_81,%mul_75,%add_163,%mul_151
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_65 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_65', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167773696}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_65(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 131072
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr1 + (x3), None)
    tmp24 = tl.load(in_ptr6 + (x3), None)
    tmp28 = tl.load(in_ptr7 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 4096.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 + tmp1
    tmp25 = tmp23 + tmp24
    tmp26 = tl.sigmoid(tmp25)
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 - tmp28
    tmp31 = (tmp30 / tmp10)
    tmp32 = tmp31 + tmp12
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp29 * tmp33
    tmp35 = tmp34 * tmp16
    tmp36 = tmp35 + tmp18
    tmp37 = tl.sigmoid(tmp36)
    tmp38 = tmp36 * tmp37
    tl.store(in_out_ptr0 + (x3), tmp21, None)
    tl.store(in_out_ptr1 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/xg/cxgywzoz257kmqvsxavm2bsveg75ijsnz5q7d3ntf5qsr6q2zkqm.py
# Topologically Sorted Source Nodes: [input_43, input_44, input_87, input_88], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_75, sigmoid_41
#   input_44 => convolution_37
#   input_87 => mul_151, sigmoid_83
#   input_88 => convolution_75
# Graph fragment:
#   %arg152_1 : Tensor "f32[4, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg152_1]
#   %sigmoid_41 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convolution_37 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_151 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_83), kwargs = {})
#   %convolution_75 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_151, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf684,%buf691
triton_poi_fused_convolution_silu_66 = async_compile.triton('triton_poi_fused_convolution_silu_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 73728, 'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_66(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/pi/cpif77tdopwcrudd5a65jcc7laiiua4b63av2elfv2qp6g4r5o6i.py
# Topologically Sorted Source Nodes: [input_43, input_44, std_target, input_87, input_88, sub_1, v_pred, std_pred], Original ATen: [aten.silu, aten.convolution, aten.std, aten.sub, aten.add]
# Source node to ATen node mapping:
#   input_43 => mul_75, sigmoid_41
#   input_44 => convolution_37
#   input_87 => mul_151, sigmoid_83
#   input_88 => convolution_75
#   std_pred => var_1
#   std_target => var
#   sub_1 => sub_59
#   v_pred => add_164
# Graph fragment:
#   %buf685 : Tensor "f32[40, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf685]
#   %arg153_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %buf692 : Tensor "f32[40, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf692]
#   %sigmoid_41 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convolution_37 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%convolution_37, [1, 2, 3]), kwargs = {correction: 1.0, keepdim: True})
#   %sigmoid_83 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_151 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_83), kwargs = {})
#   %convolution_75 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_151, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_59 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %convolution_75), kwargs = {})
#   %full_default : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 5.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -0.26666666666666666), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scalar_tensor_default_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg155_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default_3, torch.float64), kwargs = {})
#   %mul_tensor : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %convert_element_type_default), kwargs = {})
#   %add_tensor : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_1, %mul_tensor), kwargs = {})
#   %maximum_default : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default, %add_tensor), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%maximum_default, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %convert_element_type_default_1), kwargs = {})
#   %add_164 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_75, %mul_tensor_1), kwargs = {})
#   %var_1 : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_164, [1, 2, 3]), kwargs = {correction: 1.0, keepdim: True})
#   return %buf687,%buf694
triton_red_fused_add_convolution_silu_std_sub_67 = async_compile.triton('triton_red_fused_add_convolution_silu_std_sub_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_silu_std_sub_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 640, 'r0_': 1310736}}
)
@triton.jit
def triton_red_fused_add_convolution_silu_std_sub_67(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 40
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp25_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp25_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp25_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        r0_1 = (r0_index % 4)
        tmp0 = tl.load(in_ptr0 + (r0_3 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (r0_3 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
        tmp11 = tmp10 + tmp1
        tmp12 = tmp2 - tmp11
        tmp13 = tl.full([1, 1], -0.26666666666666666, tl.float64)
        tmp14 = ks0
        tmp15 = tmp14.to(tl.float64)
        tmp16 = tmp13 * tmp15
        tmp17 = tl.full([1, 1], 5.0, tl.float64)
        tmp18 = tmp17 + tmp16
        tmp19 = tl.full([1, 1], 1.0, tl.float64)
        tmp20 = triton_helpers.maximum(tmp19, tmp18)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp12 * tmp21
        tmp23 = tmp11 + tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp25_mean_next, tmp25_m2_next, tmp25_weight_next = triton_helpers.welford_reduce(
            tmp24, tmp25_mean, tmp25_m2, tmp25_weight, roffset == 0
        )
        tmp25_mean = tl.where(r0_mask & xmask, tmp25_mean_next, tmp25_mean)
        tmp25_m2 = tl.where(r0_mask & xmask, tmp25_m2_next, tmp25_m2)
        tmp25_weight = tl.where(r0_mask & xmask, tmp25_weight_next, tmp25_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp26, tmp27, tmp28 = triton_helpers.welford(tmp25_mean, tmp25_m2, tmp25_weight, 1)
    tmp25 = tmp26[:, None]
    tmp29 = tmp27[:, None]
    tmp30 = tmp28[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/oy/coyvtjr4lwm6z3ncugn7x3ybq2aql3gmtijqxns2dobtgo57knrt.py
# Topologically Sorted Source Nodes: [input_43, input_44, std_target, input_87, input_88, sub_1, v_pred, std_pred, add_121, rescale_factor, v_pred_rescaled, mul_31, mul_32, v_guided, mul_33, x_next], Original ATen: [aten.silu, aten.convolution, aten.std, aten.sub, aten.add, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_121 => add_165
#   input_43 => mul_75, sigmoid_41
#   input_44 => convolution_37
#   input_87 => mul_151, sigmoid_83
#   input_88 => convolution_75
#   mul_31 => mul_155
#   mul_32 => mul_156
#   mul_33 => mul_157
#   rescale_factor => div_22
#   std_pred => sqrt_1, var_1
#   std_target => sqrt, var
#   sub_1 => sub_59
#   v_guided => add_166
#   v_pred => add_164
#   v_pred_rescaled => mul_154
#   x_next => add_167
# Graph fragment:
#   %buf692 : Tensor "f32[40, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf692]
#   %arg153_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %buf685 : Tensor "f32[40, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf685]
#   %buf687 : Tensor "f32[40, 1, 1, 1][1, 40, 40, 40]cuda:0" = PlaceHolder[target=buf687]
#   %buf694 : Tensor "f32[40, 1, 1, 1][1, 40, 40, 40]cuda:0" = PlaceHolder[target=buf694]
#   %arg25_1 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %mul_154 : Tensor "f32[40, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=mul_154]
#   %sigmoid_41 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convolution_37 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%convolution_37, [1, 2, 3]), kwargs = {correction: 1.0, keepdim: True})
#   %sqrt : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%var,), kwargs = {})
#   %sigmoid_83 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_151 : Tensor "f32[40, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_83), kwargs = {})
#   %convolution_75 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_151, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_59 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %convolution_75), kwargs = {})
#   %full_default : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 5.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -0.26666666666666666), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scalar_tensor_default_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%arg155_1,), kwargs = {})
#   %convert_element_type_default : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%scalar_tensor_default_3, torch.float64), kwargs = {})
#   %mul_tensor : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_2, %convert_element_type_default), kwargs = {})
#   %add_tensor : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_1, %mul_tensor), kwargs = {})
#   %maximum_default : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default, %add_tensor), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%maximum_default, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %convert_element_type_default_1), kwargs = {})
#   %add_164 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_75, %mul_tensor_1), kwargs = {})
#   %var_1 : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_164, [1, 2, 3]), kwargs = {correction: 1.0, keepdim: True})
#   %sqrt_1 : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%var_1,), kwargs = {})
#   %add_165 : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_1, 1e-08), kwargs = {})
#   %div_22 : Tensor "f32[40, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt, %add_165), kwargs = {})
#   %mul_154 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, %div_22), kwargs = {})
#   %mul_155 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, 0.7), kwargs = {})
#   %mul_156 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, 0.30000000000000004), kwargs = {})
#   %add_166 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %mul_156), kwargs = {})
#   %mul_157 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.06666666666666667), kwargs = {})
#   %add_167 : Tensor "f32[40, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg25_1, %mul_157), kwargs = {})
#   return %mul_154,%add_167
triton_poi_fused_add_convolution_div_mul_silu_std_sub_68 = async_compile.triton('triton_poi_fused_add_convolution_div_mul_silu_std_sub_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mul_silu_std_sub_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1966080, 'x': 655368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mul_silu_std_sub_68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 1024
    y0 = (yindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 4*y3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (y0 + 1024*x2 + 4096*y1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp5 = tmp4 - tmp2
    tmp6 = tl.full([1, 1], -0.26666666666666666, tl.float64)
    tmp7 = ks0
    tmp8 = tmp7.to(tl.float64)
    tmp9 = tmp6 * tmp8
    tmp10 = tl.full([1, 1], 5.0, tl.float64)
    tmp11 = tmp10 + tmp9
    tmp12 = tl.full([1, 1], 1.0, tl.float64)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp5 * tmp14
    tmp16 = tmp2 + tmp15
    tmp18 = 4095.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = libdevice.sqrt(tmp19)
    tmp22 = (tmp21 / tmp18)
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = 1e-08
    tmp25 = tmp23 + tmp24
    tmp26 = (tmp20 / tmp25)
    tmp27 = tmp16 * tmp26
    tmp29 = 0.7
    tmp30 = tmp27 * tmp29
    tmp31 = 0.30000000000000004
    tmp32 = tmp16 * tmp31
    tmp33 = tmp30 + tmp32
    tmp34 = 0.06666666666666667
    tmp35 = tmp33 * tmp34
    tmp36 = tmp28 + tmp35
    tl.store(out_ptr1 + (y0 + 1024*x2 + 4096*y1), tmp36, xmask)
''', device_str='cuda')

def partition_0(args):
    arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg154_1, arg150_1, arg151_1, arg152_1, arg153_1, s57 = args
    args.clear()
    s57 = s57
    assert_size_stride(arg25_1, (40, 4, 32, 32), (4096, 1024, 32, 1))
    assert_size_stride(arg23_1, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg0_1, (40, ), (1, ))
    assert_size_stride(arg1_1, (1024, 256), (256, 1))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg3_1, (256, 1024), (1024, 1))
    assert_size_stride(arg6_1, (40, ), (1, ))
    assert_size_stride(arg5_1, (4, 256), (256, 1))
    assert_size_stride(arg7_1, (512, 512), (512, 1))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, 512), (512, 1))
    assert_size_stride(arg11_1, (256, 256), (256, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (3200, ), (1, ))
    assert_size_stride(arg13_1, (3200, 256), (256, 1))
    assert_size_stride(arg28_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (512, 256), (256, 1))
    assert_size_stride(arg32_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, 256), (256, 1))
    assert_size_stride(arg38_1, (256, 256), (256, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, 256), (256, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (6400, ), (1, ))
    assert_size_stride(arg17_1, (6400, 256), (256, 1))
    assert_size_stride(arg45_1, (512, 256), (256, 1))
    assert_size_stride(arg42_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, 256), (256, 1))
    assert_size_stride(arg46_1, (256, 256), (256, 1))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (1024, 256), (256, 1))
    assert_size_stride(arg50_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (512, 512), (512, 1))
    assert_size_stride(arg56_1, (512, 512), (512, 1))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (256, 256), (256, 1))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (12800, ), (1, ))
    assert_size_stride(arg21_1, (12800, 256), (256, 1))
    assert_size_stride(arg63_1, (1024, 256), (256, 1))
    assert_size_stride(arg60_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, 512), (512, 1))
    assert_size_stride(arg64_1, (512, 512), (512, 1))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (1536, 512), (512, 1))
    assert_size_stride(arg73_1, (1536, ), (1, ))
    assert_size_stride(arg74_1, (512, 512), (512, 1))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, 256), (256, 1))
    assert_size_stride(arg78_1, (512, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(arg80_1, (256, 256), (256, 1))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, 256), (256, 1))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (256, 256), (256, 1))
    assert_size_stride(arg86_1, (256, 256), (256, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (512, 256), (256, 1))
    assert_size_stride(arg90_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, 256), (256, 1))
    assert_size_stride(arg94_1, (256, 256), (256, 1))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (512, 256), (256, 1))
    assert_size_stride(arg98_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (256, 256), (256, 1))
    assert_size_stride(arg102_1, (256, 256), (256, 1))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (512, 256), (256, 1))
    assert_size_stride(arg106_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (256, 256), (256, 1))
    assert_size_stride(arg110_1, (256, 256), (256, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (512, 256), (256, 1))
    assert_size_stride(arg114_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (256, 256), (256, 1))
    assert_size_stride(arg118_1, (256, 256), (256, 1))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg126_1, (256, 256), (256, 1))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (128, 256), (256, 1))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg149_1, (128, ), (1, ))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg153_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((40, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg25_1, buf0, 160, 1024, stream=stream0)
        buf1 = empty_strided_cuda((128, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg23_1, buf1, 512, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf3, arg24_1, 5242880, stream=stream0)
        buf4 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg26_1, buf4, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf6 = empty_strided_cuda((40, 32, 1, 1), (32, 1, 1280, 1280), torch.float32)
        buf7 = empty_strided_cuda((40, 32, 1, 1), (32, 1, 1280, 1280), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf5, arg27_1, buf6, buf7, 1280, 4096, stream=stream0)
        buf9 = empty_strided_cuda((40, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5.run(arg0_1, buf9, 10240, stream=stream0)
        buf10 = empty_strided_cuda((40, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.t, aten.addmm]
        extern_kernels.mm(buf9, reinterpret_tensor(arg1_1, (256, 1024), (1, 256), 0), out=buf10)
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_6.run(buf11, arg2_1, 40960, stream=stream0)
        buf14 = empty_strided_cuda((40, 512), (512, 1), torch.float32)
        buf12 = reinterpret_tensor(buf14, (40, 256), (512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg4_1, buf11, reinterpret_tensor(arg3_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf12)
        buf13 = reinterpret_tensor(buf14, (40, 256), (512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [s_emb], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_7.run(arg6_1, arg5_1, buf13, 10240, stream=stream0)
        del arg5_1
        del arg6_1
        del buf12
        del buf13
        buf15 = empty_strided_cuda((40, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf14, reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf15)
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_8.run(buf16, arg8_1, 20480, stream=stream0)
        buf17 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg10_1, buf16, reinterpret_tensor(arg9_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf17)
        buf18 = empty_strided_cuda((40, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), out=buf18)
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf19, arg12_1, 10240, stream=stream0)
        buf20 = empty_strided_cuda((40, 3200), (3200, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg14_1, buf19, reinterpret_tensor(arg13_1, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf20)
        buf21 = empty_strided_cuda((40, 32, 4, 1024), (131072, 4096, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf5, arg27_1, buf6, buf7, buf21, 5120, 1024, stream=stream0)
        buf22 = empty_strided_cuda((40, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm, split, v_1, transpose, x_flat, v_t_x], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf21, (40, 128, 1024), (131072, 1024, 1), 0), out=buf22)
        buf23 = reinterpret_tensor(buf5, (40, 128, 1024), (131072, 1024, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [split, u_1, mixed], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 128, 12), (3200, 12, 1), 0), buf22, out=buf23)
        buf24 = empty_strided_cuda((40, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf21, buf23, buf20, buf24, 5120, 1024, stream=stream0)
        del buf21
        del buf23
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf25 = extern_kernels.convolution(buf24, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2, add_2, h_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf26, arg29_1, buf3, 5242880, stream=stream0)
        buf27 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg30_1, buf27, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf26, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf29 = buf7; del buf7  # reuse
        buf30 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf28, arg31_1, buf29, buf30, 1280, 4096, stream=stream0)
        buf32 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg37_1, (256, 512), (1, 256), 0), out=buf32)
        buf33 = reinterpret_tensor(buf3, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf28, arg31_1, buf29, buf30, buf33, 5120, 1024, stream=stream0)
        buf34 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, v_3, transpose_1, x_flat_1, v_t_x_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf33, (40, 128, 1024), (131072, 1024, 1), 0), out=buf34)
        buf35 = reinterpret_tensor(buf28, (40, 128, 1024), (131072, 1024, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [split_1, u_3, mixed_1], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 128, 12), (3200, 12, 1), 0), buf34, out=buf35)
        buf36 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf33, buf35, buf20, buf36, 5120, 1024, stream=stream0)
        del buf33
        del buf35
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4, x_5], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf37 = extern_kernels.convolution(buf36, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf36
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4, x_5, add_5, h_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf38, arg33_1, buf26, 5242880, stream=stream0)
        buf39 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_13.run(arg34_1, buf39, 32768, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf41 = empty_strided_cuda((40, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_transpose_view_14.run(buf40, arg35_1, buf41, 2621440, stream=stream0)
        buf42 = empty_strided_cuda((10240, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (10240, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 256), (1, 256), 0), out=buf42)
        buf43 = reinterpret_tensor(buf41, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf42, buf43, 2621440, stream=stream0)
        buf44 = reinterpret_tensor(buf19, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf32, buf44, 10240, stream=stream0)
        buf45 = reinterpret_tensor(buf11, (160, 256, 1), (256, 1, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, q_1, view_11, q_2, matmul, view_12, k_1, transpose_6], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf44, (160, 64, 1), (64, 1, 0), 0), out=buf45)
        buf46 = reinterpret_tensor(buf45, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf46, 40960, stream=stream0)
        buf47 = reinterpret_tensor(buf44, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf32, buf47, 10240, stream=stream0)
        buf48 = reinterpret_tensor(buf43, (160, 256, 64), (16384, 64, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, matmul, attn_1, matmul_1, view_13, v_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf47, (160, 1, 64), (64, 0, 1), 0), out=buf48)
        buf49 = reinterpret_tensor(buf42, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf48, buf49, 2621440, stream=stream0)
        buf50 = reinterpret_tensor(buf48, (10240, 256), (256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4, input_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (10240, 256), (256, 1), 0), reinterpret_tensor(arg38_1, (256, 256), (1, 256), 0), out=buf50)
        buf51 = reinterpret_tensor(buf50, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_20.run(buf51, buf40, arg35_1, arg39_1, 2621440, stream=stream0)
        buf52 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg40_1, buf52, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf53 = extern_kernels.convolution(buf51, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf54 = buf30; del buf30  # reuse
        buf55 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf53, arg41_1, buf54, buf55, 1280, 2048, stream=stream0)
        buf57 = reinterpret_tensor(buf47, (40, 256), (256, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg15_1, (256, 256), (1, 256), 0), out=buf57)
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf58, arg16_1, 10240, stream=stream0)
        buf59 = empty_strided_cuda((40, 6400), (6400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg18_1, buf58, reinterpret_tensor(arg17_1, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf59)
        buf60 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg45_1, (256, 512), (1, 256), 0), out=buf60)
        buf61 = reinterpret_tensor(buf51, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf53, arg41_1, buf54, buf55, buf61, 10240, 256, stream=stream0)
        buf62 = empty_strided_cuda((40, 12, 256), (3072, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, v_7, transpose_9, x_flat_2, v_t_x_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf61, (40, 256, 256), (65536, 256, 1), 0), out=buf62)
        buf63 = reinterpret_tensor(buf53, (40, 256, 256), (65536, 256, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [split_2, u_5, mixed_2], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 256, 12), (6400, 12, 1), 0), buf62, out=buf63)
        buf64 = reinterpret_tensor(buf49, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf61, buf63, buf59, buf64, 10240, 256, stream=stream0)
        del buf61
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf65 = extern_kernels.convolution(buf64, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf66 = reinterpret_tensor(buf64, (40, 256, 256), (65536, 256, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf65, arg43_1, buf40, arg35_1, buf66, 2621440, stream=stream0)
        buf67 = reinterpret_tensor(buf63, (10240, 256), (256, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (10240, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 256), (1, 256), 0), out=buf67)
        buf68 = reinterpret_tensor(buf66, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [q_4, view_21, q_5, matmul_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf67, buf68, 2621440, stream=stream0)
        buf69 = reinterpret_tensor(buf58, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_22, k_3, transpose_14, matmul_2], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf60, buf69, 10240, stream=stream0)
        buf70 = reinterpret_tensor(buf46, (160, 256, 1), (256, 1, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, q_4, view_21, q_5, matmul_2, view_22, k_3, transpose_14], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf69, (160, 64, 1), (64, 1, 0), 0), out=buf70)
        buf71 = reinterpret_tensor(buf70, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [matmul_2, attn_3], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf71, 40960, stream=stream0)
        buf72 = reinterpret_tensor(buf69, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_23, v_9, matmul_3], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf60, buf72, 10240, stream=stream0)
        buf73 = reinterpret_tensor(buf68, (160, 256, 64), (16384, 64, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, matmul_2, attn_3, matmul_3, view_23, v_9], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf71, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf72, (160, 1, 64), (64, 0, 1), 0), out=buf73)
        buf74 = reinterpret_tensor(buf67, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf73, buf74, 2621440, stream=stream0)
        buf75 = reinterpret_tensor(buf73, (10240, 256), (256, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8, input_18], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf74, (10240, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 256), (1, 256), 0), out=buf75)
        buf76 = reinterpret_tensor(buf75, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26.run(buf76, buf65, arg43_1, buf40, arg35_1, arg47_1, 2621440, stream=stream0)
        buf77 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg48_1, buf77, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf79 = buf55; del buf55  # reuse
        buf80 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf78, arg49_1, buf79, buf80, 1280, 2048, stream=stream0)
        buf82 = reinterpret_tensor(buf71, (40, 1024), (1024, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg55_1, (256, 1024), (1, 256), 0), out=buf82)
        buf83 = reinterpret_tensor(buf76, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf78, arg49_1, buf79, buf80, buf83, 10240, 256, stream=stream0)
        buf84 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, v_11, transpose_17, x_flat_3, v_t_x_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf83, (40, 256, 256), (65536, 256, 1), 0), out=buf84)
        buf85 = reinterpret_tensor(buf78, (40, 256, 256), (65536, 256, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [split_3, u_7, mixed_3], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 256, 12), (6400, 12, 1), 0), buf84, out=buf85)
        buf86 = reinterpret_tensor(buf74, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf83, buf85, buf59, buf86, 10240, 256, stream=stream0)
        del buf83
        del buf85
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf87 = extern_kernels.convolution(buf86, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf86
        buf88 = buf87; del buf87  # reuse
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13, add_13, h_5], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27.run(buf89, arg51_1, buf65, arg43_1, buf40, arg35_1, 2621440, stream=stream0)
        del buf40
        buf90 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_28.run(arg52_1, buf90, 131072, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
        buf91 = extern_kernels.convolution(buf89, buf90, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf92 = empty_strided_cuda((40, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_silu_transpose_view_29.run(buf91, arg53_1, buf92, 1310720, stream=stream0)
        buf93 = empty_strided_cuda((2560, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (2560, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf93)
        buf94 = reinterpret_tensor(buf92, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf93, buf94, 1310720, stream=stream0)
        buf95 = reinterpret_tensor(buf60, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf82, buf95, 20480, stream=stream0)
        buf96 = reinterpret_tensor(buf14, (320, 64, 1), (64, 1, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, q_7, view_31, q_8, matmul_4, view_32, k_5, transpose_22], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf95, (320, 64, 1), (64, 1, 0), 0), out=buf96)
        buf97 = reinterpret_tensor(buf96, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_32.run(buf97, 20480, stream=stream0)
        buf98 = reinterpret_tensor(buf95, (40, 8, 1, 64), (512, 64, 64, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf82, buf98, 20480, stream=stream0)
        buf99 = reinterpret_tensor(buf94, (320, 64, 64), (4096, 64, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, matmul_4, attn_5, matmul_5, view_33, v_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf97, (320, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf98, (320, 1, 64), (64, 0, 1), 0), out=buf99)
        buf100 = reinterpret_tensor(buf93, (40, 64, 8, 64), (32768, 512, 64, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf99, buf100, 1310720, stream=stream0)
        buf101 = reinterpret_tensor(buf99, (2560, 512), (512, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12, input_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf100, (2560, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 512), (1, 512), 0), out=buf101)
        buf102 = reinterpret_tensor(buf101, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_35.run(buf102, buf91, arg53_1, arg57_1, 1310720, stream=stream0)
        buf103 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(arg58_1, buf103, 262144, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf104 = extern_kernels.convolution(buf102, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf105 = buf80; del buf80  # reuse
        buf106 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37.run(buf104, arg59_1, buf105, buf106, 1280, 1024, stream=stream0)
        buf108 = reinterpret_tensor(buf72, (40, 256), (256, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg19_1, (256, 256), (1, 256), 0), out=buf108)
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf109, arg20_1, 10240, stream=stream0)
        buf110 = empty_strided_cuda((40, 12800), (12800, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg22_1, buf109, reinterpret_tensor(arg21_1, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf110)
        buf111 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg63_1, (256, 1024), (1, 256), 0), out=buf111)
        buf112 = reinterpret_tensor(buf102, (40, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf104, arg59_1, buf105, buf106, buf112, 20480, 64, stream=stream0)
        buf113 = empty_strided_cuda((40, 12, 64), (768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, v_15, transpose_25, x_flat_4, v_t_x_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (40, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf112, (40, 512, 64), (32768, 64, 1), 0), out=buf113)
        buf114 = reinterpret_tensor(buf104, (40, 512, 64), (32768, 64, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [split_4, u_9, mixed_4], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (40, 512, 12), (12800, 12, 1), 0), buf113, out=buf114)
        buf115 = reinterpret_tensor(buf100, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39.run(buf112, buf114, buf110, buf115, 20480, 64, stream=stream0)
        del buf112
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf116 = extern_kernels.convolution(buf115, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf117 = reinterpret_tensor(buf115, (40, 64, 512), (32768, 512, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40.run(buf116, arg61_1, buf91, arg53_1, buf117, 1310720, stream=stream0)
        buf118 = reinterpret_tensor(buf114, (2560, 512), (512, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (2560, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 512), (1, 512), 0), out=buf118)
        buf119 = reinterpret_tensor(buf117, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [q_10, view_41, q_11, matmul_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf118, buf119, 1310720, stream=stream0)
        buf120 = reinterpret_tensor(buf98, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_42, k_7, transpose_30, matmul_6], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf111, buf120, 20480, stream=stream0)
        buf121 = reinterpret_tensor(buf97, (320, 64, 1), (64, 1, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, q_10, view_41, q_11, matmul_6, view_42, k_7, transpose_30], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf120, (320, 64, 1), (64, 1, 0), 0), out=buf121)
        buf122 = reinterpret_tensor(buf121, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [matmul_6, attn_7], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_32.run(buf122, 20480, stream=stream0)
        buf123 = reinterpret_tensor(buf120, (40, 8, 1, 64), (512, 64, 64, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_43, v_17, matmul_7], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf111, buf123, 20480, stream=stream0)
        buf124 = reinterpret_tensor(buf119, (320, 64, 64), (4096, 64, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, matmul_6, attn_7, matmul_7, view_43, v_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (320, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf123, (320, 1, 64), (64, 0, 1), 0), out=buf124)
        buf125 = reinterpret_tensor(buf118, (40, 64, 8, 64), (32768, 512, 64, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf124, buf125, 1310720, stream=stream0)
        buf126 = reinterpret_tensor(buf124, (2560, 512), (512, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16, input_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf125, (2560, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf126)
        buf127 = reinterpret_tensor(buf126, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41.run(buf127, buf116, arg61_1, buf91, arg53_1, arg65_1, 1310720, stream=stream0)
        buf128 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(arg66_1, buf128, 262144, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf129 = extern_kernels.convolution(buf127, buf128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf128
        buf130 = buf106; del buf106  # reuse
        buf131 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37.run(buf129, arg67_1, buf130, buf131, 1280, 1024, stream=stream0)
        buf133 = reinterpret_tensor(buf127, (40, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf129, arg67_1, buf130, buf131, buf133, 20480, 64, stream=stream0)
        buf134 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, v_19, transpose_33, x_flat_5, v_t_x_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (40, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf133, (40, 512, 64), (32768, 64, 1), 0), out=buf134)
        buf135 = reinterpret_tensor(buf129, (40, 512, 64), (32768, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [split_5, u_11, mixed_5], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (40, 512, 12), (12800, 12, 1), 0), buf134, out=buf135)
        buf136 = reinterpret_tensor(buf125, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39.run(buf133, buf135, buf110, buf136, 20480, 64, stream=stream0)
        del buf133
        del buf135
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf137 = extern_kernels.convolution(buf136, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf138 = buf137; del buf137  # reuse
        buf142 = reinterpret_tensor(buf136, (40, 64, 512), (32768, 512, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42.run(buf138, arg69_1, buf116, arg61_1, buf91, arg53_1, arg70_1, arg71_1, buf142, 2560, 512, stream=stream0)
        buf143 = empty_strided_cuda((2560, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf142, (2560, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 1536), (1, 512), 0), out=buf143)
        buf144 = reinterpret_tensor(buf142, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_43.run(buf143, arg73_1, buf144, 1310720, stream=stream0)
        buf145 = reinterpret_tensor(buf91, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_transpose_view_44.run(buf143, arg73_1, buf145, 20480, 64, stream=stream0)
        buf146 = reinterpret_tensor(buf116, (320, 64, 64), (4096, 64, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8, k_8, transpose_35], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone, aten._unsafe_view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf145, (320, 64, 64), (4096, 64, 1), 0), out=buf146)
        buf149 = reinterpret_tensor(buf146, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_amax_mul_sub_view_45.run(buf149, 20480, 64, stream=stream0)
        buf150 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_46.run(buf143, arg73_1, buf150, 1310720, stream=stream0)
        del buf143
        buf151 = reinterpret_tensor(buf144, (320, 64, 64), (4096, 64, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, matmul_8, attn_9, out_20, v_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.mul, aten.sub, aten._softmax, aten.select, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf150, (320, 64, 64), (4096, 64, 1), 0), out=buf151)
        del buf149
        buf152 = reinterpret_tensor(buf150, (40, 64, 8, 64), (32768, 512, 64, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf151, buf152, 1310720, stream=stream0)
        buf153 = reinterpret_tensor(buf151, (2560, 512), (512, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21, out_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf152, (2560, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf153)
        del buf152
        buf158 = reinterpret_tensor(buf138, (40, 64, 512), (32768, 512, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47.run(buf158, buf153, arg75_1, arg76_1, arg77_1, 2560, 512, stream=stream0)
        del buf153
        buf157 = reinterpret_tensor(buf123, (40, 512), (512, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg85_1, (256, 512), (1, 256), 0), out=buf157)
        buf159 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_48.run(arg78_1, buf159, 131072, 16, stream=stream0)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        buf160 = extern_kernels.convolution(reinterpret_tensor(buf158, (40, 512, 8, 8), (32768, 1, 4096, 512), 0), buf159, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf159
        buf161 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg80_1, (256, 256), (1, 256), 0), out=buf161)
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf162, arg81_1, 10240, stream=stream0)
        buf163 = empty_strided_cuda((40, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf162, reinterpret_tensor(arg82_1, (256, 256), (1, 256), 0), out=buf163)
        buf164 = reinterpret_tensor(buf65, (40, 256, 16, 16), (65536, 256, 16, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_49.run(buf160, arg79_1, buf89, buf163, arg83_1, buf164, 10240, 256, stream=stream0)
        buf165 = reinterpret_tensor(buf89, (10240, 256), (1, 10240), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_t_transpose_view_50.run(buf164, buf165, 2621440, stream=stream0)
        buf166 = reinterpret_tensor(buf160, (10240, 256), (256, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        extern_kernels.mm(buf165, reinterpret_tensor(arg84_1, (256, 256), (1, 256), 0), out=buf166)
        buf167 = reinterpret_tensor(buf165, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [q_14, view_53, q_15, matmul_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf166, buf167, 2621440, stream=stream0)
        buf168 = reinterpret_tensor(buf163, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_54, k_10, transpose_42, matmul_10], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf157, buf168, 10240, stream=stream0)
        buf169 = reinterpret_tensor(buf111, (160, 256, 1), (256, 1, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, q_14, view_53, q_15, matmul_10, view_54, k_10, transpose_42], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf168, (160, 64, 1), (64, 1, 0), 0), out=buf169)
        buf170 = reinterpret_tensor(buf169, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [matmul_10, attn_11], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf170, 40960, stream=stream0)
        buf171 = reinterpret_tensor(buf168, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_55, v_22, matmul_11], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf157, buf171, 10240, stream=stream0)
        buf172 = reinterpret_tensor(buf167, (160, 256, 64), (16384, 64, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, matmul_10, attn_11, matmul_11, view_55, v_22], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf170, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf171, (160, 1, 64), (64, 0, 1), 0), out=buf172)
        buf173 = reinterpret_tensor(buf166, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf172, buf173, 2621440, stream=stream0)
        buf174 = reinterpret_tensor(buf172, (10240, 256), (256, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26, input_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf173, (10240, 256), (256, 1), 0), reinterpret_tensor(arg86_1, (256, 256), (1, 256), 0), out=buf174)
        buf175 = reinterpret_tensor(buf174, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_transpose_view_51.run(buf175, buf164, arg87_1, 10240, 256, stream=stream0)
        buf176 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg88_1, buf176, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf177 = extern_kernels.convolution(buf175, buf176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf178 = buf131; del buf131  # reuse
        buf179 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf177, arg89_1, buf178, buf179, 1280, 2048, stream=stream0)
        buf181 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg93_1, (256, 512), (1, 256), 0), out=buf181)
        buf182 = reinterpret_tensor(buf175, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf177, arg89_1, buf178, buf179, buf182, 10240, 256, stream=stream0)
        buf183 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, v_24, transpose_45, x_flat_7, v_t_x_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf182, (40, 256, 256), (65536, 256, 1), 0), out=buf183)
        buf184 = reinterpret_tensor(buf177, (40, 256, 256), (65536, 256, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [split_6, u_13, mixed_6], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 256, 12), (6400, 12, 1), 0), buf183, out=buf184)
        buf185 = reinterpret_tensor(buf173, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf182, buf184, buf59, buf185, 10240, 256, stream=stream0)
        del buf182
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf186 = extern_kernels.convolution(buf185, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf187 = reinterpret_tensor(buf185, (40, 256, 256), (65536, 256, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_52.run(buf186, arg91_1, buf164, buf187, 10240, 256, stream=stream0)
        buf188 = reinterpret_tensor(buf184, (10240, 256), (256, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (10240, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf188)
        buf189 = reinterpret_tensor(buf187, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [q_17, view_63, q_18, matmul_12], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf188, buf189, 2621440, stream=stream0)
        buf190 = reinterpret_tensor(buf171, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_64, k_12, transpose_50, matmul_12], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf181, buf190, 10240, stream=stream0)
        buf191 = reinterpret_tensor(buf170, (160, 256, 1), (256, 1, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, q_17, view_63, q_18, matmul_12, view_64, k_12, transpose_50], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf190, (160, 64, 1), (64, 1, 0), 0), out=buf191)
        buf192 = reinterpret_tensor(buf191, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [matmul_12, attn_13], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf192, 40960, stream=stream0)
        buf193 = reinterpret_tensor(buf190, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_65, v_26, matmul_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf181, buf193, 10240, stream=stream0)
        buf194 = reinterpret_tensor(buf189, (160, 256, 64), (16384, 64, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, matmul_12, attn_13, matmul_13, view_65, v_26], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf192, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf193, (160, 1, 64), (64, 0, 1), 0), out=buf194)
        buf195 = reinterpret_tensor(buf188, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf194, buf195, 2621440, stream=stream0)
        buf196 = reinterpret_tensor(buf194, (10240, 256), (256, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30, input_30], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf195, (10240, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 256), (1, 256), 0), out=buf196)
        buf197 = reinterpret_tensor(buf196, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53.run(buf197, buf186, arg91_1, buf164, arg95_1, 10240, 256, stream=stream0)
        buf198 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg96_1, buf198, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf199 = extern_kernels.convolution(buf197, buf198, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf200 = buf179; del buf179  # reuse
        buf201 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf199, arg97_1, buf200, buf201, 1280, 2048, stream=stream0)
        buf203 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg101_1, (256, 512), (1, 256), 0), out=buf203)
        buf204 = reinterpret_tensor(buf197, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf199, arg97_1, buf200, buf201, buf204, 10240, 256, stream=stream0)
        buf205 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, v_28, transpose_53, x_flat_8, v_t_x_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf204, (40, 256, 256), (65536, 256, 1), 0), out=buf205)
        buf206 = reinterpret_tensor(buf199, (40, 256, 256), (65536, 256, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [split_7, u_15, mixed_7], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 256, 12), (6400, 12, 1), 0), buf205, out=buf206)
        buf207 = reinterpret_tensor(buf195, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf204, buf206, buf59, buf207, 10240, 256, stream=stream0)
        del buf204
        del buf206
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf208 = extern_kernels.convolution(buf207, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf209 = buf208; del buf208  # reuse
        buf210 = reinterpret_tensor(buf207, (40, 256, 256), (65536, 256, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54.run(buf209, arg99_1, buf186, arg91_1, buf164, buf210, 2621440, stream=stream0)
        del buf164
        buf211 = reinterpret_tensor(buf186, (10240, 256), (256, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [h_12, view_72, q_19, q_20], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (10240, 256), (256, 1), 0), reinterpret_tensor(arg100_1, (256, 256), (1, 256), 0), out=buf211)
        buf212 = reinterpret_tensor(buf210, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [q_20, view_73, q_21, matmul_14], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf211, buf212, 2621440, stream=stream0)
        buf213 = reinterpret_tensor(buf193, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_74, k_14, transpose_58, matmul_14], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf203, buf213, 10240, stream=stream0)
        buf214 = reinterpret_tensor(buf192, (160, 256, 1), (256, 1, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, q_20, view_73, q_21, matmul_14, view_74, k_14, transpose_58], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf212, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf213, (160, 64, 1), (64, 1, 0), 0), out=buf214)
        buf215 = reinterpret_tensor(buf214, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [matmul_14, attn_15], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf215, 40960, stream=stream0)
        buf216 = reinterpret_tensor(buf213, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_75, v_30, matmul_15], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf203, buf216, 10240, stream=stream0)
        buf217 = reinterpret_tensor(buf212, (160, 256, 64), (16384, 64, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, matmul_14, attn_15, matmul_15, view_75, v_30], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf215, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf216, (160, 1, 64), (64, 0, 1), 0), out=buf217)
        buf218 = reinterpret_tensor(buf211, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf217, buf218, 2621440, stream=stream0)
        buf219 = reinterpret_tensor(buf217, (10240, 256), (256, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34, input_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf218, (10240, 256), (256, 1), 0), reinterpret_tensor(arg102_1, (256, 256), (1, 256), 0), out=buf219)
        buf220 = reinterpret_tensor(buf219, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_55.run(buf220, buf209, arg103_1, 2621440, stream=stream0)
        buf221 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg104_1, buf221, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf222 = extern_kernels.convolution(buf220, buf221, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf223 = buf201; del buf201  # reuse
        buf224 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf222, arg105_1, buf223, buf224, 1280, 2048, stream=stream0)
        buf226 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg109_1, (256, 512), (1, 256), 0), out=buf226)
        buf227 = reinterpret_tensor(buf220, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf222, arg105_1, buf223, buf224, buf227, 10240, 256, stream=stream0)
        buf228 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, v_32, transpose_61, x_flat_9, v_t_x_8], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf227, (40, 256, 256), (65536, 256, 1), 0), out=buf228)
        buf229 = reinterpret_tensor(buf222, (40, 256, 256), (65536, 256, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [split_8, u_17, mixed_8], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 256, 12), (6400, 12, 1), 0), buf228, out=buf229)
        buf230 = reinterpret_tensor(buf218, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf227, buf229, buf59, buf230, 10240, 256, stream=stream0)
        del buf227
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf231 = extern_kernels.convolution(buf230, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf232 = reinterpret_tensor(buf230, (40, 256, 256), (65536, 256, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_56.run(buf231, arg107_1, buf209, buf232, 2621440, stream=stream0)
        buf233 = reinterpret_tensor(buf229, (10240, 256), (256, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (10240, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), out=buf233)
        buf234 = reinterpret_tensor(buf232, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [q_23, view_83, q_24, matmul_16], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf233, buf234, 2621440, stream=stream0)
        buf235 = reinterpret_tensor(buf216, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_84, k_16, transpose_66, matmul_16], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf226, buf235, 10240, stream=stream0)
        buf236 = reinterpret_tensor(buf215, (160, 256, 1), (256, 1, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, q_23, view_83, q_24, matmul_16, view_84, k_16, transpose_66], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf234, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf235, (160, 64, 1), (64, 1, 0), 0), out=buf236)
        buf237 = reinterpret_tensor(buf236, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [matmul_16, attn_17], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf237, 40960, stream=stream0)
        buf238 = reinterpret_tensor(buf235, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_85, v_34, matmul_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf226, buf238, 10240, stream=stream0)
        buf239 = reinterpret_tensor(buf234, (160, 256, 64), (16384, 64, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, matmul_16, attn_17, matmul_17, view_85, v_34], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf238, (160, 1, 64), (64, 0, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf233, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf239, buf240, 2621440, stream=stream0)
        buf241 = reinterpret_tensor(buf239, (10240, 256), (256, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38, input_34], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf240, (10240, 256), (256, 1), 0), reinterpret_tensor(arg110_1, (256, 256), (1, 256), 0), out=buf241)
        buf242 = reinterpret_tensor(buf241, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57.run(buf242, buf231, arg107_1, buf209, arg111_1, 2621440, stream=stream0)
        buf243 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg112_1, buf243, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf244 = extern_kernels.convolution(buf242, buf243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf245 = buf224; del buf224  # reuse
        buf246 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf244, arg113_1, buf245, buf246, 1280, 2048, stream=stream0)
        buf248 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg117_1, (256, 512), (1, 256), 0), out=buf248)
        buf249 = reinterpret_tensor(buf242, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf244, arg113_1, buf245, buf246, buf249, 10240, 256, stream=stream0)
        buf250 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, v_36, transpose_69, x_flat_10, v_t_x_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf249, (40, 256, 256), (65536, 256, 1), 0), out=buf250)
        buf251 = reinterpret_tensor(buf244, (40, 256, 256), (65536, 256, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [split_9, u_19, mixed_9], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 256, 12), (6400, 12, 1), 0), buf250, out=buf251)
        buf252 = reinterpret_tensor(buf240, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf249, buf251, buf59, buf252, 10240, 256, stream=stream0)
        del buf249
        del buf251
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf253 = extern_kernels.convolution(buf252, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf254 = buf253; del buf253  # reuse
        buf255 = reinterpret_tensor(buf252, (40, 256, 256), (65536, 256, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58.run(buf254, arg115_1, buf231, arg107_1, buf209, buf255, 2621440, stream=stream0)
        del buf209
        buf256 = reinterpret_tensor(buf231, (10240, 256), (256, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (10240, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 256), (1, 256), 0), out=buf256)
        buf257 = reinterpret_tensor(buf255, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [q_26, view_93, q_27, matmul_18], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf256, buf257, 2621440, stream=stream0)
        buf258 = reinterpret_tensor(buf238, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_94, k_18, transpose_74, matmul_18], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf248, buf258, 10240, stream=stream0)
        buf259 = reinterpret_tensor(buf237, (160, 256, 1), (256, 1, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, q_26, view_93, q_27, matmul_18, view_94, k_18, transpose_74], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf258, (160, 64, 1), (64, 1, 0), 0), out=buf259)
        buf260 = reinterpret_tensor(buf259, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [matmul_18, attn_19], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf260, 40960, stream=stream0)
        buf261 = reinterpret_tensor(buf258, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_95, v_38, matmul_19], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf248, buf261, 10240, stream=stream0)
        buf262 = reinterpret_tensor(buf257, (160, 256, 64), (16384, 64, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, matmul_18, attn_19, matmul_19, view_95, v_38], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf261, (160, 1, 64), (64, 0, 1), 0), out=buf262)
        buf263 = reinterpret_tensor(buf256, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf262, buf263, 2621440, stream=stream0)
        buf264 = reinterpret_tensor(buf262, (10240, 256), (256, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42, input_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf263, (10240, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 256), (1, 256), 0), out=buf264)
        buf265 = reinterpret_tensor(buf264, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_55.run(buf265, buf254, arg119_1, 2621440, stream=stream0)
        buf266 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg120_1, buf266, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf267 = extern_kernels.convolution(buf265, buf266, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf268 = buf246; del buf246  # reuse
        buf269 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf267, arg121_1, buf268, buf269, 1280, 2048, stream=stream0)
        buf271 = reinterpret_tensor(buf265, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf267, arg121_1, buf268, buf269, buf271, 10240, 256, stream=stream0)
        buf272 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, v_40, transpose_77, x_flat_11, v_t_x_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf271, (40, 256, 256), (65536, 256, 1), 0), out=buf272)
        buf273 = reinterpret_tensor(buf267, (40, 256, 256), (65536, 256, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [split_10, u_21, mixed_10], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (40, 256, 12), (6400, 12, 1), 0), buf272, out=buf273)
        buf274 = reinterpret_tensor(buf263, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf271, buf273, buf59, buf274, 10240, 256, stream=stream0)
        del buf271
        del buf273
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf275 = extern_kernels.convolution(buf274, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf274
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59.run(buf276, arg123_1, buf254, 2621440, stream=stream0)
        buf277 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60.run(arg124_1, buf277, 32768, 16, stream=stream0)
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf278 = extern_kernels.convolution(buf276, buf277, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf277
        buf279 = reinterpret_tensor(buf261, (40, 256), (256, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg126_1, (256, 256), (1, 256), 0), out=buf279)
        buf280 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf280, arg127_1, 10240, stream=stream0)
        buf281 = empty_strided_cuda((40, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf280, reinterpret_tensor(arg128_1, (256, 128), (1, 256), 0), out=buf281)
        buf282 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_61.run(buf282, arg125_1, buf38, buf281, arg129_1, 5242880, stream=stream0)
        buf283 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg130_1, buf283, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf282, buf283, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf285 = buf269; del buf269  # reuse
        buf286 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf284, arg131_1, buf285, buf286, 1280, 4096, stream=stream0)
        buf288 = reinterpret_tensor(buf38, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf284, arg131_1, buf285, buf286, buf288, 5120, 1024, stream=stream0)
        buf289 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, v_42, transpose_78, x_flat_12, v_t_x_11], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf288, (40, 128, 1024), (131072, 1024, 1), 0), out=buf289)
        buf290 = reinterpret_tensor(buf284, (40, 128, 1024), (131072, 1024, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [split_11, u_23, mixed_11], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 128, 12), (3200, 12, 1), 0), buf289, out=buf290)
        buf291 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf288, buf290, buf20, buf291, 5120, 1024, stream=stream0)
        del buf288
        del buf290
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43, x_44], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf292 = extern_kernels.convolution(buf291, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf293 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43, x_44, add_47, h_18], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf293, arg133_1, buf282, 5242880, stream=stream0)
        buf294 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg134_1, buf294, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf293, buf294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf296 = buf286; del buf286  # reuse
        buf297 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf295, arg135_1, buf296, buf297, 1280, 4096, stream=stream0)
        buf299 = reinterpret_tensor(buf282, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf295, arg135_1, buf296, buf297, buf299, 5120, 1024, stream=stream0)
        buf300 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, v_44, transpose_79, x_flat_13, v_t_x_12], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf299, (40, 128, 1024), (131072, 1024, 1), 0), out=buf300)
        buf301 = reinterpret_tensor(buf295, (40, 128, 1024), (131072, 1024, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [split_12, u_25, mixed_12], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 128, 12), (3200, 12, 1), 0), buf300, out=buf301)
        buf302 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf299, buf301, buf20, buf302, 5120, 1024, stream=stream0)
        del buf299
        del buf301
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46, x_47], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf303 = extern_kernels.convolution(buf302, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46, x_47, add_50, h_19], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf304, arg137_1, buf293, 5242880, stream=stream0)
        buf305 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg138_1, buf305, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf304, buf305, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf307 = buf297; del buf297  # reuse
        buf308 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf306, arg139_1, buf307, buf308, 1280, 4096, stream=stream0)
        buf310 = reinterpret_tensor(buf293, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf306, arg139_1, buf307, buf308, buf310, 5120, 1024, stream=stream0)
        buf311 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, v_46, transpose_80, x_flat_14, v_t_x_13], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf310, (40, 128, 1024), (131072, 1024, 1), 0), out=buf311)
        buf312 = reinterpret_tensor(buf306, (40, 128, 1024), (131072, 1024, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [split_13, u_27, mixed_13], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 128, 12), (3200, 12, 1), 0), buf311, out=buf312)
        buf313 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf310, buf312, buf20, buf313, 5120, 1024, stream=stream0)
        del buf310
        del buf312
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49, x_50], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf314 = extern_kernels.convolution(buf313, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf315 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49, x_50, add_53, h_20], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf315, arg141_1, buf304, 5242880, stream=stream0)
        buf316 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg142_1, buf316, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf315, buf316, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf318 = buf308; del buf308  # reuse
        buf319 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf317, arg143_1, buf318, buf319, 1280, 4096, stream=stream0)
        buf321 = reinterpret_tensor(buf304, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf317, arg143_1, buf318, buf319, buf321, 5120, 1024, stream=stream0)
        buf322 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, v_48, transpose_81, x_flat_15, v_t_x_14], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf321, (40, 128, 1024), (131072, 1024, 1), 0), out=buf322)
        buf323 = reinterpret_tensor(buf317, (40, 128, 1024), (131072, 1024, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [split_14, u_29, mixed_14], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 128, 12), (3200, 12, 1), 0), buf322, out=buf323)
        buf324 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf321, buf323, buf20, buf324, 5120, 1024, stream=stream0)
        del buf321
        del buf323
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52, x_53], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf325 = extern_kernels.convolution(buf324, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf326 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52, x_53, add_56, h_21], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf326, arg145_1, buf315, 5242880, stream=stream0)
        buf327 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg146_1, buf327, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf326, buf327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf329 = buf319; del buf319  # reuse
        buf330 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf328, arg147_1, buf329, buf330, 1280, 4096, stream=stream0)
        buf332 = reinterpret_tensor(buf315, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf328, arg147_1, buf329, buf330, buf332, 5120, 1024, stream=stream0)
        buf333 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, v_50, transpose_82, x_flat_16, v_t_x_15], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf332, (40, 128, 1024), (131072, 1024, 1), 0), out=buf333)
        buf334 = reinterpret_tensor(buf328, (40, 128, 1024), (131072, 1024, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [split_15, u_31, mixed_15], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (40, 128, 12), (3200, 12, 1), 0), buf333, out=buf334)
        buf335 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf332, buf334, buf20, buf335, 5120, 1024, stream=stream0)
        del buf332
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf336 = extern_kernels.convolution(buf335, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf337 = buf330; del buf330  # reuse
        buf338 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_62.run(buf336, arg149_1, buf326, buf337, buf338, 1280, 4096, stream=stream0)
        buf340 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg25_1, buf340, 160, 1024, stream=stream0)
        buf341 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg23_1, buf341, 512, 9, stream=stream0)
        del arg23_1
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf340, buf341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf340
        del buf341
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf343, arg24_1, 5242880, stream=stream0)
        del arg24_1
        buf344 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg26_1, buf344, 16384, 9, stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf343, buf344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf346 = empty_strided_cuda((40, 32, 1, 1), (32, 1, 1280, 1280), torch.float32)
        buf347 = empty_strided_cuda((40, 32, 1, 1), (32, 1, 1280, 1280), torch.float32)
        # Topologically Sorted Source Nodes: [x_57, x_norm_17], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf345, arg27_1, buf346, buf347, 1280, 4096, stream=stream0)
        buf349 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [getitem_71, arange_1, mul_14, truediv_1, freqs_1, getitem_72, args_1, cos_1, sin_1, emb_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5.run(arg0_1, buf349, 10240, stream=stream0)
        buf350 = reinterpret_tensor(buf260, (40, 1024), (1024, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [getitem_71, arange_1, mul_14, truediv_1, freqs_1, getitem_72, args_1, cos_1, sin_1, emb_1, input_45], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.t, aten.addmm]
        extern_kernels.mm(buf349, reinterpret_tensor(arg1_1, (256, 1024), (1, 256), 0), out=buf350)
        del arg1_1
        buf351 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_6.run(buf351, arg2_1, 40960, stream=stream0)
        del arg2_1
        buf354 = buf248; del buf248  # reuse
        buf352 = reinterpret_tensor(buf354, (40, 256), (512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_45, input_46, input_47], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg4_1, buf351, reinterpret_tensor(arg3_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf352)
        del arg3_1
        del arg4_1
        buf353 = reinterpret_tensor(buf354, (40, 256), (512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [unsqueeze_13, s_emb_1, cat_3], Original ATen: [aten.unsqueeze, aten.expand, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_expand_unsqueeze_63.run(arg154_1, buf353, 10240, stream=stream0)
        del arg154_1
        del buf352
        del buf353
        buf355 = reinterpret_tensor(buf122, (40, 512), (512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf354, reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf355)
        del arg7_1
        buf356 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_8.run(buf356, arg8_1, 20480, stream=stream0)
        del arg8_1
        buf357 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49, input_50], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg10_1, buf356, reinterpret_tensor(arg9_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf357)
        del arg10_1
        del arg9_1
        buf358 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), out=buf358)
        del arg11_1
        buf359 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf359, arg12_1, 10240, stream=stream0)
        del arg12_1
        buf360 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52, input_53], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg14_1, buf359, reinterpret_tensor(arg13_1, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf360)
        del arg13_1
        del arg14_1
        buf361 = reinterpret_tensor(buf335, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf345, arg27_1, buf346, buf347, buf361, 5120, 1024, stream=stream0)
        del arg27_1
        buf362 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, v_52, transpose_83, x_flat_17, v_t_x_16], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf361, (40, 128, 1024), (131072, 1024, 1), 0), out=buf362)
        buf363 = reinterpret_tensor(buf345, (40, 128, 1024), (131072, 1024, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [split_16, u_33, mixed_16], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 128, 12), (3200, 12, 1), 0), buf362, out=buf363)
        buf364 = reinterpret_tensor(buf334, (40, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, x_flat_17, out_56, view_131, shift_33, out_57, x_58], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf361, buf363, buf360, buf364, 5120, 1024, stream=stream0)
        del buf361
        del buf363
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, x_flat_17, out_56, view_131, shift_33, out_57, x_58, x_59], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf365 = extern_kernels.convolution(buf364, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg28_1
        buf366 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, x_flat_17, out_56, view_131, shift_33, out_57, x_58, x_59, add_62, h_24], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf366, arg29_1, buf343, 5242880, stream=stream0)
        del arg29_1
        buf367 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg30_1, buf367, 16384, 9, stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf366, buf367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf369 = buf347; del buf347  # reuse
        buf370 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf368, arg31_1, buf369, buf370, 1280, 4096, stream=stream0)
        buf372 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg37_1, (256, 512), (1, 256), 0), out=buf372)
        del arg37_1
        buf373 = reinterpret_tensor(buf343, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf368, arg31_1, buf369, buf370, buf373, 5120, 1024, stream=stream0)
        del arg31_1
        buf374 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, v_54, transpose_84, x_flat_18, v_t_x_17], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf373, (40, 128, 1024), (131072, 1024, 1), 0), out=buf374)
        buf375 = reinterpret_tensor(buf368, (40, 128, 1024), (131072, 1024, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [split_17, u_35, mixed_17], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 128, 12), (3200, 12, 1), 0), buf374, out=buf375)
        buf376 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, x_flat_18, out_58, view_136, shift_35, out_59, x_61], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf373, buf375, buf360, buf376, 5120, 1024, stream=stream0)
        del buf373
        del buf375
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, x_flat_18, out_58, view_136, shift_35, out_59, x_61, x_62], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf377 = extern_kernels.convolution(buf376, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg32_1
        del buf376
        buf378 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, x_flat_18, out_58, view_136, shift_35, out_59, x_61, x_62, add_65, h_25], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf378, arg33_1, buf366, 5242880, stream=stream0)
        del arg33_1
        buf379 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [h_26], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_13.run(arg34_1, buf379, 32768, 9, stream=stream0)
        del arg34_1
        # Topologically Sorted Source Nodes: [h_26], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf378, buf379, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf379
        buf381 = reinterpret_tensor(buf276, (40, 256, 256), (65536, 256, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [h_26, view_137, q_28, q_29], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_transpose_view_14.run(buf380, arg35_1, buf381, 2621440, stream=stream0)
        buf382 = reinterpret_tensor(buf254, (10240, 256), (256, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [h_26, view_137, q_28, q_29], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (10240, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 256), (1, 256), 0), out=buf382)
        del arg36_1
        buf383 = reinterpret_tensor(buf381, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [q_29, view_138, q_30, matmul_20], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf382, buf383, 2621440, stream=stream0)
        buf384 = reinterpret_tensor(buf359, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, view_139, k_20, transpose_89, matmul_20], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf372, buf384, 10240, stream=stream0)
        buf385 = reinterpret_tensor(buf351, (160, 256, 1), (256, 1, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, q_29, view_138, q_30, matmul_20, view_139, k_20, transpose_89], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf383, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf384, (160, 64, 1), (64, 1, 0), 0), out=buf385)
        buf386 = reinterpret_tensor(buf385, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [matmul_20, attn_21], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf386, 40960, stream=stream0)
        buf387 = reinterpret_tensor(buf384, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, view_140, v_56, matmul_21], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf372, buf387, 10240, stream=stream0)
        buf388 = reinterpret_tensor(buf383, (160, 256, 64), (16384, 64, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, matmul_20, attn_21, matmul_21, view_140, v_56], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf386, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf387, (160, 1, 64), (64, 0, 1), 0), out=buf388)
        buf389 = reinterpret_tensor(buf382, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [matmul_21, transpose_90, out_60], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf388, buf389, 2621440, stream=stream0)
        buf390 = reinterpret_tensor(buf388, (10240, 256), (256, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [matmul_21, transpose_90, out_60, input_60], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf389, (10240, 256), (256, 1), 0), reinterpret_tensor(arg38_1, (256, 256), (1, 256), 0), out=buf390)
        del arg38_1
        buf391 = reinterpret_tensor(buf390, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_20.run(buf391, buf380, arg35_1, arg39_1, 2621440, stream=stream0)
        del arg39_1
        buf392 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg40_1, buf392, 65536, 9, stream=stream0)
        del arg40_1
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf393 = extern_kernels.convolution(buf391, buf392, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf394 = buf370; del buf370  # reuse
        buf395 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf393, arg41_1, buf394, buf395, 1280, 2048, stream=stream0)
        buf397 = reinterpret_tensor(buf387, (40, 256), (256, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg15_1, (256, 256), (1, 256), 0), out=buf397)
        del arg15_1
        buf398 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf398, arg16_1, 10240, stream=stream0)
        del arg16_1
        buf399 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55, input_56], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg18_1, buf398, reinterpret_tensor(arg17_1, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf399)
        del arg17_1
        del arg18_1
        buf400 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg45_1, (256, 512), (1, 256), 0), out=buf400)
        del arg45_1
        buf401 = reinterpret_tensor(buf391, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf393, arg41_1, buf394, buf395, buf401, 10240, 256, stream=stream0)
        del arg41_1
        buf402 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, v_58, transpose_92, x_flat_19, v_t_x_18], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf401, (40, 256, 256), (65536, 256, 1), 0), out=buf402)
        buf403 = reinterpret_tensor(buf393, (40, 256, 256), (65536, 256, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [split_18, u_37, mixed_18], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 256, 12), (6400, 12, 1), 0), buf402, out=buf403)
        buf404 = reinterpret_tensor(buf389, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf401, buf403, buf399, buf404, 10240, 256, stream=stream0)
        del buf401
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf405 = extern_kernels.convolution(buf404, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg42_1
        buf406 = reinterpret_tensor(buf404, (40, 256, 256), (65536, 256, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, view_147, q_31, q_32], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf405, arg43_1, buf380, arg35_1, buf406, 2621440, stream=stream0)
        buf407 = reinterpret_tensor(buf403, (10240, 256), (256, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, view_147, q_31, q_32], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (10240, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 256), (1, 256), 0), out=buf407)
        del arg44_1
        buf408 = reinterpret_tensor(buf406, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [q_32, view_148, q_33, matmul_22], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf407, buf408, 2621440, stream=stream0)
        buf409 = reinterpret_tensor(buf398, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, view_149, k_22, transpose_97, matmul_22], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf400, buf409, 10240, stream=stream0)
        buf410 = reinterpret_tensor(buf386, (160, 256, 1), (256, 1, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, q_32, view_148, q_33, matmul_22, view_149, k_22, transpose_97], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf408, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf409, (160, 64, 1), (64, 1, 0), 0), out=buf410)
        buf411 = reinterpret_tensor(buf410, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [matmul_22, attn_23], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf411, 40960, stream=stream0)
        buf412 = reinterpret_tensor(buf409, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, view_150, v_60, matmul_23], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf400, buf412, 10240, stream=stream0)
        buf413 = reinterpret_tensor(buf408, (160, 256, 64), (16384, 64, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, matmul_22, attn_23, matmul_23, view_150, v_60], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf411, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf412, (160, 1, 64), (64, 0, 1), 0), out=buf413)
        buf414 = reinterpret_tensor(buf407, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [matmul_23, transpose_98, out_64], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf413, buf414, 2621440, stream=stream0)
        buf415 = reinterpret_tensor(buf413, (10240, 256), (256, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [matmul_23, transpose_98, out_64, input_62], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf414, (10240, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 256), (1, 256), 0), out=buf415)
        del arg46_1
        buf416 = reinterpret_tensor(buf415, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26.run(buf416, buf405, arg43_1, buf380, arg35_1, arg47_1, 2621440, stream=stream0)
        del arg47_1
        buf417 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg48_1, buf417, 65536, 9, stream=stream0)
        del arg48_1
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf418 = extern_kernels.convolution(buf416, buf417, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf419 = buf395; del buf395  # reuse
        buf420 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf418, arg49_1, buf419, buf420, 1280, 2048, stream=stream0)
        buf422 = reinterpret_tensor(buf411, (40, 1024), (1024, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg55_1, (256, 1024), (1, 256), 0), out=buf422)
        del arg55_1
        buf423 = reinterpret_tensor(buf416, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf418, arg49_1, buf419, buf420, buf423, 10240, 256, stream=stream0)
        del arg49_1
        buf424 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, v_62, transpose_100, x_flat_20, v_t_x_19], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf423, (40, 256, 256), (65536, 256, 1), 0), out=buf424)
        buf425 = reinterpret_tensor(buf418, (40, 256, 256), (65536, 256, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [split_19, u_39, mixed_19], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 256, 12), (6400, 12, 1), 0), buf424, out=buf425)
        buf426 = reinterpret_tensor(buf414, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, x_flat_20, out_66, view_156, shift_39, out_67, x_69], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf423, buf425, buf399, buf426, 10240, 256, stream=stream0)
        del buf423
        del buf425
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, x_flat_20, out_66, view_156, shift_39, out_67, x_69, x_70], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf427 = extern_kernels.convolution(buf426, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg50_1
        del buf426
        buf428 = buf427; del buf427  # reuse
        buf429 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, x_flat_20, out_66, view_156, shift_39, out_67, x_69, x_70, add_73, h_28], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27.run(buf429, arg51_1, buf405, arg43_1, buf380, arg35_1, 2621440, stream=stream0)
        del arg35_1
        del arg43_1
        del arg51_1
        del buf380
        buf430 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_28.run(arg52_1, buf430, 131072, 9, stream=stream0)
        del arg52_1
        # Topologically Sorted Source Nodes: [h_28, h_29], Original ATen: [aten.silu, aten.convolution]
        buf431 = extern_kernels.convolution(buf429, buf430, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf430
        buf432 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, view_157, q_34, q_35], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_silu_transpose_view_29.run(buf431, arg53_1, buf432, 1310720, stream=stream0)
        buf433 = empty_strided_cuda((2560, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_28, h_29, view_157, q_34, q_35], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (2560, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf433)
        del arg54_1
        buf434 = reinterpret_tensor(buf432, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [q_35, view_158, q_36, matmul_24], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf433, buf434, 1310720, stream=stream0)
        buf435 = reinterpret_tensor(buf400, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, view_159, k_24, transpose_105, matmul_24], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf422, buf435, 20480, stream=stream0)
        buf436 = reinterpret_tensor(buf354, (320, 64, 1), (64, 1, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, q_35, view_158, q_36, matmul_24, view_159, k_24, transpose_105], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf434, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf435, (320, 64, 1), (64, 1, 0), 0), out=buf436)
        buf437 = reinterpret_tensor(buf436, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [matmul_24, attn_25], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_32.run(buf437, 20480, stream=stream0)
        buf438 = reinterpret_tensor(buf435, (40, 8, 1, 64), (512, 64, 64, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, view_160, v_64, matmul_25], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf422, buf438, 20480, stream=stream0)
        buf439 = reinterpret_tensor(buf434, (320, 64, 64), (4096, 64, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, matmul_24, attn_25, matmul_25, view_160, v_64], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf437, (320, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf438, (320, 1, 64), (64, 0, 1), 0), out=buf439)
        buf440 = reinterpret_tensor(buf433, (40, 64, 8, 64), (32768, 512, 64, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [matmul_25, transpose_106, out_68], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf439, buf440, 1310720, stream=stream0)
        buf441 = reinterpret_tensor(buf439, (2560, 512), (512, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [matmul_25, transpose_106, out_68, input_64], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf440, (2560, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 512), (1, 512), 0), out=buf441)
        del arg56_1
        buf442 = reinterpret_tensor(buf441, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_35.run(buf442, buf431, arg53_1, arg57_1, 1310720, stream=stream0)
        del arg57_1
        buf443 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(arg58_1, buf443, 262144, 9, stream=stream0)
        del arg58_1
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf444 = extern_kernels.convolution(buf442, buf443, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf445 = buf420; del buf420  # reuse
        buf446 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37.run(buf444, arg59_1, buf445, buf446, 1280, 1024, stream=stream0)
        buf448 = reinterpret_tensor(buf412, (40, 256), (256, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg19_1, (256, 256), (1, 256), 0), out=buf448)
        del arg19_1
        buf449 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf449, arg20_1, 10240, stream=stream0)
        del arg20_1
        buf450 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58, input_59], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg22_1, buf449, reinterpret_tensor(arg21_1, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf450)
        del arg21_1
        del arg22_1
        buf451 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg63_1, (256, 1024), (1, 256), 0), out=buf451)
        del arg63_1
        buf452 = reinterpret_tensor(buf442, (40, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf444, arg59_1, buf445, buf446, buf452, 20480, 64, stream=stream0)
        del arg59_1
        buf453 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, v_66, transpose_108, x_flat_21, v_t_x_20], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (40, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf452, (40, 512, 64), (32768, 64, 1), 0), out=buf453)
        buf454 = reinterpret_tensor(buf444, (40, 512, 64), (32768, 64, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [split_20, u_41, mixed_20], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (40, 512, 12), (12800, 12, 1), 0), buf453, out=buf454)
        buf455 = reinterpret_tensor(buf440, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39.run(buf452, buf454, buf450, buf455, 20480, 64, stream=stream0)
        del buf452
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf456 = extern_kernels.convolution(buf455, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del arg60_1
        buf457 = reinterpret_tensor(buf455, (40, 64, 512), (32768, 512, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, view_167, q_37, q_38], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40.run(buf456, arg61_1, buf431, arg53_1, buf457, 1310720, stream=stream0)
        buf458 = reinterpret_tensor(buf454, (2560, 512), (512, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, view_167, q_37, q_38], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (2560, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 512), (1, 512), 0), out=buf458)
        del arg62_1
        buf459 = reinterpret_tensor(buf457, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [q_38, view_168, q_39, matmul_26], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf458, buf459, 1310720, stream=stream0)
        buf460 = reinterpret_tensor(buf438, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, view_169, k_26, transpose_113, matmul_26], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf451, buf460, 20480, stream=stream0)
        buf461 = reinterpret_tensor(buf437, (320, 64, 1), (64, 1, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, q_38, view_168, q_39, matmul_26, view_169, k_26, transpose_113], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf459, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf460, (320, 64, 1), (64, 1, 0), 0), out=buf461)
        buf462 = reinterpret_tensor(buf461, (40, 8, 64, 1), (512, 64, 1, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [matmul_26, attn_27], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_32.run(buf462, 20480, stream=stream0)
        buf463 = reinterpret_tensor(buf460, (40, 8, 1, 64), (512, 64, 64, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, view_170, v_68, matmul_27], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf451, buf463, 20480, stream=stream0)
        buf464 = reinterpret_tensor(buf459, (320, 64, 64), (4096, 64, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, matmul_26, attn_27, matmul_27, view_170, v_68], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (320, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf463, (320, 1, 64), (64, 0, 1), 0), out=buf464)
        del buf462
        buf465 = reinterpret_tensor(buf458, (40, 64, 8, 64), (32768, 512, 64, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [matmul_27, transpose_114, out_72], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf464, buf465, 1310720, stream=stream0)
        buf466 = reinterpret_tensor(buf464, (2560, 512), (512, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [matmul_27, transpose_114, out_72, input_66], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf465, (2560, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf466)
        del arg64_1
        buf467 = reinterpret_tensor(buf466, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41.run(buf467, buf456, arg61_1, buf431, arg53_1, arg65_1, 1310720, stream=stream0)
        del arg65_1
        buf468 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(arg66_1, buf468, 262144, 9, stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf469 = extern_kernels.convolution(buf467, buf468, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf468
        buf470 = buf446; del buf446  # reuse
        buf471 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37.run(buf469, arg67_1, buf470, buf471, 1280, 1024, stream=stream0)
        buf473 = reinterpret_tensor(buf467, (40, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf469, arg67_1, buf470, buf471, buf473, 20480, 64, stream=stream0)
        del arg67_1
        buf474 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, v_70, transpose_116, x_flat_22, v_t_x_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (40, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf473, (40, 512, 64), (32768, 64, 1), 0), out=buf474)
        buf475 = reinterpret_tensor(buf469, (40, 512, 64), (32768, 64, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [split_21, u_43, mixed_21], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (40, 512, 12), (12800, 12, 1), 0), buf474, out=buf475)
        del buf474
        buf476 = reinterpret_tensor(buf465, (40, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, x_flat_22, out_74, view_176, shift_43, out_75, x_77], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39.run(buf473, buf475, buf450, buf476, 20480, 64, stream=stream0)
        del buf450
        del buf473
        del buf475
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, x_flat_22, out_74, view_176, shift_43, out_75, x_77, x_78], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf477 = extern_kernels.convolution(buf476, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (40, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del arg68_1
        buf478 = buf477; del buf477  # reuse
        buf482 = reinterpret_tensor(buf476, (40, 64, 512), (32768, 512, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, x_flat_22, out_74, view_176, shift_43, out_75, x_77, x_78, add_81, h_31, view_177, x_flat_23, x_norm_23], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42.run(buf478, arg69_1, buf456, arg61_1, buf431, arg53_1, arg70_1, arg71_1, buf482, 2560, 512, stream=stream0)
        del arg53_1
        del arg61_1
        del arg69_1
        del arg70_1
        del arg71_1
        buf483 = empty_strided_cuda((2560, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, x_norm_23, qkv_3], Original ATen: [aten.silu, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf482, (2560, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 1536), (1, 512), 0), out=buf483)
        del arg72_1
        buf484 = reinterpret_tensor(buf482, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, q_40, matmul_28], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_43.run(buf483, arg73_1, buf484, 1310720, stream=stream0)
        buf485 = reinterpret_tensor(buf456, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, k_27, transpose_118, matmul_28], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_transpose_view_44.run(buf483, arg73_1, buf485, 20480, 64, stream=stream0)
        buf486 = reinterpret_tensor(buf431, (320, 64, 64), (4096, 64, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, q_40, matmul_28, k_27, transpose_118], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone, aten._unsafe_view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf484, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf485, (320, 64, 64), (4096, 64, 1), 0), out=buf486)
        buf489 = reinterpret_tensor(buf486, (40, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [matmul_28, attn_29], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_amax_mul_sub_view_45.run(buf489, 20480, 64, stream=stream0)
        buf490 = buf485; del buf485  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, v_71, out_76], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_46.run(buf483, arg73_1, buf490, 1310720, stream=stream0)
        del arg73_1
        del buf483
        buf491 = reinterpret_tensor(buf484, (320, 64, 64), (4096, 64, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, matmul_28, attn_29, out_76, v_71], Original ATen: [aten.addmm, aten.view, aten.permute, aten.mul, aten.sub, aten._softmax, aten.select, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (320, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf490, (320, 64, 64), (4096, 64, 1), 0), out=buf491)
        del buf489
        buf492 = reinterpret_tensor(buf490, (40, 64, 8, 64), (32768, 512, 64, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [out_76, transpose_119, out_77], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf491, buf492, 1310720, stream=stream0)
        buf493 = reinterpret_tensor(buf491, (2560, 512), (512, 1), 0); del buf491  # reuse
        # Topologically Sorted Source Nodes: [out_76, transpose_119, out_77, out_78], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf492, (2560, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf493)
        del arg74_1
        del buf492
        buf498 = reinterpret_tensor(buf478, (40, 64, 512), (32768, 512, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47.run(buf498, buf493, arg75_1, arg76_1, arg77_1, 2560, 512, stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        del buf493
        buf497 = reinterpret_tensor(buf463, (40, 512), (512, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg85_1, (256, 512), (1, 256), 0), out=buf497)
        del arg85_1
        buf499 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80, transpose_120, out_81, h_32], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_48.run(arg78_1, buf499, 131072, 16, stream=stream0)
        del arg78_1
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80, transpose_120, out_81, h_32], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        buf500 = extern_kernels.convolution(reinterpret_tensor(buf498, (40, 512, 8, 8), (32768, 1, 4096, 512), 0), buf499, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf498
        del buf499
        buf501 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg80_1, (256, 256), (1, 256), 0), out=buf501)
        del arg80_1
        buf502 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf502, arg81_1, 10240, stream=stream0)
        del arg81_1
        buf503 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69, input_70], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf502, reinterpret_tensor(arg82_1, (256, 256), (1, 256), 0), out=buf503)
        del arg82_1
        del buf502
        buf504 = reinterpret_tensor(buf405, (40, 256, 16, 16), (65536, 256, 16, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80, transpose_120, out_81, h_32, input_70, input_71, unsqueeze_18, gate_2, h_16_gated_1, h_33], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_49.run(buf500, arg79_1, buf429, buf503, arg83_1, buf504, 10240, 256, stream=stream0)
        del arg79_1
        del arg83_1
        buf505 = reinterpret_tensor(buf500, (10240, 256), (1, 10240), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [view_179, q_41, q_42], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_t_transpose_view_50.run(buf504, buf505, 2621440, stream=stream0)
        buf506 = reinterpret_tensor(buf429, (10240, 256), (256, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [view_179, q_41, q_42], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        extern_kernels.mm(buf505, reinterpret_tensor(arg84_1, (256, 256), (1, 256), 0), out=buf506)
        del arg84_1
        buf507 = reinterpret_tensor(buf505, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [q_42, view_180, q_43, matmul_30], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf506, buf507, 2621440, stream=stream0)
        buf508 = reinterpret_tensor(buf503, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, view_181, k_29, transpose_125, matmul_30], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf497, buf508, 10240, stream=stream0)
        buf509 = reinterpret_tensor(buf451, (160, 256, 1), (256, 1, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, q_42, view_180, q_43, matmul_30, view_181, k_29, transpose_125], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf507, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf508, (160, 64, 1), (64, 1, 0), 0), out=buf509)
        buf510 = reinterpret_tensor(buf509, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [matmul_30, attn_31], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf510, 40960, stream=stream0)
        buf511 = reinterpret_tensor(buf508, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, view_182, v_73, matmul_31], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf497, buf511, 10240, stream=stream0)
        buf512 = reinterpret_tensor(buf507, (160, 256, 64), (16384, 64, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, matmul_30, attn_31, matmul_31, view_182, v_73], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf510, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf511, (160, 1, 64), (64, 0, 1), 0), out=buf512)
        buf513 = reinterpret_tensor(buf506, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [matmul_31, transpose_126, out_82], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf512, buf513, 2621440, stream=stream0)
        buf514 = reinterpret_tensor(buf512, (10240, 256), (256, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [matmul_31, transpose_126, out_82, input_72], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf513, (10240, 256), (256, 1), 0), reinterpret_tensor(arg86_1, (256, 256), (1, 256), 0), out=buf514)
        del arg86_1
        buf515 = reinterpret_tensor(buf514, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_transpose_view_51.run(buf515, buf504, arg87_1, 10240, 256, stream=stream0)
        del arg87_1
        buf516 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg88_1, buf516, 65536, 9, stream=stream0)
        del arg88_1
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf517 = extern_kernels.convolution(buf515, buf516, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf518 = buf471; del buf471  # reuse
        buf519 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf517, arg89_1, buf518, buf519, 1280, 2048, stream=stream0)
        buf521 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg93_1, (256, 512), (1, 256), 0), out=buf521)
        del arg93_1
        buf522 = reinterpret_tensor(buf515, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf517, arg89_1, buf518, buf519, buf522, 10240, 256, stream=stream0)
        del arg89_1
        buf523 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, v_75, transpose_128, x_flat_24, v_t_x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf522, (40, 256, 256), (65536, 256, 1), 0), out=buf523)
        buf524 = reinterpret_tensor(buf517, (40, 256, 256), (65536, 256, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [split_22, u_45, mixed_22], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 256, 12), (6400, 12, 1), 0), buf523, out=buf524)
        buf525 = reinterpret_tensor(buf513, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf522, buf524, buf399, buf525, 10240, 256, stream=stream0)
        del buf522
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf526 = extern_kernels.convolution(buf525, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg90_1
        buf527 = reinterpret_tensor(buf525, (40, 256, 256), (65536, 256, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, view_189, q_44, q_45], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_52.run(buf526, arg91_1, buf504, buf527, 10240, 256, stream=stream0)
        buf528 = reinterpret_tensor(buf524, (10240, 256), (256, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, view_189, q_44, q_45], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (10240, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf528)
        del arg92_1
        buf529 = reinterpret_tensor(buf527, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [q_45, view_190, q_46, matmul_32], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf528, buf529, 2621440, stream=stream0)
        buf530 = reinterpret_tensor(buf511, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, view_191, k_31, transpose_133, matmul_32], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf521, buf530, 10240, stream=stream0)
        buf531 = reinterpret_tensor(buf510, (160, 256, 1), (256, 1, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, q_45, view_190, q_46, matmul_32, view_191, k_31, transpose_133], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf529, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf530, (160, 64, 1), (64, 1, 0), 0), out=buf531)
        buf532 = reinterpret_tensor(buf531, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [matmul_32, attn_33], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf532, 40960, stream=stream0)
        buf533 = reinterpret_tensor(buf530, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, view_192, v_77, matmul_33], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf521, buf533, 10240, stream=stream0)
        buf534 = reinterpret_tensor(buf529, (160, 256, 64), (16384, 64, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, matmul_32, attn_33, matmul_33, view_192, v_77], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf532, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf533, (160, 1, 64), (64, 0, 1), 0), out=buf534)
        buf535 = reinterpret_tensor(buf528, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [matmul_33, transpose_134, out_86], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf534, buf535, 2621440, stream=stream0)
        buf536 = reinterpret_tensor(buf534, (10240, 256), (256, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [matmul_33, transpose_134, out_86, input_74], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf535, (10240, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 256), (1, 256), 0), out=buf536)
        del arg94_1
        buf537 = reinterpret_tensor(buf536, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53.run(buf537, buf526, arg91_1, buf504, arg95_1, 10240, 256, stream=stream0)
        del arg95_1
        buf538 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg96_1, buf538, 65536, 9, stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf539 = extern_kernels.convolution(buf537, buf538, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf540 = buf519; del buf519  # reuse
        buf541 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf539, arg97_1, buf540, buf541, 1280, 2048, stream=stream0)
        buf543 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg101_1, (256, 512), (1, 256), 0), out=buf543)
        del arg101_1
        buf544 = reinterpret_tensor(buf537, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf537  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf539, arg97_1, buf540, buf541, buf544, 10240, 256, stream=stream0)
        del arg97_1
        buf545 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, v_79, transpose_136, x_flat_25, v_t_x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf544, (40, 256, 256), (65536, 256, 1), 0), out=buf545)
        buf546 = reinterpret_tensor(buf539, (40, 256, 256), (65536, 256, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [split_23, u_47, mixed_23], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 256, 12), (6400, 12, 1), 0), buf545, out=buf546)
        buf547 = reinterpret_tensor(buf535, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, x_flat_25, out_88, view_198, shift_47, out_89, x_85], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf544, buf546, buf399, buf547, 10240, 256, stream=stream0)
        del buf544
        del buf546
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, x_flat_25, out_88, view_198, shift_47, out_89, x_85, x_86], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf548 = extern_kernels.convolution(buf547, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg98_1
        buf549 = buf548; del buf548  # reuse
        buf550 = reinterpret_tensor(buf547, (40, 256, 256), (65536, 256, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, x_flat_25, out_88, view_198, shift_47, out_89, x_85, x_86, add_91, h_35, view_199, q_47, q_48], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54.run(buf549, arg99_1, buf526, arg91_1, buf504, buf550, 2621440, stream=stream0)
        del arg91_1
        del arg99_1
        del buf504
        buf551 = reinterpret_tensor(buf526, (10240, 256), (256, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [h_35, view_199, q_47, q_48], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf550, (10240, 256), (256, 1), 0), reinterpret_tensor(arg100_1, (256, 256), (1, 256), 0), out=buf551)
        del arg100_1
        buf552 = reinterpret_tensor(buf550, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [q_48, view_200, q_49, matmul_34], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf551, buf552, 2621440, stream=stream0)
        buf553 = reinterpret_tensor(buf533, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, view_201, k_33, transpose_141, matmul_34], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf543, buf553, 10240, stream=stream0)
        buf554 = reinterpret_tensor(buf532, (160, 256, 1), (256, 1, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, q_48, view_200, q_49, matmul_34, view_201, k_33, transpose_141], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf552, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf553, (160, 64, 1), (64, 1, 0), 0), out=buf554)
        buf555 = reinterpret_tensor(buf554, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [matmul_34, attn_35], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf555, 40960, stream=stream0)
        buf556 = reinterpret_tensor(buf553, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, view_202, v_81, matmul_35], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf543, buf556, 10240, stream=stream0)
        buf557 = reinterpret_tensor(buf552, (160, 256, 64), (16384, 64, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, matmul_34, attn_35, matmul_35, view_202, v_81], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf555, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf556, (160, 1, 64), (64, 0, 1), 0), out=buf557)
        buf558 = reinterpret_tensor(buf551, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [matmul_35, transpose_142, out_90], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf557, buf558, 2621440, stream=stream0)
        buf559 = reinterpret_tensor(buf557, (10240, 256), (256, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [matmul_35, transpose_142, out_90, input_76], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf558, (10240, 256), (256, 1), 0), reinterpret_tensor(arg102_1, (256, 256), (1, 256), 0), out=buf559)
        del arg102_1
        buf560 = reinterpret_tensor(buf559, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_55.run(buf560, buf549, arg103_1, 2621440, stream=stream0)
        del arg103_1
        buf561 = buf538; del buf538  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg104_1, buf561, 65536, 9, stream=stream0)
        del arg104_1
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf562 = extern_kernels.convolution(buf560, buf561, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf563 = buf541; del buf541  # reuse
        buf564 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf562, arg105_1, buf563, buf564, 1280, 2048, stream=stream0)
        buf566 = buf543; del buf543  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg109_1, (256, 512), (1, 256), 0), out=buf566)
        del arg109_1
        buf567 = reinterpret_tensor(buf560, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf562, arg105_1, buf563, buf564, buf567, 10240, 256, stream=stream0)
        del arg105_1
        buf568 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, v_83, transpose_144, x_flat_26, v_t_x_24], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf567, (40, 256, 256), (65536, 256, 1), 0), out=buf568)
        buf569 = reinterpret_tensor(buf562, (40, 256, 256), (65536, 256, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [split_24, u_49, mixed_24], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 256, 12), (6400, 12, 1), 0), buf568, out=buf569)
        buf570 = reinterpret_tensor(buf558, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf567, buf569, buf399, buf570, 10240, 256, stream=stream0)
        del buf567
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf571 = extern_kernels.convolution(buf570, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg106_1
        buf572 = reinterpret_tensor(buf570, (40, 256, 256), (65536, 256, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, view_209, q_50, q_51], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_56.run(buf571, arg107_1, buf549, buf572, 2621440, stream=stream0)
        buf573 = reinterpret_tensor(buf569, (10240, 256), (256, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, view_209, q_50, q_51], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (10240, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), out=buf573)
        del arg108_1
        buf574 = reinterpret_tensor(buf572, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf572  # reuse
        # Topologically Sorted Source Nodes: [q_51, view_210, q_52, matmul_36], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf573, buf574, 2621440, stream=stream0)
        buf575 = reinterpret_tensor(buf556, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, view_211, k_35, transpose_149, matmul_36], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf566, buf575, 10240, stream=stream0)
        buf576 = reinterpret_tensor(buf555, (160, 256, 1), (256, 1, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, q_51, view_210, q_52, matmul_36, view_211, k_35, transpose_149], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf574, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf575, (160, 64, 1), (64, 1, 0), 0), out=buf576)
        buf577 = reinterpret_tensor(buf576, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf576  # reuse
        # Topologically Sorted Source Nodes: [matmul_36, attn_37], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf577, 40960, stream=stream0)
        buf578 = reinterpret_tensor(buf575, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf575  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, view_212, v_85, matmul_37], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf566, buf578, 10240, stream=stream0)
        buf579 = reinterpret_tensor(buf574, (160, 256, 64), (16384, 64, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, matmul_36, attn_37, matmul_37, view_212, v_85], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf578, (160, 1, 64), (64, 0, 1), 0), out=buf579)
        buf580 = reinterpret_tensor(buf573, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [matmul_37, transpose_150, out_94], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf579, buf580, 2621440, stream=stream0)
        buf581 = reinterpret_tensor(buf579, (10240, 256), (256, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [matmul_37, transpose_150, out_94, input_78], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf580, (10240, 256), (256, 1), 0), reinterpret_tensor(arg110_1, (256, 256), (1, 256), 0), out=buf581)
        del arg110_1
        buf582 = reinterpret_tensor(buf581, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf581  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57.run(buf582, buf571, arg107_1, buf549, arg111_1, 2621440, stream=stream0)
        del arg111_1
        buf583 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg112_1, buf583, 65536, 9, stream=stream0)
        del arg112_1
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf584 = extern_kernels.convolution(buf582, buf583, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf585 = buf564; del buf564  # reuse
        buf586 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf584, arg113_1, buf585, buf586, 1280, 2048, stream=stream0)
        buf588 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg117_1, (256, 512), (1, 256), 0), out=buf588)
        del arg117_1
        buf589 = reinterpret_tensor(buf582, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf584, arg113_1, buf585, buf586, buf589, 10240, 256, stream=stream0)
        del arg113_1
        buf590 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, v_87, transpose_152, x_flat_27, v_t_x_25], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf589, (40, 256, 256), (65536, 256, 1), 0), out=buf590)
        buf591 = reinterpret_tensor(buf584, (40, 256, 256), (65536, 256, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [split_25, u_51, mixed_25], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 256, 12), (6400, 12, 1), 0), buf590, out=buf591)
        buf592 = reinterpret_tensor(buf580, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, x_flat_27, out_96, view_218, shift_51, out_97, x_93], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf589, buf591, buf399, buf592, 10240, 256, stream=stream0)
        del buf589
        del buf591
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, x_flat_27, out_96, view_218, shift_51, out_97, x_93, x_94], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf593 = extern_kernels.convolution(buf592, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg114_1
        buf594 = buf593; del buf593  # reuse
        buf595 = reinterpret_tensor(buf592, (40, 256, 256), (65536, 256, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, x_flat_27, out_96, view_218, shift_51, out_97, x_93, x_94, add_99, h_37, view_219, q_53, q_54], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58.run(buf594, arg115_1, buf571, arg107_1, buf549, buf595, 2621440, stream=stream0)
        del arg107_1
        del arg115_1
        del buf549
        buf596 = reinterpret_tensor(buf571, (10240, 256), (256, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [h_37, view_219, q_53, q_54], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf595, (10240, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 256), (1, 256), 0), out=buf596)
        del arg116_1
        buf597 = reinterpret_tensor(buf595, (40, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [q_54, view_220, q_55, matmul_38], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf596, buf597, 2621440, stream=stream0)
        buf598 = reinterpret_tensor(buf578, (40, 4, 64, 1), (256, 64, 1, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, view_221, k_37, transpose_157, matmul_38], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf588, buf598, 10240, stream=stream0)
        buf599 = reinterpret_tensor(buf577, (160, 256, 1), (256, 1, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, q_54, view_220, q_55, matmul_38, view_221, k_37, transpose_157], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf597, (160, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf598, (160, 64, 1), (64, 1, 0), 0), out=buf599)
        buf600 = reinterpret_tensor(buf599, (40, 4, 256, 1), (1024, 256, 1, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [matmul_38, attn_39], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf600, 40960, stream=stream0)
        buf601 = reinterpret_tensor(buf598, (40, 4, 1, 64), (256, 64, 64, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, view_222, v_89, matmul_39], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf588, buf601, 10240, stream=stream0)
        del buf588
        buf602 = reinterpret_tensor(buf597, (160, 256, 64), (16384, 64, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, matmul_38, attn_39, matmul_39, view_222, v_89], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf600, (160, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf601, (160, 1, 64), (64, 0, 1), 0), out=buf602)
        del buf600
        buf603 = reinterpret_tensor(buf596, (40, 256, 4, 64), (65536, 256, 64, 1), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [matmul_39, transpose_158, out_98], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf602, buf603, 2621440, stream=stream0)
        buf604 = reinterpret_tensor(buf602, (10240, 256), (256, 1), 0); del buf602  # reuse
        # Topologically Sorted Source Nodes: [matmul_39, transpose_158, out_98, input_80], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf603, (10240, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 256), (1, 256), 0), out=buf604)
        del arg118_1
        buf605 = reinterpret_tensor(buf604, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf604  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_55.run(buf605, buf594, arg119_1, 2621440, stream=stream0)
        del arg119_1
        buf606 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg120_1, buf606, 65536, 9, stream=stream0)
        del arg120_1
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf607 = extern_kernels.convolution(buf605, buf606, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf607, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf606
        buf608 = buf586; del buf586  # reuse
        buf609 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf607, arg121_1, buf608, buf609, 1280, 2048, stream=stream0)
        buf611 = reinterpret_tensor(buf605, (40, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf607, arg121_1, buf608, buf609, buf611, 10240, 256, stream=stream0)
        del arg121_1
        buf612 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, v_91, transpose_160, x_flat_28, v_t_x_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf611, (40, 256, 256), (65536, 256, 1), 0), out=buf612)
        buf613 = reinterpret_tensor(buf607, (40, 256, 256), (65536, 256, 1), 0); del buf607  # reuse
        # Topologically Sorted Source Nodes: [split_26, u_53, mixed_26], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (40, 256, 12), (6400, 12, 1), 0), buf612, out=buf613)
        del buf612
        buf614 = reinterpret_tensor(buf603, (40, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf603  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf611, buf613, buf399, buf614, 10240, 256, stream=stream0)
        del buf399
        del buf611
        del buf613
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf615 = extern_kernels.convolution(buf614, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf615, (40, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg122_1
        del buf614
        buf616 = buf615; del buf615  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59.run(buf616, arg123_1, buf594, 2621440, stream=stream0)
        del arg123_1
        del buf594
        buf617 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38, h_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60.run(arg124_1, buf617, 32768, 16, stream=stream0)
        del arg124_1
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38, h_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf618 = extern_kernels.convolution(buf616, buf617, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf616
        del buf617
        buf619 = reinterpret_tensor(buf601, (40, 256), (256, 1), 0); del buf601  # reuse
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf357, reinterpret_tensor(arg126_1, (256, 256), (1, 256), 0), out=buf619)
        del arg126_1
        del buf357
        buf620 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf620, arg127_1, 10240, stream=stream0)
        del arg127_1
        buf621 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf620, reinterpret_tensor(arg128_1, (256, 128), (1, 256), 0), out=buf621)
        del arg128_1
        del buf620
        buf622 = buf618; del buf618  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38, h_39, input_84, input_85, unsqueeze_25, gate_3, h_32_gated_1, h_40], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_61.run(buf622, arg125_1, buf378, buf621, arg129_1, 5242880, stream=stream0)
        del arg125_1
        del arg129_1
        del buf621
        buf623 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg130_1, buf623, 16384, 9, stream=stream0)
        del arg130_1
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf622, buf623, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf625 = buf609; del buf609  # reuse
        buf626 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf624, arg131_1, buf625, buf626, 1280, 4096, stream=stream0)
        buf628 = reinterpret_tensor(buf378, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf624, arg131_1, buf625, buf626, buf628, 5120, 1024, stream=stream0)
        del arg131_1
        buf629 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, v_93, transpose_161, x_flat_29, v_t_x_27], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf628, (40, 128, 1024), (131072, 1024, 1), 0), out=buf629)
        buf630 = reinterpret_tensor(buf624, (40, 128, 1024), (131072, 1024, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [split_27, u_55, mixed_27], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 128, 12), (3200, 12, 1), 0), buf629, out=buf630)
        buf631 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, x_flat_29, out_102, view_233, shift_55, out_103, x_100], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf628, buf630, buf360, buf631, 5120, 1024, stream=stream0)
        del buf628
        del buf630
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, x_flat_29, out_102, view_233, shift_55, out_103, x_100, x_101], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf632 = extern_kernels.convolution(buf631, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg132_1
        buf633 = buf632; del buf632  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, x_flat_29, out_102, view_233, shift_55, out_103, x_100, x_101, add_107, h_41], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf633, arg133_1, buf622, 5242880, stream=stream0)
        del arg133_1
        buf634 = buf623; del buf623  # reuse
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg134_1, buf634, 16384, 9, stream=stream0)
        del arg134_1
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf635 = extern_kernels.convolution(buf633, buf634, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf635, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf636 = buf626; del buf626  # reuse
        buf637 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf635, arg135_1, buf636, buf637, 1280, 4096, stream=stream0)
        buf639 = reinterpret_tensor(buf622, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf635, arg135_1, buf636, buf637, buf639, 5120, 1024, stream=stream0)
        del arg135_1
        buf640 = buf629; del buf629  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, v_95, transpose_162, x_flat_30, v_t_x_28], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf639, (40, 128, 1024), (131072, 1024, 1), 0), out=buf640)
        buf641 = reinterpret_tensor(buf635, (40, 128, 1024), (131072, 1024, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [split_28, u_57, mixed_28], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 128, 12), (3200, 12, 1), 0), buf640, out=buf641)
        buf642 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, x_flat_30, out_104, view_238, shift_57, out_105, x_103], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf639, buf641, buf360, buf642, 5120, 1024, stream=stream0)
        del buf639
        del buf641
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, x_flat_30, out_104, view_238, shift_57, out_105, x_103, x_104], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf643 = extern_kernels.convolution(buf642, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg136_1
        buf644 = buf643; del buf643  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, x_flat_30, out_104, view_238, shift_57, out_105, x_103, x_104, add_110, h_42], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf644, arg137_1, buf633, 5242880, stream=stream0)
        del arg137_1
        buf645 = buf634; del buf634  # reuse
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg138_1, buf645, 16384, 9, stream=stream0)
        del arg138_1
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf646 = extern_kernels.convolution(buf644, buf645, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf647 = buf637; del buf637  # reuse
        buf648 = buf636; del buf636  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf646, arg139_1, buf647, buf648, 1280, 4096, stream=stream0)
        buf650 = reinterpret_tensor(buf633, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf646, arg139_1, buf647, buf648, buf650, 5120, 1024, stream=stream0)
        del arg139_1
        buf651 = buf640; del buf640  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, v_97, transpose_163, x_flat_31, v_t_x_29], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf650, (40, 128, 1024), (131072, 1024, 1), 0), out=buf651)
        buf652 = reinterpret_tensor(buf646, (40, 128, 1024), (131072, 1024, 1), 0); del buf646  # reuse
        # Topologically Sorted Source Nodes: [split_29, u_59, mixed_29], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 128, 12), (3200, 12, 1), 0), buf651, out=buf652)
        buf653 = buf642; del buf642  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, x_flat_31, out_106, view_243, shift_59, out_107, x_106], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf650, buf652, buf360, buf653, 5120, 1024, stream=stream0)
        del buf650
        del buf652
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, x_flat_31, out_106, view_243, shift_59, out_107, x_106, x_107], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf654 = extern_kernels.convolution(buf653, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg140_1
        buf655 = buf654; del buf654  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, x_flat_31, out_106, view_243, shift_59, out_107, x_106, x_107, add_113, h_43], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf655, arg141_1, buf644, 5242880, stream=stream0)
        del arg141_1
        buf656 = buf645; del buf645  # reuse
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg142_1, buf656, 16384, 9, stream=stream0)
        del arg142_1
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf657 = extern_kernels.convolution(buf655, buf656, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf658 = buf648; del buf648  # reuse
        buf659 = buf647; del buf647  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf657, arg143_1, buf658, buf659, 1280, 4096, stream=stream0)
        buf661 = reinterpret_tensor(buf644, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf657, arg143_1, buf658, buf659, buf661, 5120, 1024, stream=stream0)
        del arg143_1
        buf662 = buf651; del buf651  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, v_99, transpose_164, x_flat_32, v_t_x_30], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf661, (40, 128, 1024), (131072, 1024, 1), 0), out=buf662)
        buf663 = reinterpret_tensor(buf657, (40, 128, 1024), (131072, 1024, 1), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [split_30, u_61, mixed_30], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 128, 12), (3200, 12, 1), 0), buf662, out=buf663)
        buf664 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, x_flat_32, out_108, view_248, shift_61, out_109, x_109], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf661, buf663, buf360, buf664, 5120, 1024, stream=stream0)
        del buf661
        del buf663
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, x_flat_32, out_108, view_248, shift_61, out_109, x_109, x_110], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf665 = extern_kernels.convolution(buf664, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf665, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg144_1
        buf666 = buf665; del buf665  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, x_flat_32, out_108, view_248, shift_61, out_109, x_109, x_110, add_116, h_44], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf666, arg145_1, buf655, 5242880, stream=stream0)
        del arg145_1
        buf667 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg146_1, buf667, 16384, 9, stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf668 = extern_kernels.convolution(buf666, buf667, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf667
        buf669 = buf659; del buf659  # reuse
        buf670 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf668, arg147_1, buf669, buf670, 1280, 4096, stream=stream0)
        buf672 = reinterpret_tensor(buf655, (40, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf655  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf668, arg147_1, buf669, buf670, buf672, 5120, 1024, stream=stream0)
        del arg147_1
        buf673 = buf662; del buf662  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, v_101, transpose_165, x_flat_33, v_t_x_31], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf672, (40, 128, 1024), (131072, 1024, 1), 0), out=buf673)
        buf674 = reinterpret_tensor(buf668, (40, 128, 1024), (131072, 1024, 1), 0); del buf668  # reuse
        # Topologically Sorted Source Nodes: [split_31, u_63, mixed_31], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (40, 128, 12), (3200, 12, 1), 0), buf673, out=buf674)
        del buf673
        buf675 = buf664; del buf664  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf672, buf674, buf360, buf675, 5120, 1024, stream=stream0)
        del buf360
        del buf672
        del buf674
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112, x_113], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf676 = extern_kernels.convolution(buf675, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (40, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg148_1
        del buf675
        buf677 = buf670; del buf670  # reuse
        buf678 = buf669; del buf669  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112, x_113, add_119, h_45, input_86], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_62.run(buf676, arg149_1, buf666, buf677, buf678, 1280, 4096, stream=stream0)
        buf680 = empty_strided_cuda((40, ), (1, ), torch.float32)
        buf681 = empty_strided_cuda((), (), torch.bool)
        # Topologically Sorted Source Nodes: [lt, zeros_like, full_like, sigma, gt, add_noise], Original ATen: [aten.lt, aten.zeros_like, aten.full_like, aten.where, aten.gt, aten.any]
        stream0 = get_raw_stream(0)
        triton_per_fused_any_full_like_gt_lt_where_zeros_like_64.run(arg0_1, buf680, buf681, 1, 40, stream=stream0)
        del arg0_1
        buf682 = buf336; del buf336  # reuse
        buf683 = buf682; del buf682  # reuse
        buf689 = buf676; del buf676  # reuse
        buf690 = buf689; del buf689  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42, x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112, x_113, add_119, h_45, input_86, input_43, input_87], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_65.run(buf683, buf690, arg149_1, buf326, buf337, buf338, arg150_1, arg151_1, buf666, buf677, buf678, 5242880, stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del buf326
        del buf337
        del buf338
        del buf666
        del buf677
        del buf678
        buf684 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf691 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44, input_87, input_88], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_66.run(arg152_1, buf684, buf691, 512, 9, stream=stream0)
        del arg152_1
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten.convolution]
        buf685 = extern_kernels.convolution(buf683, buf684, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf685, (40, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
        del buf683
        del buf684
        # Topologically Sorted Source Nodes: [input_87, input_88], Original ATen: [aten.silu, aten.convolution]
        buf692 = extern_kernels.convolution(buf690, buf691, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf692, (40, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
        del buf690
        del buf691
        buf687 = empty_strided_cuda((40, 1, 1, 1), (1, 40, 40, 40), torch.float32)
        buf694 = empty_strided_cuda((40, 1, 1, 1), (1, 40, 40, 40), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44, std_target, input_87, input_88, sub_1, v_pred, std_pred], Original ATen: [aten.silu, aten.convolution, aten.std, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_silu_std_sub_67.run(buf685, arg153_1, buf692, buf687, buf694, s57, 40, 4096, stream=stream0)
        buf697 = empty_strided_cuda((40, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44, std_target, input_87, input_88, sub_1, v_pred, std_pred, add_121, rescale_factor, v_pred_rescaled, mul_31, mul_32, v_guided, mul_33, x_next], Original ATen: [aten.silu, aten.convolution, aten.std, aten.sub, aten.add, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mul_silu_std_sub_68.run(buf692, arg153_1, buf685, buf687, buf694, arg25_1, buf697, s57, 40960, 4, stream=stream0)
        del arg153_1
        del arg25_1
        del buf685
        del buf687
        del buf692
        del buf694
    return (buf680, buf681, buf697, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1 = args
        args.clear()
        s57 = arg155_1
        partition0_args = [arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg154_1, arg150_1, arg151_1, arg152_1, arg153_1, s57]
        del arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg154_1, arg150_1, arg151_1, arg152_1, arg153_1
        (buf680, buf681, buf697) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf681, buf697, buf680, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.int64)
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
    arg23_1 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((40, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((4, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = 1
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
