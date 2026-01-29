# AOT ID: ['0_inference']
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



# kernel path: ./.compile_cache/lq/clqb67mcf2ca5q3n5xjeajnszhpthhe3ytfjwd22756reecp4cje.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h => convolution
# Graph fragment:
#   %arg25_1 : Tensor "f32[30, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %convolution : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 983040, 'x': 491520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 120
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
#   %convolution : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/pn/cpnl73pwaq3od6vvohw2se7e7dnshalo4cta65vnixf2le3py4lk.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h => convolution
# Graph fragment:
#   %buf2 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf2]
#   %arg24_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %convolution : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 47186432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
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
#   %convolution_1 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/yd/cydb6awhxkgk6wmcikzgagrvdiwsn4bxf37fcu26so4qg44uoxfp.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution_1
#   x_norm => var_mean, view
# Graph fragment:
#   %buf5 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %convolution_1 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [30, 32, 4, 1024]), kwargs = {})
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
    size_hints={'x': 1024, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15360, 'r0_': 15729152}}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: ./.compile_cache/if/cifvkj5nhco6wdx3jybxbh3yv3dcq4bv23pstscserv5dk5e4vgm.py
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
#   %arg0_1 : Tensor "f32[30][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %unsqueeze : Tensor "f32[30, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, -9.210340371976184), kwargs = {})
#   %div : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 128), kwargs = {})
#   %exp : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 0), kwargs = {})
#   %mul_1 : Tensor "f32[30, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %cos : Tensor "f32[30, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %sin : Tensor "f32[30, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %cat : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   return %cat
triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5 = async_compile.triton('triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 61440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
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


# kernel path: ./.compile_cache/vc/cvcvqklurjd4zpqv26pbjax534vgwwbhruchdw2ykrq24imsqist.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_19
#   input_2 => mul_2, sigmoid
# Graph fragment:
#   %mm_default_19 : Tensor "f32[30, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_default_19]
#   %arg2_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add_tensor_19 : Tensor "f32[30, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_19, %arg2_1), kwargs = {})
#   %sigmoid : Tensor "f32[30, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_19,), kwargs = {})
#   %mul_2 : Tensor "f32[30, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_19, %sigmoid), kwargs = {})
#   return %mul_2
triton_poi_fused_addmm_silu_6 = async_compile.triton('triton_poi_fused_addmm_silu_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 372736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/wx/cwxu7ueue76pds2tpaz44b24saebbwosm6wgb6skvbfuivgerrl2.py
# Topologically Sorted Source Nodes: [s_emb], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   s_emb => embedding
# Graph fragment:
#   %arg6_1 : Tensor "i64[30][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg5_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg6_1), kwargs = {})
#   return %embedding
triton_poi_fused_embedding_7 = async_compile.triton('triton_poi_fused_embedding_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
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


# kernel path: ./.compile_cache/al/caltnnnj7y2xkle4ogjacy6mnhlwcwyp6uo5zw4hdculzahbkmlr.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_4 => add_tensor_18
#   input_5 => mul_3, sigmoid_1
# Graph fragment:
#   %mm_default_18 : Tensor "f32[30, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_18]
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %add_tensor_18 : Tensor "f32[30, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %arg8_1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[30, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_18,), kwargs = {})
#   %mul_3 : Tensor "f32[30, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_18, %sigmoid_1), kwargs = {})
#   return %mul_3
triton_poi_fused_addmm_silu_8 = async_compile.triton('triton_poi_fused_addmm_silu_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 186368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/66/c66jvizj6galxh63utzb3trblcrdklkttphiwmdefkxe4pdghmbk.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_7 => add_tensor_17
#   input_8 => mul_4, sigmoid_2
# Graph fragment:
#   %mm_default_17 : Tensor "f32[30, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_17]
#   %arg12_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_tensor_17 : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_17, %arg12_1), kwargs = {})
#   %sigmoid_2 : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_17,), kwargs = {})
#   %mul_4 : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_17, %sigmoid_2), kwargs = {})
#   return %mul_4
triton_poi_fused_addmm_silu_9 = async_compile.triton('triton_poi_fused_addmm_silu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 93184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
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


# kernel path: ./.compile_cache/tc/ctcygpnc57bdkytglzwffp5krt3yl5wpi3gcaz2eatyhiwfup7m5.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution_1
#   x_norm => add, mul_7, rsqrt, sub, var_mean, view
# Graph fragment:
#   %buf5 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %getitem_1 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=buf7]
#   %convolution_1 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [30, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   return %mul_7
triton_poi_fused_convolution_native_group_norm_10 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 15729152, 'x': 31457280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 128
    y4 = (yindex % 128)
    y5 = yindex // 4
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 128*x3 + 131072*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 4096.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3 + 1024*y6), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/so/csoxj6d2kduipj4aahoemfwd7qdfa5hfseyenbud4p2sepyqspta.py
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
#   %mul_7 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0" = PlaceHolder[target=mul_7]
#   %bmm_1 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %addmm_5 : Tensor "f32[30, 3200][3200, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %convolution_1 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [30, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [30, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [30, 128, -1]), kwargs = {})
#   %add_1 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [30, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "f32[30, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [30, 128, 1, 1]), kwargs = {})
#   %add_2 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   return %mul_8
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 31472640, 'x': 31457280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (3072 + y0 + 3200*y1), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (y0 + 128*x2 + 131072*y1), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/ph/cphhz3thwwzy6ieypsmqy43ym7ih3dyhlbwbl6n6jtnn7o742vfw.py
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
#   %buf25 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf25]
#   %arg29_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %convolution : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution]
#   %convolution_1 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [30, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [30, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [30, 128, -1]), kwargs = {})
#   %add_1 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [30, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "f32[30, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [30, 128, 1, 1]), kwargs = {})
#   %add_2 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   %convolution_2 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_8, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution), kwargs = {})
#   %sigmoid_6 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_9 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_6), kwargs = {})
#   return %mul_9
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62915072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
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
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/5a/c5azuzuopcrikg25vx3dnwwya3uinwxwx2bmcpfdbo4d6nwb7pii.py
# Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   q => permute_12
#   q_1 => clone
#   view_10 => view_14
# Graph fragment:
#   %buf40 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_14 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_5, [30, 256, 256]), kwargs = {})
#   %permute_12 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_12,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_convolution_transpose_view_14 = async_compile.triton('triton_poi_fused_clone_convolution_transpose_view_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_transpose_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23593984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_transpose_view_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/ff/cffg25quutlsmby3nm72r73pvketvpadrcd2h5rzrwa5d2vtcbq2.py
# Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_1
#   q_1 => view_16
#   q_2 => permute_15
#   view_11 => view_17
# Graph fragment:
#   %mm : Tensor "f32[7680, 256][256, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_16 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [30, 256, 256]), kwargs = {})
#   %view_17 : Tensor "f32[30, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [30, 256, 4, 64]), kwargs = {})
#   %permute_15 : Tensor "f32[30, 4, 256, 64][65536, 64, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "f32[30, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_clone_transpose_view_15 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/gx/cgxelim5t73q267tky25mri5oyffm3acyzmixm3nrtbvlxrpkksu.py
# Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   k_1 => permute_16
#   kv => unsqueeze_2
#   matmul => clone_2
#   transpose_6 => permute_18
#   view_12 => view_18
# Graph fragment:
#   %mm_1 : Tensor "f32[30, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "f32[30, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_18 : Tensor "f32[30, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_10, [30, 1, 4, 64]), kwargs = {})
#   %permute_16 : Tensor "f32[30, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %permute_18 : Tensor "f32[30, 4, 64, 1][512, 64, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_16, [0, 1, 3, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[30, 4, 64, 1][256, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused_clone_split_transpose_unsqueeze_view_16 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 92160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/il/cilfs2xjivznbwjsyxbpjqy2jpzvppb7vyhjlmcbevema4ofadg4.py
# Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div_1, exp_1, sum_1
#   matmul => view_22
# Graph fragment:
#   %bmm_4 : Tensor "f32[120, 256, 1][256, 1, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %view_22 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_4, [30, 4, 256, 1]), kwargs = {})
#   %mul_tensor_18 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, 1), kwargs = {})
#   %amax_default_9 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_18, [-1], True), kwargs = {})
#   %sub_tensor_9 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_18, %amax_default_9), kwargs = {})
#   %mul_tensor_19 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_9, 0.125), kwargs = {})
#   %exp_1 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_19,), kwargs = {})
#   %sum_1 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[30, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_1), kwargs = {})
#   return %expand_2
triton_poi_fused__softmax_amax_mul_sub_view_17 = async_compile.triton('triton_poi_fused__softmax_amax_mul_sub_view_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_amax_mul_sub_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 368640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_amax_mul_sub_view_17(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 - tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.exp(tmp5)
    tmp7 = (tmp6 / tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/w3/cw3j2nyvotlwztiw5k7uuxq2tkvbgyamfuem5gfszohw3l5onasg.py
# Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   kv => unsqueeze_2
#   matmul_1 => clone_3
#   v_5 => permute_17
#   view_13 => view_19
# Graph fragment:
#   %mm_1 : Tensor "f32[30, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "f32[30, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_19 : Tensor "f32[30, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_11, [30, 1, 4, 64]), kwargs = {})
#   %permute_17 : Tensor "f32[30, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[30, 4, 1, 64][256, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_split_transpose_unsqueeze_view_18 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 92160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/p6/cp6rzyucwiss3udj47bbbdacwlbyml3xq6owdrcshffvfet7bmiw.py
# Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => view_25
#   out_4 => clone_4
#   transpose_7 => permute_19
# Graph fragment:
#   %bmm_5 : Tensor "f32[120, 256, 64][16384, 64, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %view_25 : Tensor "f32[30, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [30, 4, 256, 64]), kwargs = {})
#   %permute_19 : Tensor "f32[30, 256, 4, 64][65536, 64, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_25, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[30, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_transpose_view_19 = async_compile.triton('triton_poi_fused_clone_transpose_view_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/cg/ccgfzx5nqu3kkhbozdskm3lootd573g2qarw4rqw5y5dapya3anb.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_16, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
# Graph fragment:
#   %buf40 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %mm_default_16 : Tensor "f32[7680, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_16]
#   %arg39_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   return %add_8
triton_poi_fused_add_addmm_convolution_transpose_view_20 = async_compile.triton('triton_poi_fused_add_addmm_convolution_transpose_view_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_transpose_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31459328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_transpose_view_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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
#   input_16 => add_tensor_16, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
# Graph fragment:
#   %arg40_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg40_1]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/pq/cpqoi4dwsdk5bbtlwydk7aspli6vtoimcabtxj2vn5b56jodcber.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_16, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
#   x_norm_2 => var_mean_2, view_30
# Graph fragment:
#   %buf53 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf53]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [30, 32, 8, 256]), kwargs = {})
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
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15360, 'r0_': 7865344}}
)
@triton.jit
def triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: ./.compile_cache/i3/ci3jjxhzah3p32fxlgepn67ffvfs3mujlg3g4rnwkucrijakcgwi.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_16, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
#   x_norm_2 => add_9, mul_14, rsqrt_2, sub_3, var_mean_2, view_30
# Graph fragment:
#   %buf53 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf53]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %getitem_13 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf55 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=buf55]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   return %mul_14
triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 7865344, 'x': 15728640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 256
    y4 = (yindex % 256)
    y5 = yindex // 8
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + 256*x3 + 65536*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 2048.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3 + 256*y6), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/65/c65724bhxh65frjodqwq3gxl2kul5wjqf3wqavmrltuomgd3s7ff.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_16, view_28
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
#   %mul_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0" = PlaceHolder[target=mul_14]
#   %bmm_7 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %addmm_7 : Tensor "f32[30, 6400][6400, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [30, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [30, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [30, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [30, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   return %mul_15
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 15759360, 'x': 15728640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (6144 + y0 + 6400*y1), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (y0 + 256*x2 + 65536*y1), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/mc/cmcwf662f7gugxsv62kf4byy336vmjh622cvgvkqwn5tfc654kxe.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   input_16 => add_tensor_16, view_28
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
#   %buf65 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf65]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf40 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [30, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [30, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [30, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [30, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %view_37 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_16, [30, 256, 256]), kwargs = {})
#   %permute_23 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_37, [0, 2, 1]), kwargs = {})
#   %clone_6 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_6
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31459328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/ib/cibphyldljckfazchlufb526xz4yjar3ydr2uqzanghllky2iaq2.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   input_16 => add_tensor_16, view_28
#   input_18 => add_tensor_14, view_51
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
#   %buf65 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf65]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf40 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %mm_default_14 : Tensor "f32[7680, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_14]
#   %arg47_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [30, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [30, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [30, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [30, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %add_tensor_14 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_14, %arg47_1), kwargs = {})
#   %view_51 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_14, [30, 256, 256]), kwargs = {})
#   %permute_32 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_51, [0, 2, 1]), kwargs = {})
#   %view_52 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_32, [30, 256, 16, 16]), kwargs = {})
#   %add_13 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %view_52), kwargs = {})
#   return %add_13
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 39324672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/tz/ctzzy4pldph72wphhx3ryutr2rjtknenrfuho4l6kcqqqykxtgyy.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13, add_13, h_5], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_13 => add_17
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   h_5 => mul_20, sigmoid_12
#   input_16 => add_tensor_16, view_28
#   input_18 => add_tensor_14, view_51
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
#   %buf87 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf87]
#   %arg51_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %buf65 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf65]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf40 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf40]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %add_17 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_17]
#   %convolution_5 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [30, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [30, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [30, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [30, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [30, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [30, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %add_tensor_14 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_14, %arg47_1), kwargs = {})
#   %view_51 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_14, [30, 256, 256]), kwargs = {})
#   %permute_32 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_51, [0, 2, 1]), kwargs = {})
#   %view_52 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_32, [30, 256, 16, 16]), kwargs = {})
#   %add_13 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %view_52), kwargs = {})
#   %convolution_8 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_13, %arg48_1, %arg49_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_53 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_8, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_3 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_5 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_53, %getitem_20), kwargs = {})
#   %add_14 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_19, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_18 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %view_54 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_18, [30, 256, 16, 16]), kwargs = {})
#   %view_58 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_54, [30, 256, -1]), kwargs = {})
#   %add_15 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_58, %bmm_11), kwargs = {})
#   %view_59 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_15, [30, 256, 16, 16]), kwargs = {})
#   %view_57 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_23, [30, 256, 1, 1]), kwargs = {})
#   %add_16 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_59, %view_57), kwargs = {})
#   %sigmoid_11 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_16,), kwargs = {})
#   %mul_19 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sigmoid_11), kwargs = {})
#   %convolution_9 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_19, %arg50_1, %arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_17 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %mul_16), kwargs = {})
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   return %add_17,%mul_20
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 39324672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/pq/cpqx6so4vp6qdb53zsf6qwn244uzmopxosnvpdkqzqamqu4cdbbs.py
# Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   q_6 => permute_34
#   q_7 => clone_12
#   view_30 => view_60
# Graph fragment:
#   %buf91 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_60 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [30, 512, 64]), kwargs = {})
#   %permute_34 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_60, [0, 2, 1]), kwargs = {})
#   %clone_12 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_12
triton_poi_fused_clone_convolution_silu_transpose_view_29 = async_compile.triton('triton_poi_fused_clone_convolution_silu_transpose_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_silu_transpose_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11798528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_silu_transpose_view_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: ./.compile_cache/rj/crj3acc4j7eiwde3naiwogza5mvvv4r5brf5fuj3d7aqxnsnfzqk.py
# Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_4 => clone_13
#   q_7 => view_62
#   q_8 => permute_37
#   view_31 => view_63
# Graph fragment:
#   %mm_4 : Tensor "f32[1920, 512][512, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %view_62 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [30, 64, 512]), kwargs = {})
#   %view_63 : Tensor "f32[30, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_62, [30, 64, 8, 64]), kwargs = {})
#   %permute_37 : Tensor "f32[30, 8, 64, 64][32768, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_63, [0, 2, 1, 3]), kwargs = {})
#   %clone_13 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_13
triton_poi_fused__unsafe_view_clone_transpose_view_30 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: ./.compile_cache/rh/crhl46jxnoxtimrtqivmzfuqqm45grmhdu2ziu5ecmblycfwqkyk.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   k_5 => permute_38
#   kv_2 => unsqueeze_4
#   matmul_4 => clone_14
#   transpose_22 => permute_40
#   view_32 => view_64
# Graph fragment:
#   %mm_5 : Tensor "f32[30, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "f32[30, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_64 : Tensor "f32[30, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_24, [30, 1, 8, 64]), kwargs = {})
#   %permute_38 : Tensor "f32[30, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_64, [0, 2, 1, 3]), kwargs = {})
#   %permute_40 : Tensor "f32[30, 8, 64, 1][1024, 64, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_38, [0, 1, 3, 2]), kwargs = {})
#   %clone_14 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_14
triton_poi_fused_clone_split_transpose_unsqueeze_view_31 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 184320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/if/cifbwyexykbx2rdzqqotbe74ssb3q4r5mbxectsqs44pkd3n3v3u.py
# Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_5 => div_3, exp_3, sum_3
#   matmul_4 => view_68
# Graph fragment:
#   %bmm_12 : Tensor "f32[240, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_12]
#   %view_68 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_12, [30, 8, 64, 1]), kwargs = {})
#   %mul_tensor_14 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_68, 1), kwargs = {})
#   %amax_default_7 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_14, [-1], True), kwargs = {})
#   %sub_tensor_7 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_14, %amax_default_7), kwargs = {})
#   %mul_tensor_15 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_7, 0.125), kwargs = {})
#   %exp_3 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_15,), kwargs = {})
#   %sum_3 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [-1], True), kwargs = {})
#   %div_3 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_3, %sum_3), kwargs = {})
#   return %expand_10
triton_poi_fused__softmax_amax_mul_sub_view_32 = async_compile.triton('triton_poi_fused__softmax_amax_mul_sub_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_amax_mul_sub_view_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 184320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_amax_mul_sub_view_32(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 - tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.exp(tmp5)
    tmp7 = (tmp6 / tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/7i/c7iu7xko5sswlmc6p3ejx5ywwkkyqurc5eq5rynnnpozad7tiy3z.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   kv_2 => unsqueeze_4
#   matmul_5 => clone_15
#   v_13 => permute_39
#   view_33 => view_65
# Graph fragment:
#   %mm_5 : Tensor "f32[30, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "f32[30, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_65 : Tensor "f32[30, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_25, [30, 1, 8, 64]), kwargs = {})
#   %permute_39 : Tensor "f32[30, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_65, [0, 2, 1, 3]), kwargs = {})
#   %clone_15 : Tensor "f32[30, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_15
triton_poi_fused_clone_split_transpose_unsqueeze_view_33 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 184320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/ui/cuialaeiggtujr47ap4heffy7appcsjnszltj4y6iqggvyh2bgrr.py
# Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_5 => view_71
#   out_12 => clone_16
#   transpose_23 => permute_41
# Graph fragment:
#   %bmm_13 : Tensor "f32[240, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %view_71 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_13, [30, 8, 64, 64]), kwargs = {})
#   %permute_41 : Tensor "f32[30, 64, 8, 64][32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_71, [0, 2, 1, 3]), kwargs = {})
#   %clone_16 : Tensor "f32[30, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_16
triton_poi_fused_clone_transpose_view_34 = async_compile.triton('triton_poi_fused_clone_transpose_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: ./.compile_cache/ql/cql5nekgl6dpml46mkooxcarn7bai62ipk3kqmfsfzlzd2lyj3yf.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_13, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
# Graph fragment:
#   %buf91 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %mm_default_13 : Tensor "f32[1920, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_13]
#   %arg57_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   return %add_18
triton_poi_fused_add_addmm_convolution_silu_transpose_view_35 = async_compile.triton('triton_poi_fused_add_addmm_convolution_silu_transpose_view_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_silu_transpose_view_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15732736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_silu_transpose_view_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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
#   input_20 => add_tensor_13, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
# Graph fragment:
#   %arg58_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg58_1]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/bk/cbkln6vypm7djbvpqfbzsvuvv72yyywd5om4523ntbiuf4dof3w2.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_13, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
#   x_norm_4 => var_mean_4, view_76
# Graph fragment:
#   %buf104 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf104]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [30, 32, 16, 64]), kwargs = {})
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
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15360, 'r0_': 3934208}}
)
@triton.jit
def triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: ./.compile_cache/ap/capog3kefvgwh4ivafuw4puahfl33a2ja5kjtpvf4woglhjnruzq.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_13, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
#   x_norm_4 => add_19, mul_22, rsqrt_4, sub_7, var_mean_4, view_76
# Graph fragment:
#   %buf104 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf104]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %getitem_27 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf106 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=buf106]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [30, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   return %mul_22
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3934208, 'x': 7864320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
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


# kernel path: ./.compile_cache/5a/c5aezsq2qmcc4q7e3dmcg7jiiefyqj3aeqdvh5ix2zriexahhdjs.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_13, view_74
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
#   %mul_22 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0" = PlaceHolder[target=mul_22]
#   %bmm_15 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0" = PlaceHolder[target=bmm_15]
#   %addmm_9 : Tensor "f32[30, 12800][12800, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [30, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [30, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [30, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [30, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[30, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [30, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   return %mul_23
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 7925760, 'x': 7864320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
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


# kernel path: ./.compile_cache/cc/cccd3swjv7upgpi7npdns2zbijy7cpsuzc6jnmohne2ii4aps5i2.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   input_20 => add_tensor_13, view_74
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
#   %buf116 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf116]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf91 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [30, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [30, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [30, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [30, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[30, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [30, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %view_83 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_24, [30, 512, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_83, [0, 2, 1]), kwargs = {})
#   %clone_18 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_45,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_18
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15732736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: ./.compile_cache/tq/ctqlp67yoqjo2fqbysdmn2kazrr6ydnfa6ea4eoqk22zz4unfz4e.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   input_20 => add_tensor_13, view_74
#   input_22 => add_tensor_11, view_97
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
#   %buf116 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf116]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf91 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %mm_default_11 : Tensor "f32[1920, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_11]
#   %arg65_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [30, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [30, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [30, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [30, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[30, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [30, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %add_tensor_11 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_11, %arg65_1), kwargs = {})
#   %view_97 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_11, [30, 64, 512]), kwargs = {})
#   %permute_54 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_97, [0, 2, 1]), kwargs = {})
#   %view_98 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_54, [30, 512, 8, 8]), kwargs = {})
#   %add_23 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %view_98), kwargs = {})
#   return %add_23
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 19666944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: ./.compile_cache/dh/cdh22xmlfclwblvlbyxaapvvbzhwiw5gv2xsdfhbgadmgd2vgxkx.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_17 => add_22
#   add_21 => add_27
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   h_8 => mul_28, sigmoid_16
#   input_20 => add_tensor_13, view_74
#   input_22 => add_tensor_11, view_97
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
#   %buf137 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf137]
#   %arg69_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %buf116 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf116]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf91 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf91]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %add_27 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %getitem_39 : Tensor "f32[30, 64, 1][64, 1, 1920]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf140 : Tensor "f32[30, 64, 1][64, 1, 1920]cuda:0" = PlaceHolder[target=buf140]
#   %arg70_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg70_1]
#   %arg71_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg71_1]
#   %sigmoid_12 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_13 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [30, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [30, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [30, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [30, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [30, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [30, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[30, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [30, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %add_tensor_11 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_11, %arg65_1), kwargs = {})
#   %view_97 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_11, [30, 64, 512]), kwargs = {})
#   %permute_54 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_97, [0, 2, 1]), kwargs = {})
#   %view_98 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_54, [30, 512, 8, 8]), kwargs = {})
#   %add_23 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %view_98), kwargs = {})
#   %convolution_13 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_23, %arg66_1, %arg67_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_99 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_13, [30, 32, 16, 64]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_99, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_5 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_9 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_99, %getitem_34), kwargs = {})
#   %add_24 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_33, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %mul_26 : Tensor "f32[30, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_5), kwargs = {})
#   %view_100 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_26, [30, 512, 8, 8]), kwargs = {})
#   %view_104 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_100, [30, 512, -1]), kwargs = {})
#   %add_25 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_104, %bmm_19), kwargs = {})
#   %view_105 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_25, [30, 512, 8, 8]), kwargs = {})
#   %view_103 : Tensor "f32[30, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_37, [30, 512, 1, 1]), kwargs = {})
#   %add_26 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_105, %view_103), kwargs = {})
#   %sigmoid_15 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_26,), kwargs = {})
#   %mul_27 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %sigmoid_15), kwargs = {})
#   %convolution_14 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_27, %arg68_1, %arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_27 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %mul_24), kwargs = {})
#   %sigmoid_16 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [30, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %clone_24 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_56,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_24, %getitem_39), kwargs = {})
#   %add_28 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_29 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_6), kwargs = {})
#   %mul_30 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %arg70_1), kwargs = {})
#   %add_29 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %arg71_1), kwargs = {})
#   return %add_27,%getitem_39,%buf140,%add_29
triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42 = async_compile.triton('triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 27535360}}
)
@triton.jit
def triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1920
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


# kernel path: ./.compile_cache/k5/ck5psx22ttkjep5jpzzbmcrdjyc4yecn7pfmlq4ql4zskdo7u2fk.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   matmul_8 => clone_25
#   q_12 => select
#   qkv => add_tensor_10, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
# Graph fragment:
#   %mm_default_10 : Tensor "f32[1920, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_10]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_10 : Tensor "f32[1920, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[30, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [30, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[30, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [30, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 30, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select : Tensor "f32[30, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 0), kwargs = {})
#   %clone_25 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_16,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_25
triton_poi_fused_addmm_clone_permute_select_view_43 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11798528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_view_43(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: ./.compile_cache/ke/ckerhaaybwbkjyl5bixvigrlmoh6e45yjvkzomq7frjrckzm6zkv.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   k_8 => select_1
#   matmul_8 => clone_26
#   qkv => add_tensor_10, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   transpose_35 => permute_59
# Graph fragment:
#   %mm_default_10 : Tensor "f32[1920, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_10]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_10 : Tensor "f32[1920, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[30, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [30, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[30, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [30, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 30, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_1 : Tensor "f32[30, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 1), kwargs = {})
#   %permute_59 : Tensor "f32[30, 8, 64, 64][98304, 64, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_26 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_17,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_26
triton_poi_fused_addmm_clone_permute_select_transpose_view_44 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_transpose_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_transpose_view_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3934208, 'x': 7864320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_transpose_view_44(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
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


# kernel path: ./.compile_cache/wk/cwkd32jyjklfuzczhskyonibvnyg2bhou5bjy462c3mobfjy5jhw.py
# Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_9 => div_5, exp_5, sum_5
#   matmul_8 => view_112
# Graph fragment:
#   %bmm_20 : Tensor "f32[240, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %amax_default_5 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 15360]cuda:0" = PlaceHolder[target=amax_default_5]
#   %sum_5 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 15360]cuda:0" = PlaceHolder[target=sum_5]
#   %view_112 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_20, [30, 8, 64, 64]), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_112, 1), kwargs = {})
#   %amax_default_5 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_10, [-1], True), kwargs = {})
#   %sub_tensor_5 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_10, %amax_default_5), kwargs = {})
#   %mul_tensor_11 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_5, 0.125), kwargs = {})
#   %exp_5 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_11,), kwargs = {})
#   %sum_5 : Tensor "f32[30, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1], True), kwargs = {})
#   %div_5 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_5), kwargs = {})
#   return %amax_default_5,%sum_5,%expand_18
triton_per_fused__softmax_amax_mul_sub_view_45 = async_compile.triton('triton_per_fused__softmax_amax_mul_sub_view_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_amax_mul_sub_view_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 11796480}}
)
@triton.jit
def triton_per_fused__softmax_amax_mul_sub_view_45(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 15360
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tmp2 - tmp6
    tmp8 = 0.125
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = (tmp10 / tmp14)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/ak/cakkdxooobn3ogruckfajfwc24zcy7vzvf7ov7hpiebnj7zyrpuu.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   out_20 => clone_27
#   qkv => add_tensor_10, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   v_20 => select_2
# Graph fragment:
#   %mm_default_10 : Tensor "f32[1920, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_10]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_10 : Tensor "f32[1920, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[30, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [30, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[30, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [30, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 30, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_2 : Tensor "f32[30, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 2), kwargs = {})
#   %clone_27 : Tensor "f32[30, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_27
triton_poi_fused_addmm_clone_permute_select_view_46 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_view_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11798528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_view_46(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: ./.compile_cache/xb/cxbrmmbs3dfav2ytbedh43ymfvc2me5lkk5dif3ahyovw6c5kbqu.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_8 => mul_28, sigmoid_16
#   out_22 => add_tensor_9, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %add_27 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %mm_default_9 : Tensor "f32[1920, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_9]
#   %arg75_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %getitem_41 : Tensor "f32[30, 64, 1][64, 1, 1920]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf155 : Tensor "f32[30, 64, 1][64, 1, 1920]cuda:0" = PlaceHolder[target=buf155]
#   %arg76_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg76_1]
#   %arg77_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %sigmoid_16 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [30, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_9 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_9, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_9, [30, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   return %getitem_41,%buf155,%add_32
triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 15734784}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1920
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
#   out_22 => add_tensor_9, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %arg78_1 : Tensor "f32[512, 256, 4, 4][4096, 16, 4, 1]cuda:0" = PlaceHolder[target=arg78_1]
#   %sigmoid_16 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [30, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_9 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_9, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_9, [30, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [30, 512, 8, 8]), kwargs = {})
#   %convolution_15 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_119, %arg78_1, %arg79_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/2a/c2ayxzel6cu2avitrnswbzy5yrac5olej2xbcubyv7glftqrbkky.py
# Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_24 => add_tensor_8
#   input_25 => mul_34, sigmoid_17
#   input_26 => constant_pad_nd_default
# Graph fragment:
#   %mul_34 : Tensor "f32[30, 256][256, 1]cuda:0" = PlaceHolder[target=mul_34]
#   %add_tensor_8 : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_8, %arg81_1), kwargs = {})
#   %sigmoid_17 : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_8,), kwargs = {})
#   %mul_34 : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_8, %sigmoid_17), kwargs = {})
#   %constant_pad_nd_default : Tensor "f32[32, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_34, [0, 0, 0, 2]), kwargs = {})
#   return %constant_pad_nd_default
triton_poi_fused_addmm_silu_49 = async_compile.triton('triton_poi_fused_addmm_silu_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 30, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/oo/cooh7tzn7tshgpyclccwypo34fq7gmiuzzxj3kx62igtod667ski.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   gate => unsqueeze_7
#   h_10 => add_33
#   h_16_gated => mul_35
#   h_8 => mul_28, sigmoid_16
#   h_9 => convolution_15
#   input_26 => add_tensor_7, slice_tensor
#   input_27 => sigmoid_18
#   out_22 => add_tensor_9, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   unsqueeze_4 => unsqueeze_6
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %buf160 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf160]
#   %arg79_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %mul_20 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_20]
#   %mm_default_7 : Tensor "f32[32, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_7]
#   %arg83_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %sigmoid_16 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[30, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[30, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [30, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_9 : Tensor "f32[1920, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_9, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_9, [30, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[30, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[30, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[30, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[30, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[30, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [30, 512, 8, 8]), kwargs = {})
#   %convolution_15 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_119, %arg78_1, %arg79_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_tensor_7 : Tensor "f32[32, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_7, %arg83_1), kwargs = {})
#   %slice_tensor : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_tensor_7, 0, 0, -2), kwargs = {})
#   %sigmoid_18 : Tensor "f32[30, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%slice_tensor,), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[30, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_18, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[30, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %mul_35 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_7), kwargs = {})
#   %add_33 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %mul_35), kwargs = {})
#   return %add_33
triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50 = async_compile.triton('triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 15728640, 'x': 15761408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 256
    y0 = (yindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + 256*y1), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp2 + tmp8
    tl.store(out_ptr0 + (y0 + 256*x2 + 65536*y1), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/wx/cwxxzmgy53l5oash2wfaqzhnpvgo3nbmluporld5im52hnmrt4fb.py
# Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
# Source node to ATen node mapping:
#   q_13 => permute_65
#   q_14 => mm_8, permute_66, view_121
#   view_52 => view_120
# Graph fragment:
#   %add_33 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %view_120 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_33, [30, 256, 256]), kwargs = {})
#   %permute_65 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_120, [0, 2, 1]), kwargs = {})
#   %view_121 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_65, [7680, 256]), kwargs = {})
#   %permute_66 : Tensor "f32[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg84_1, [1, 0]), kwargs = {})
#   %mm_8 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_121, %permute_66), kwargs = {})
#   return %buf166
triton_poi_fused_mm_t_transpose_view_51 = async_compile.triton('triton_poi_fused_mm_t_transpose_view_51', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_t_transpose_view_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_t_transpose_view_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 7680)
    x1 = xindex // 7680
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256*x1 + 65536*(x0 // 256) + ((x0 % 256))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/s2/cs2fpvmjj7o4vwm7esrwqny47ttc4d2zk6nlktnhtbgwxmnnhvfc.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   input_28 => add_tensor_6, view_134
#   out_27 => view_135
#   transpose_44 => permute_74
#   x_22 => add_34
# Graph fragment:
#   %add_33 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_default_6 : Tensor "f32[7680, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_6]
#   %arg87_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %add_tensor_6 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_6, [30, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [30, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   return %add_34
triton_poi_fused_add_addmm_transpose_view_52 = async_compile.triton('triton_poi_fused_add_addmm_transpose_view_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_transpose_view_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 7864320, 'x': 23593984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_transpose_view_52(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 65536*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/7l/c7l5epjrzbwlgcruuix5hw3imly7enqhv6drluxnw4h6376eb4gs.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   h_11 => mul_39, sigmoid_20
#   input_28 => add_tensor_6, view_134
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
#   %buf187 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf187]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %add_tensor_6 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_6, [30, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [30, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [30, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [30, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [30, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [30, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %view_143 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [30, 256, 256]), kwargs = {})
#   %permute_76 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_143, [0, 2, 1]), kwargs = {})
#   %clone_36 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_76,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_36
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 7864320, 'x': 23593984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 256*x2 + 65536*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2 + 256*y3), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/mr/cmryo4cmdxbp3xaqmo2afxsplakya5flzjdeowlidvrf2dqcdwbv.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   h_11 => mul_39, sigmoid_20
#   input_28 => add_tensor_6, view_134
#   input_30 => add_tensor_5, view_157
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
#   %buf187 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf187]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_default_5 : Tensor "f32[7680, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_5]
#   %arg95_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg95_1]
#   %add_tensor_6 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_6, [30, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [30, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [30, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [30, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [30, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [30, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %add_tensor_5 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_5, %arg95_1), kwargs = {})
#   %view_157 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_5, [30, 256, 256]), kwargs = {})
#   %permute_85 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1]), kwargs = {})
#   %view_158 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_85, [30, 256, 16, 16]), kwargs = {})
#   %add_39 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %view_158), kwargs = {})
#   return %add_39
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 7864320, 'x': 31459328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 256*x2 + 65536*y1), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/2l/c2ltgogomch773chhpibfclkuh6avggnh542hqvr2vbge7lupvmh.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   add_31 => add_43
#   h_11 => mul_39, sigmoid_20
#   h_12 => mul_43, sigmoid_22
#   input_28 => add_tensor_6, view_134
#   input_30 => add_tensor_5, view_157
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
#   %buf209 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf209]
#   %arg99_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg99_1]
#   %buf187 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf187]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %add_43 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %add_tensor_6 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_6, [30, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [30, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [30, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [30, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [30, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [30, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %add_tensor_5 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_5, %arg95_1), kwargs = {})
#   %view_157 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_5, [30, 256, 256]), kwargs = {})
#   %permute_85 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1]), kwargs = {})
#   %view_158 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_85, [30, 256, 16, 16]), kwargs = {})
#   %add_39 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %view_158), kwargs = {})
#   %convolution_18 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_39, %arg96_1, %arg97_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_159 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_18, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_159, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_7 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_16 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_159, %getitem_52), kwargs = {})
#   %add_40 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_51, 1e-06), kwargs = {})
#   %rsqrt_9 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
#   %mul_41 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %rsqrt_9), kwargs = {})
#   %view_160 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_41, [30, 256, 16, 16]), kwargs = {})
#   %view_164 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [30, 256, -1]), kwargs = {})
#   %add_41 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_164, %bmm_29), kwargs = {})
#   %view_165 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_41, [30, 256, 16, 16]), kwargs = {})
#   %view_163 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_55, [30, 256, 1, 1]), kwargs = {})
#   %add_42 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_165, %view_163), kwargs = {})
#   %sigmoid_21 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_42,), kwargs = {})
#   %mul_42 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %sigmoid_21), kwargs = {})
#   %convolution_19 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_42, %arg98_1, %arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_43 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %mul_39), kwargs = {})
#   %sigmoid_22 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %view_166 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_43, [30, 256, 256]), kwargs = {})
#   %permute_87 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_166, [0, 2, 1]), kwargs = {})
#   %clone_42 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_87,), kwargs = {memory_format: torch.contiguous_format})
#   return %add_43,%clone_42
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 47187968}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/wf/cwf3nwyjkd76prri4frsvz6tqqh2ym445mgk7bngai3jwr7ncxyt.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_12 => mul_43, sigmoid_22
#   input_32 => add_tensor_4, view_180
#   out_35 => view_181
#   transpose_60 => permute_96
#   x_30 => add_44
# Graph fragment:
#   %add_43 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %mm_default_4 : Tensor "f32[7680, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_4]
#   %arg103_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg103_1]
#   %sigmoid_22 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_4 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [30, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [30, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   return %add_44
triton_poi_fused_add_addmm_silu_transpose_view_56 = async_compile.triton('triton_poi_fused_add_addmm_silu_transpose_view_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_silu_transpose_view_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31458304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_silu_transpose_view_56(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/ha/chaodbl3vn5d6ncmqcyafglcdjhcj7df2dbbn4elohh75devwuzk.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_35 => add_48
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   input_32 => add_tensor_4, view_180
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
#   %buf232 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf232]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %sigmoid_22 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_4 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [30, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [30, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [30, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [30, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [30, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [30, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %view_189 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_47, [30, 256, 256]), kwargs = {})
#   %permute_98 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_189, [0, 2, 1]), kwargs = {})
#   %clone_48 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_98,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_48
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31458304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/4r/c4rimstyjjrjgemecywpi4fktc5ton6qk7xrbsvtwtvlhwhupdf5.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_35 => add_48
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   input_32 => add_tensor_4, view_180
#   input_34 => add_tensor_3, view_203
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
#   %buf232 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf232]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %mm_default_3 : Tensor "f32[7680, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_3]
#   %arg111_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg111_1]
#   %sigmoid_22 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_4 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [30, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [30, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [30, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [30, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [30, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [30, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %add_tensor_3 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_3, %arg111_1), kwargs = {})
#   %view_203 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_3, [30, 256, 256]), kwargs = {})
#   %permute_107 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_203, [0, 2, 1]), kwargs = {})
#   %view_204 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_107, [30, 256, 16, 16]), kwargs = {})
#   %add_49 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %view_204), kwargs = {})
#   return %add_49
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 39323648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/r6/cr63grfbkbcpaylvccyifrhdgmpnggf2auuh6yhr4fenoi4yvkrd.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_35 => add_48
#   add_39 => add_53
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   h_14 => mul_51, sigmoid_26
#   input_32 => add_tensor_4, view_180
#   input_34 => add_tensor_3, view_203
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
#   %buf254 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf254]
#   %arg115_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg115_1]
#   %buf232 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf232]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %add_53 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_53]
#   %sigmoid_22 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_4 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_4, [30, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [30, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [30, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [30, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [30, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [30, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %add_tensor_3 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_3, %arg111_1), kwargs = {})
#   %view_203 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_3, [30, 256, 256]), kwargs = {})
#   %permute_107 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_203, [0, 2, 1]), kwargs = {})
#   %view_204 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_107, [30, 256, 16, 16]), kwargs = {})
#   %add_49 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %view_204), kwargs = {})
#   %convolution_22 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_49, %arg112_1, %arg113_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_205 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_22, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_205, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_9 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_20 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_205, %getitem_66), kwargs = {})
#   %add_50 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_65, 1e-06), kwargs = {})
#   %rsqrt_11 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_50,), kwargs = {})
#   %mul_49 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_11), kwargs = {})
#   %view_206 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_49, [30, 256, 16, 16]), kwargs = {})
#   %view_210 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_206, [30, 256, -1]), kwargs = {})
#   %add_51 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_210, %bmm_37), kwargs = {})
#   %view_211 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_51, [30, 256, 16, 16]), kwargs = {})
#   %view_209 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_69, [30, 256, 1, 1]), kwargs = {})
#   %add_52 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_211, %view_209), kwargs = {})
#   %sigmoid_25 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_52,), kwargs = {})
#   %mul_50 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_52, %sigmoid_25), kwargs = {})
#   %convolution_23 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_50, %arg114_1, %arg115_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_53 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_23, %mul_47), kwargs = {})
#   %sigmoid_26 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %view_212 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_51, [30, 256, 256]), kwargs = {})
#   %permute_109 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_212, [0, 2, 1]), kwargs = {})
#   %clone_54 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_109,), kwargs = {memory_format: torch.contiguous_format})
#   return %add_53,%clone_54
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 55052288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/cb/ccbvsjze3es4errmf4wyjfdff22ztnodohi3nffklhhwiis6jukp.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_43 => add_58
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   input_36 => add_tensor_2, view_226
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
#   %buf276 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf276]
#   %arg123_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg123_1]
#   %add_53 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_53]
#   %sigmoid_26 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_2 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [30, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [30, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [30, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [30, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [30, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [30, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   return %mul_55
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 31458304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
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


# kernel path: ./.compile_cache/au/caumf2oiqpqbz7d2o7qavjb3j7sp2ermfnabmxomvzhj25z6tpwj.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_43 => add_58
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   h_16 => convolution_26
#   input_36 => add_tensor_2, view_226
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
#   %sigmoid_26 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_2 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [30, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [30, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [30, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [30, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [30, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [30, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   %convolution_26 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_55, %arg124_1, %arg125_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf278
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_61 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_61', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4194304, 'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_61(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: ./.compile_cache/kw/ckw3b6lchk32exhh2hapc3c7zxmm5zfzl4xe7aw3hq2m4kyn7vay.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   add_43 => add_58
#   gate_1 => unsqueeze_14
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   h_16 => convolution_26
#   h_17 => add_59
#   h_32_gated => mul_57
#   input_36 => add_tensor_2, view_226
#   input_40 => add_tensor
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
#   %buf279 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf279]
#   %arg125_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %mul_12 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_12]
#   %mm_default : Tensor "f32[30, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg129_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg129_1]
#   %sigmoid_26 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_2 : Tensor "f32[7680, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [30, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[30, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[30, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [30, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [30, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[30, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [30, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [30, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[30, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [30, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[30, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [30, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[30, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   %convolution_26 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_55, %arg124_1, %arg125_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_tensor : Tensor "f32[30, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg129_1), kwargs = {})
#   %sigmoid_30 : Tensor "f32[30, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor,), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[30, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_30, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[30, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_57 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_14), kwargs = {})
#   %add_59 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %mul_57), kwargs = {})
#   return %add_59
triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62 = async_compile.triton('triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62930944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
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


# kernel path: ./.compile_cache/oq/coqk22pgph5sanefbe547h5cpd6waeisufle4bjpbt6mqpwdxb7o.py
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
#   %buf337 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf337]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %mul_69 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_69]
#   %convolution_35 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_69, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_263 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_35, [30, 32, 4, 1024]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_263, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_15 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub_27 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_263, %getitem_98), kwargs = {})
#   %add_76 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_97, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_70 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_17), kwargs = {})
#   %view_264 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [30, 128, 32, 32]), kwargs = {})
#   %view_268 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_264, [30, 128, -1]), kwargs = {})
#   %add_77 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_268, %bmm_51), kwargs = {})
#   %view_269 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [30, 128, 32, 32]), kwargs = {})
#   %view_267 : Tensor "f32[30, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_101, [30, 128, 1, 1]), kwargs = {})
#   %add_78 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_269, %view_267), kwargs = {})
#   %sigmoid_39 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convolution_36 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_71, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %mul_69), kwargs = {})
#   %sigmoid_40 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_72 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_40), kwargs = {})
#   %view_270 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_72, [30, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_103,%buf339
triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15360, 'r0_': 31457792}}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: ./.compile_cache/xm/cxmu3mxmszbbm6zef2srx2enkbh2mkjriv26qs6h2pcw3t3yeo5i.py
# Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42, input_43], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_59 => add_79
#   h_22 => mul_72, sigmoid_40
#   input_42 => add_80, add_81, mul_73, mul_74, rsqrt_18, sub_28, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, var_mean_18, view_270, view_271
#   input_43 => mul_75, sigmoid_41
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
#   %buf337 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf337]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %mul_69 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_69]
#   %getitem_103 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=getitem_103]
#   %buf339 : Tensor "f32[30, 32, 1, 1][32, 1, 960, 960]cuda:0" = PlaceHolder[target=buf339]
#   %arg150_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg150_1]
#   %arg151_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg151_1]
#   %add_81 : Tensor "f32[30, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_81]
#   %convolution_35 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_69, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_263 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_35, [30, 32, 4, 1024]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_263, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_15 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub_27 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_263, %getitem_98), kwargs = {})
#   %add_76 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_97, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_70 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_17), kwargs = {})
#   %view_264 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [30, 128, 32, 32]), kwargs = {})
#   %view_268 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_264, [30, 128, -1]), kwargs = {})
#   %add_77 : Tensor "f32[30, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_268, %bmm_51), kwargs = {})
#   %view_269 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [30, 128, 32, 32]), kwargs = {})
#   %view_267 : Tensor "f32[30, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_101, [30, 128, 1, 1]), kwargs = {})
#   %add_78 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_269, %view_267), kwargs = {})
#   %sigmoid_39 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convolution_36 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_71, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %mul_69), kwargs = {})
#   %sigmoid_40 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_72 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_40), kwargs = {})
#   %view_270 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_72, [30, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %getitem_103), kwargs = {})
#   %add_80 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_102, 1e-05), kwargs = {})
#   %rsqrt_18 : Tensor "f32[30, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   %mul_73 : Tensor "f32[30, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_18), kwargs = {})
#   %view_271 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_73, [30, 128, 32, 32]), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, 0), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, 2), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 3), kwargs = {})
#   %mul_74 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_271, %unsqueeze_17), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg151_1, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 3), kwargs = {})
#   %add_81 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_20), kwargs = {})
#   %sigmoid_41 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   return %add_81,%mul_75
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_64 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62916096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
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
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/qs/cqsjtptav7otv5tk665ritli3zhjzxnmoz44pdt72c6eybevwt2y.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_75, sigmoid_41
#   input_44 => convolution_37
# Graph fragment:
#   %arg152_1 : Tensor "f32[4, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg152_1]
#   %sigmoid_41 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convolution_37 : Tensor "f32[30, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf343
triton_poi_fused_convolution_silu_65 = async_compile.triton('triton_poi_fused_convolution_silu_65', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 36864, 'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_65(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: ./.compile_cache/nv/cnvfi6alzudnqlduf5gzjrwlont3d2hhvd5cuvyhys46cs4lrrug.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_75, sigmoid_41
#   input_44 => convolution_37
# Graph fragment:
#   %buf344 : Tensor "f32[30, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf344]
#   %arg153_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %sigmoid_41 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[30, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convolution_37 : Tensor "f32[30, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_37
triton_poi_fused_convolution_silu_66 = async_compile.triton('triton_poi_fused_convolution_silu_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 491536, 'x': 983040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_66(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 120
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 4096*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1024*y3), tmp2, xmask & ymask)
''', device_str='cuda')

def partition_0(args):
    arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1 = args
    args.clear()
    assert_size_stride(arg25_1, (30, 4, 32, 32), (4096, 1024, 32, 1))
    assert_size_stride(arg23_1, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg0_1, (30, ), (1, ))
    assert_size_stride(arg1_1, (1024, 256), (256, 1))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg3_1, (256, 1024), (1024, 1))
    assert_size_stride(arg6_1, (30, ), (1, ))
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
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (4, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg153_1, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((30, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg25_1, buf0, 120, 1024, stream=stream0)
        del arg25_1
        buf1 = empty_strided_cuda((128, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg23_1, buf1, 512, 9, stream=stream0)
        del arg23_1
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf3, arg24_1, 3932160, stream=stream0)
        del arg24_1
        buf4 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg26_1, buf4, 16384, 9, stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf4
        buf6 = empty_strided_cuda((30, 32, 1, 1), (32, 1, 960, 960), torch.float32)
        buf7 = empty_strided_cuda((30, 32, 1, 1), (32, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf5, arg27_1, buf6, buf7, 960, 4096, stream=stream0)
        buf9 = empty_strided_cuda((30, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5.run(arg0_1, buf9, 7680, stream=stream0)
        del arg0_1
        buf10 = empty_strided_cuda((30, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.t, aten.addmm]
        extern_kernels.mm(buf9, reinterpret_tensor(arg1_1, (256, 1024), (1, 256), 0), out=buf10)
        del arg1_1
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_6.run(buf11, arg2_1, 30720, stream=stream0)
        del arg2_1
        buf14 = empty_strided_cuda((30, 512), (512, 1), torch.float32)
        buf12 = reinterpret_tensor(buf14, (30, 256), (512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg4_1, buf11, reinterpret_tensor(arg3_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf12)
        del arg3_1
        del arg4_1
        del buf11
        buf13 = reinterpret_tensor(buf14, (30, 256), (512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [s_emb], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_7.run(arg6_1, arg5_1, buf13, 7680, stream=stream0)
        del arg5_1
        del arg6_1
        del buf12
        del buf13
        buf15 = empty_strided_cuda((30, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf14, reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf15)
        del arg7_1
        del buf14
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_8.run(buf16, arg8_1, 15360, stream=stream0)
        del arg8_1
        buf17 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg10_1, buf16, reinterpret_tensor(arg9_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf17)
        del arg10_1
        del arg9_1
        del buf16
        buf18 = empty_strided_cuda((30, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), out=buf18)
        del arg11_1
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf19, arg12_1, 7680, stream=stream0)
        del arg12_1
        buf20 = empty_strided_cuda((30, 3200), (3200, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg14_1, buf19, reinterpret_tensor(arg13_1, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf20)
        del arg13_1
        del arg14_1
        del buf19
        buf21 = empty_strided_cuda((30, 32, 4, 1024), (131072, 4096, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf5, arg27_1, buf6, buf7, buf21, 3840, 1024, stream=stream0)
        del arg27_1
        buf22 = empty_strided_cuda((30, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm, split, v_1, transpose, x_flat, v_t_x], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf21, (30, 128, 1024), (131072, 1024, 1), 0), out=buf22)
        buf23 = reinterpret_tensor(buf5, (30, 128, 1024), (131072, 1024, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [split, u_1, mixed], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 128, 12), (3200, 12, 1), 0), buf22, out=buf23)
        del buf22
        buf24 = empty_strided_cuda((30, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf21, buf23, buf20, buf24, 3840, 1024, stream=stream0)
        del buf21
        del buf23
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf25 = extern_kernels.convolution(buf24, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg28_1
        del buf24
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2, add_2, h_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf26, arg29_1, buf3, 3932160, stream=stream0)
        del arg29_1
        buf27 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg30_1, buf27, 16384, 9, stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf26, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf27
        buf29 = buf7; del buf7  # reuse
        buf30 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf28, arg31_1, buf29, buf30, 960, 4096, stream=stream0)
        buf32 = empty_strided_cuda((30, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg37_1, (256, 512), (1, 256), 0), out=buf32)
        del arg37_1
        buf33 = reinterpret_tensor(buf3, (30, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf28, arg31_1, buf29, buf30, buf33, 3840, 1024, stream=stream0)
        del arg31_1
        del buf29
        del buf30
        buf34 = empty_strided_cuda((30, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, v_3, transpose_1, x_flat_1, v_t_x_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf33, (30, 128, 1024), (131072, 1024, 1), 0), out=buf34)
        buf35 = reinterpret_tensor(buf28, (30, 128, 1024), (131072, 1024, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [split_1, u_3, mixed_1], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 128, 12), (3200, 12, 1), 0), buf34, out=buf35)
        del buf34
        buf36 = empty_strided_cuda((30, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf33, buf35, buf20, buf36, 3840, 1024, stream=stream0)
        del buf33
        del buf35
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4, x_5], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf37 = extern_kernels.convolution(buf36, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg32_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4, x_5, add_5, h_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf38, arg33_1, buf26, 3932160, stream=stream0)
        del arg33_1
        del buf26
        buf39 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_13.run(arg34_1, buf39, 32768, 9, stream=stream0)
        del arg34_1
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf39
        buf41 = empty_strided_cuda((30, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_transpose_view_14.run(buf40, arg35_1, buf41, 1966080, stream=stream0)
        buf42 = empty_strided_cuda((7680, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (7680, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 256), (1, 256), 0), out=buf42)
        del arg36_1
        buf43 = reinterpret_tensor(buf41, (30, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf42, buf43, 1966080, stream=stream0)
        buf44 = empty_strided_cuda((30, 4, 64, 1), (256, 64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf32, buf44, 7680, stream=stream0)
        buf45 = empty_strided_cuda((120, 256, 1), (256, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [kv, chunk, q_1, view_11, q_2, matmul, view_12, k_1, transpose_6], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (120, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf44, (120, 64, 1), (64, 1, 0), 0), out=buf45)
        buf46 = reinterpret_tensor(buf45, (30, 4, 256, 1), (1024, 256, 1, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf46, 30720, stream=stream0)
        buf47 = reinterpret_tensor(buf44, (30, 4, 1, 64), (256, 64, 64, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf32, buf47, 7680, stream=stream0)
        buf48 = reinterpret_tensor(buf43, (120, 256, 64), (16384, 64, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, matmul, attn_1, matmul_1, view_13, v_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (120, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf47, (120, 1, 64), (64, 0, 1), 0), out=buf48)
        buf49 = reinterpret_tensor(buf42, (30, 256, 4, 64), (65536, 256, 64, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf48, buf49, 1966080, stream=stream0)
        buf50 = reinterpret_tensor(buf48, (7680, 256), (256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4, input_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (7680, 256), (256, 1), 0), reinterpret_tensor(arg38_1, (256, 256), (1, 256), 0), out=buf50)
        del arg38_1
        buf51 = reinterpret_tensor(buf50, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_20.run(buf51, buf40, arg35_1, arg39_1, 1966080, stream=stream0)
        del arg39_1
        buf52 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg40_1, buf52, 65536, 9, stream=stream0)
        del arg40_1
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf53 = extern_kernels.convolution(buf51, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf54 = empty_strided_cuda((30, 32, 1, 1), (32, 1, 960, 960), torch.float32)
        buf55 = empty_strided_cuda((30, 32, 1, 1), (32, 1, 960, 960), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf53, arg41_1, buf54, buf55, 960, 2048, stream=stream0)
        buf57 = reinterpret_tensor(buf47, (30, 256), (256, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg15_1, (256, 256), (1, 256), 0), out=buf57)
        del arg15_1
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf58, arg16_1, 7680, stream=stream0)
        del arg16_1
        buf59 = empty_strided_cuda((30, 6400), (6400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg18_1, buf58, reinterpret_tensor(arg17_1, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf59)
        del arg17_1
        del arg18_1
        buf60 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg45_1, (256, 512), (1, 256), 0), out=buf60)
        del arg45_1
        buf61 = reinterpret_tensor(buf51, (30, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf53, arg41_1, buf54, buf55, buf61, 7680, 256, stream=stream0)
        del arg41_1
        buf62 = empty_strided_cuda((30, 12, 256), (3072, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, v_7, transpose_9, x_flat_2, v_t_x_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf61, (30, 256, 256), (65536, 256, 1), 0), out=buf62)
        buf63 = reinterpret_tensor(buf53, (30, 256, 256), (65536, 256, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [split_2, u_5, mixed_2], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 256, 12), (6400, 12, 1), 0), buf62, out=buf63)
        buf64 = reinterpret_tensor(buf49, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf61, buf63, buf59, buf64, 7680, 256, stream=stream0)
        del buf61
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf65 = extern_kernels.convolution(buf64, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg42_1
        buf66 = reinterpret_tensor(buf64, (30, 256, 256), (65536, 256, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf65, arg43_1, buf40, arg35_1, buf66, 1966080, stream=stream0)
        buf67 = reinterpret_tensor(buf63, (7680, 256), (256, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (7680, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 256), (1, 256), 0), out=buf67)
        del arg44_1
        buf68 = reinterpret_tensor(buf66, (30, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [q_4, view_21, q_5, matmul_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf67, buf68, 1966080, stream=stream0)
        buf69 = reinterpret_tensor(buf58, (30, 4, 64, 1), (256, 64, 1, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_22, k_3, transpose_14, matmul_2], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf60, buf69, 7680, stream=stream0)
        buf70 = reinterpret_tensor(buf46, (120, 256, 1), (256, 1, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, q_4, view_21, q_5, matmul_2, view_22, k_3, transpose_14], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (120, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf69, (120, 64, 1), (64, 1, 0), 0), out=buf70)
        buf71 = reinterpret_tensor(buf70, (30, 4, 256, 1), (1024, 256, 1, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [matmul_2, attn_3], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf71, 30720, stream=stream0)
        buf72 = reinterpret_tensor(buf69, (30, 4, 1, 64), (256, 64, 64, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_23, v_9, matmul_3], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf60, buf72, 7680, stream=stream0)
        buf73 = reinterpret_tensor(buf68, (120, 256, 64), (16384, 64, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, matmul_2, attn_3, matmul_3, view_23, v_9], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf71, (120, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf72, (120, 1, 64), (64, 0, 1), 0), out=buf73)
        buf74 = reinterpret_tensor(buf67, (30, 256, 4, 64), (65536, 256, 64, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf73, buf74, 1966080, stream=stream0)
        buf75 = reinterpret_tensor(buf73, (7680, 256), (256, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8, input_18], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf74, (7680, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 256), (1, 256), 0), out=buf75)
        del arg46_1
        buf76 = reinterpret_tensor(buf75, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26.run(buf76, buf65, arg43_1, buf40, arg35_1, arg47_1, 1966080, stream=stream0)
        del arg47_1
        buf77 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg48_1, buf77, 65536, 9, stream=stream0)
        del arg48_1
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf78 = extern_kernels.convolution(buf76, buf77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf79 = buf55; del buf55  # reuse
        buf80 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf78, arg49_1, buf79, buf80, 960, 2048, stream=stream0)
        buf82 = reinterpret_tensor(buf71, (30, 1024), (1024, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg55_1, (256, 1024), (1, 256), 0), out=buf82)
        del arg55_1
        buf83 = reinterpret_tensor(buf76, (30, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf78, arg49_1, buf79, buf80, buf83, 7680, 256, stream=stream0)
        del arg49_1
        buf84 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, v_11, transpose_17, x_flat_3, v_t_x_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf83, (30, 256, 256), (65536, 256, 1), 0), out=buf84)
        buf85 = reinterpret_tensor(buf78, (30, 256, 256), (65536, 256, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [split_3, u_7, mixed_3], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 256, 12), (6400, 12, 1), 0), buf84, out=buf85)
        buf86 = reinterpret_tensor(buf74, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf83, buf85, buf59, buf86, 7680, 256, stream=stream0)
        del buf83
        del buf85
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf87 = extern_kernels.convolution(buf86, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg50_1
        del buf86
        buf88 = buf87; del buf87  # reuse
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13, add_13, h_5], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27.run(buf89, arg51_1, buf65, arg43_1, buf40, arg35_1, 1966080, stream=stream0)
        del arg35_1
        del arg43_1
        del arg51_1
        del buf40
        del buf65
        buf90 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_28.run(arg52_1, buf90, 131072, 9, stream=stream0)
        del arg52_1
        # Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
        buf91 = extern_kernels.convolution(buf89, buf90, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (30, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf90
        buf92 = empty_strided_cuda((30, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_silu_transpose_view_29.run(buf91, arg53_1, buf92, 983040, stream=stream0)
        buf93 = empty_strided_cuda((1920, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (1920, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf93)
        del arg54_1
        buf94 = reinterpret_tensor(buf92, (30, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf93, buf94, 983040, stream=stream0)
        buf95 = reinterpret_tensor(buf60, (30, 8, 64, 1), (512, 64, 1, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf82, buf95, 15360, stream=stream0)
        buf96 = empty_strided_cuda((240, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, q_7, view_31, q_8, matmul_4, view_32, k_5, transpose_22], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (240, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf95, (240, 64, 1), (64, 1, 0), 0), out=buf96)
        buf97 = reinterpret_tensor(buf96, (30, 8, 64, 1), (512, 64, 1, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_32.run(buf97, 15360, stream=stream0)
        buf98 = reinterpret_tensor(buf95, (30, 8, 1, 64), (512, 64, 64, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf82, buf98, 15360, stream=stream0)
        buf99 = reinterpret_tensor(buf94, (240, 64, 64), (4096, 64, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, matmul_4, attn_5, matmul_5, view_33, v_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf97, (240, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf98, (240, 1, 64), (64, 0, 1), 0), out=buf99)
        buf100 = reinterpret_tensor(buf93, (30, 64, 8, 64), (32768, 512, 64, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf99, buf100, 983040, stream=stream0)
        buf101 = reinterpret_tensor(buf99, (1920, 512), (512, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12, input_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1920, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 512), (1, 512), 0), out=buf101)
        del arg56_1
        buf102 = reinterpret_tensor(buf101, (30, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_35.run(buf102, buf91, arg53_1, arg57_1, 983040, stream=stream0)
        del arg57_1
        buf103 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(arg58_1, buf103, 262144, 9, stream=stream0)
        del arg58_1
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf104 = extern_kernels.convolution(buf102, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (30, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf105 = buf80; del buf80  # reuse
        buf106 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37.run(buf104, arg59_1, buf105, buf106, 960, 1024, stream=stream0)
        buf108 = reinterpret_tensor(buf72, (30, 256), (256, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg19_1, (256, 256), (1, 256), 0), out=buf108)
        del arg19_1
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf109, arg20_1, 7680, stream=stream0)
        del arg20_1
        buf110 = empty_strided_cuda((30, 12800), (12800, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg22_1, buf109, reinterpret_tensor(arg21_1, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf110)
        del arg21_1
        del arg22_1
        buf111 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg63_1, (256, 1024), (1, 256), 0), out=buf111)
        del arg63_1
        buf112 = reinterpret_tensor(buf102, (30, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf104, arg59_1, buf105, buf106, buf112, 15360, 64, stream=stream0)
        del arg59_1
        buf113 = empty_strided_cuda((30, 12, 64), (768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, v_15, transpose_25, x_flat_4, v_t_x_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (30, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf112, (30, 512, 64), (32768, 64, 1), 0), out=buf113)
        buf114 = reinterpret_tensor(buf104, (30, 512, 64), (32768, 64, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [split_4, u_9, mixed_4], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (30, 512, 12), (12800, 12, 1), 0), buf113, out=buf114)
        buf115 = reinterpret_tensor(buf100, (30, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39.run(buf112, buf114, buf110, buf115, 15360, 64, stream=stream0)
        del buf112
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf116 = extern_kernels.convolution(buf115, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (30, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del arg60_1
        buf117 = reinterpret_tensor(buf115, (30, 64, 512), (32768, 512, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40.run(buf116, arg61_1, buf91, arg53_1, buf117, 983040, stream=stream0)
        buf118 = reinterpret_tensor(buf114, (1920, 512), (512, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (1920, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 512), (1, 512), 0), out=buf118)
        del arg62_1
        buf119 = reinterpret_tensor(buf117, (30, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [q_10, view_41, q_11, matmul_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf118, buf119, 983040, stream=stream0)
        buf120 = reinterpret_tensor(buf98, (30, 8, 64, 1), (512, 64, 1, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_42, k_7, transpose_30, matmul_6], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf111, buf120, 15360, stream=stream0)
        buf121 = reinterpret_tensor(buf97, (240, 64, 1), (64, 1, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, q_10, view_41, q_11, matmul_6, view_42, k_7, transpose_30], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (240, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf120, (240, 64, 1), (64, 1, 0), 0), out=buf121)
        buf122 = reinterpret_tensor(buf121, (30, 8, 64, 1), (512, 64, 1, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [matmul_6, attn_7], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_32.run(buf122, 15360, stream=stream0)
        buf123 = reinterpret_tensor(buf120, (30, 8, 1, 64), (512, 64, 64, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_43, v_17, matmul_7], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf111, buf123, 15360, stream=stream0)
        buf124 = reinterpret_tensor(buf119, (240, 64, 64), (4096, 64, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, matmul_6, attn_7, matmul_7, view_43, v_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (240, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf123, (240, 1, 64), (64, 0, 1), 0), out=buf124)
        del buf122
        buf125 = reinterpret_tensor(buf118, (30, 64, 8, 64), (32768, 512, 64, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf124, buf125, 983040, stream=stream0)
        buf126 = reinterpret_tensor(buf124, (1920, 512), (512, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16, input_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1920, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf126)
        del arg64_1
        buf127 = reinterpret_tensor(buf126, (30, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41.run(buf127, buf116, arg61_1, buf91, arg53_1, arg65_1, 983040, stream=stream0)
        del arg65_1
        buf128 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(arg66_1, buf128, 262144, 9, stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf129 = extern_kernels.convolution(buf127, buf128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (30, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf128
        buf130 = buf106; del buf106  # reuse
        buf131 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_37.run(buf129, arg67_1, buf130, buf131, 960, 1024, stream=stream0)
        buf133 = reinterpret_tensor(buf127, (30, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf129, arg67_1, buf130, buf131, buf133, 15360, 64, stream=stream0)
        del arg67_1
        buf134 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, v_19, transpose_33, x_flat_5, v_t_x_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (30, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf133, (30, 512, 64), (32768, 64, 1), 0), out=buf134)
        buf135 = reinterpret_tensor(buf129, (30, 512, 64), (32768, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [split_5, u_11, mixed_5], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (30, 512, 12), (12800, 12, 1), 0), buf134, out=buf135)
        del buf134
        buf136 = reinterpret_tensor(buf125, (30, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_39.run(buf133, buf135, buf110, buf136, 15360, 64, stream=stream0)
        del buf110
        del buf133
        del buf135
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf137 = extern_kernels.convolution(buf136, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (30, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del arg68_1
        buf138 = buf137; del buf137  # reuse
        buf142 = reinterpret_tensor(buf136, (30, 64, 512), (32768, 512, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42.run(buf138, arg69_1, buf116, arg61_1, buf91, arg53_1, arg70_1, arg71_1, buf142, 1920, 512, stream=stream0)
        del arg53_1
        del arg61_1
        del arg69_1
        del arg70_1
        del arg71_1
        buf143 = empty_strided_cuda((1920, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf142, (1920, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 1536), (1, 512), 0), out=buf143)
        del arg72_1
        buf144 = reinterpret_tensor(buf142, (30, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_43.run(buf143, arg73_1, buf144, 983040, stream=stream0)
        buf145 = reinterpret_tensor(buf91, (30, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_transpose_view_44.run(buf143, arg73_1, buf145, 15360, 64, stream=stream0)
        buf146 = reinterpret_tensor(buf116, (240, 64, 64), (4096, 64, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8, k_8, transpose_35], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone, aten._unsafe_view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (240, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf145, (240, 64, 64), (4096, 64, 1), 0), out=buf146)
        buf149 = reinterpret_tensor(buf146, (30, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_amax_mul_sub_view_45.run(buf149, 15360, 64, stream=stream0)
        buf150 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_46.run(buf143, arg73_1, buf150, 983040, stream=stream0)
        del arg73_1
        del buf143
        buf151 = reinterpret_tensor(buf144, (240, 64, 64), (4096, 64, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, matmul_8, attn_9, out_20, v_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.mul, aten.sub, aten._softmax, aten.select, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (240, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf150, (240, 64, 64), (4096, 64, 1), 0), out=buf151)
        del buf149
        buf152 = reinterpret_tensor(buf150, (30, 64, 8, 64), (32768, 512, 64, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_34.run(buf151, buf152, 983040, stream=stream0)
        buf153 = reinterpret_tensor(buf151, (1920, 512), (512, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21, out_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1920, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf153)
        del arg74_1
        del buf152
        buf158 = reinterpret_tensor(buf138, (30, 64, 512), (32768, 512, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_47.run(buf158, buf153, arg75_1, arg76_1, arg77_1, 1920, 512, stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        del buf153
        buf157 = reinterpret_tensor(buf123, (30, 512), (512, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg85_1, (256, 512), (1, 256), 0), out=buf157)
        del arg85_1
        buf159 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_48.run(arg78_1, buf159, 131072, 16, stream=stream0)
        del arg78_1
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        buf160 = extern_kernels.convolution(reinterpret_tensor(buf158, (30, 512, 8, 8), (32768, 1, 4096, 512), 0), buf159, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf158
        del buf159
        buf161 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg80_1, (256, 256), (1, 256), 0), out=buf161)
        del arg80_1
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf162, arg81_1, 7680, stream=stream0)
        del arg81_1
        buf163 = empty_strided_cuda((32, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_49.run(buf162, buf163, 8192, stream=stream0)
        buf164 = empty_strided_cuda((32, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf163, reinterpret_tensor(arg82_1, (256, 256), (1, 256), 0), out=buf164)
        del arg82_1
        del buf163
        buf165 = empty_strided_cuda((30, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50.run(buf160, arg79_1, buf89, buf164, arg83_1, buf165, 7680, 256, stream=stream0)
        del arg79_1
        del arg83_1
        del buf164
        buf166 = reinterpret_tensor(buf89, (7680, 256), (1, 7680), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_t_transpose_view_51.run(buf165, buf166, 1966080, stream=stream0)
        buf167 = reinterpret_tensor(buf160, (7680, 256), (256, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        extern_kernels.mm(buf166, reinterpret_tensor(arg84_1, (256, 256), (1, 256), 0), out=buf167)
        del arg84_1
        buf168 = reinterpret_tensor(buf166, (30, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [q_14, view_53, q_15, matmul_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf167, buf168, 1966080, stream=stream0)
        buf169 = reinterpret_tensor(buf162, (30, 4, 64, 1), (256, 64, 1, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_54, k_10, transpose_42, matmul_10], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf157, buf169, 7680, stream=stream0)
        buf170 = reinterpret_tensor(buf111, (120, 256, 1), (256, 1, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, q_14, view_53, q_15, matmul_10, view_54, k_10, transpose_42], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (120, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf169, (120, 64, 1), (64, 1, 0), 0), out=buf170)
        buf171 = reinterpret_tensor(buf170, (30, 4, 256, 1), (1024, 256, 1, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [matmul_10, attn_11], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf171, 30720, stream=stream0)
        buf172 = reinterpret_tensor(buf169, (30, 4, 1, 64), (256, 64, 64, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_55, v_22, matmul_11], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf157, buf172, 7680, stream=stream0)
        buf173 = reinterpret_tensor(buf168, (120, 256, 64), (16384, 64, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, matmul_10, attn_11, matmul_11, view_55, v_22], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (120, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf172, (120, 1, 64), (64, 0, 1), 0), out=buf173)
        buf174 = reinterpret_tensor(buf167, (30, 256, 4, 64), (65536, 256, 64, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf173, buf174, 1966080, stream=stream0)
        buf175 = reinterpret_tensor(buf173, (7680, 256), (256, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26, input_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf174, (7680, 256), (256, 1), 0), reinterpret_tensor(arg86_1, (256, 256), (1, 256), 0), out=buf175)
        del arg86_1
        buf176 = reinterpret_tensor(buf175, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_transpose_view_52.run(buf176, buf165, arg87_1, 7680, 256, stream=stream0)
        del arg87_1
        buf177 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg88_1, buf177, 65536, 9, stream=stream0)
        del arg88_1
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf179 = buf131; del buf131  # reuse
        buf180 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf178, arg89_1, buf179, buf180, 960, 2048, stream=stream0)
        buf182 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg93_1, (256, 512), (1, 256), 0), out=buf182)
        del arg93_1
        buf183 = reinterpret_tensor(buf176, (30, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf178, arg89_1, buf179, buf180, buf183, 7680, 256, stream=stream0)
        del arg89_1
        buf184 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, v_24, transpose_45, x_flat_7, v_t_x_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf183, (30, 256, 256), (65536, 256, 1), 0), out=buf184)
        buf185 = reinterpret_tensor(buf178, (30, 256, 256), (65536, 256, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [split_6, u_13, mixed_6], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 256, 12), (6400, 12, 1), 0), buf184, out=buf185)
        buf186 = reinterpret_tensor(buf174, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf183, buf185, buf59, buf186, 7680, 256, stream=stream0)
        del buf183
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf187 = extern_kernels.convolution(buf186, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg90_1
        buf188 = reinterpret_tensor(buf186, (30, 256, 256), (65536, 256, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53.run(buf187, arg91_1, buf165, buf188, 7680, 256, stream=stream0)
        buf189 = reinterpret_tensor(buf185, (7680, 256), (256, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (7680, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf189)
        del arg92_1
        buf190 = reinterpret_tensor(buf188, (30, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [q_17, view_63, q_18, matmul_12], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf189, buf190, 1966080, stream=stream0)
        buf191 = reinterpret_tensor(buf172, (30, 4, 64, 1), (256, 64, 1, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_64, k_12, transpose_50, matmul_12], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf182, buf191, 7680, stream=stream0)
        buf192 = reinterpret_tensor(buf171, (120, 256, 1), (256, 1, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, q_17, view_63, q_18, matmul_12, view_64, k_12, transpose_50], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (120, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf191, (120, 64, 1), (64, 1, 0), 0), out=buf192)
        buf193 = reinterpret_tensor(buf192, (30, 4, 256, 1), (1024, 256, 1, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [matmul_12, attn_13], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf193, 30720, stream=stream0)
        buf194 = reinterpret_tensor(buf191, (30, 4, 1, 64), (256, 64, 64, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_65, v_26, matmul_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf182, buf194, 7680, stream=stream0)
        buf195 = reinterpret_tensor(buf190, (120, 256, 64), (16384, 64, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, matmul_12, attn_13, matmul_13, view_65, v_26], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (120, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf194, (120, 1, 64), (64, 0, 1), 0), out=buf195)
        buf196 = reinterpret_tensor(buf189, (30, 256, 4, 64), (65536, 256, 64, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf195, buf196, 1966080, stream=stream0)
        buf197 = reinterpret_tensor(buf195, (7680, 256), (256, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30, input_30], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf196, (7680, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 256), (1, 256), 0), out=buf197)
        del arg94_1
        buf198 = reinterpret_tensor(buf197, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54.run(buf198, buf187, arg91_1, buf165, arg95_1, 7680, 256, stream=stream0)
        del arg95_1
        buf199 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg96_1, buf199, 65536, 9, stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf200 = extern_kernels.convolution(buf198, buf199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf201 = buf180; del buf180  # reuse
        buf202 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf200, arg97_1, buf201, buf202, 960, 2048, stream=stream0)
        buf204 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg101_1, (256, 512), (1, 256), 0), out=buf204)
        del arg101_1
        buf205 = reinterpret_tensor(buf198, (30, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf200, arg97_1, buf201, buf202, buf205, 7680, 256, stream=stream0)
        del arg97_1
        buf206 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, v_28, transpose_53, x_flat_8, v_t_x_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf205, (30, 256, 256), (65536, 256, 1), 0), out=buf206)
        buf207 = reinterpret_tensor(buf200, (30, 256, 256), (65536, 256, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [split_7, u_15, mixed_7], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 256, 12), (6400, 12, 1), 0), buf206, out=buf207)
        buf208 = reinterpret_tensor(buf196, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf205, buf207, buf59, buf208, 7680, 256, stream=stream0)
        del buf205
        del buf207
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf209 = extern_kernels.convolution(buf208, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg98_1
        buf210 = buf209; del buf209  # reuse
        buf211 = reinterpret_tensor(buf208, (30, 256, 256), (65536, 256, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55.run(buf210, arg99_1, buf187, arg91_1, buf165, buf211, 1966080, stream=stream0)
        del arg91_1
        del arg99_1
        del buf165
        buf212 = reinterpret_tensor(buf187, (7680, 256), (256, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [h_12, view_72, q_19, q_20], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (7680, 256), (256, 1), 0), reinterpret_tensor(arg100_1, (256, 256), (1, 256), 0), out=buf212)
        del arg100_1
        buf213 = reinterpret_tensor(buf211, (30, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [q_20, view_73, q_21, matmul_14], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf212, buf213, 1966080, stream=stream0)
        buf214 = reinterpret_tensor(buf194, (30, 4, 64, 1), (256, 64, 1, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_74, k_14, transpose_58, matmul_14], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf204, buf214, 7680, stream=stream0)
        buf215 = reinterpret_tensor(buf193, (120, 256, 1), (256, 1, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, q_20, view_73, q_21, matmul_14, view_74, k_14, transpose_58], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (120, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf214, (120, 64, 1), (64, 1, 0), 0), out=buf215)
        buf216 = reinterpret_tensor(buf215, (30, 4, 256, 1), (1024, 256, 1, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [matmul_14, attn_15], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf216, 30720, stream=stream0)
        buf217 = reinterpret_tensor(buf214, (30, 4, 1, 64), (256, 64, 64, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_75, v_30, matmul_15], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf204, buf217, 7680, stream=stream0)
        buf218 = reinterpret_tensor(buf213, (120, 256, 64), (16384, 64, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, matmul_14, attn_15, matmul_15, view_75, v_30], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (120, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf217, (120, 1, 64), (64, 0, 1), 0), out=buf218)
        buf219 = reinterpret_tensor(buf212, (30, 256, 4, 64), (65536, 256, 64, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf218, buf219, 1966080, stream=stream0)
        buf220 = reinterpret_tensor(buf218, (7680, 256), (256, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34, input_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf219, (7680, 256), (256, 1), 0), reinterpret_tensor(arg102_1, (256, 256), (1, 256), 0), out=buf220)
        del arg102_1
        buf221 = reinterpret_tensor(buf220, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_56.run(buf221, buf210, arg103_1, 1966080, stream=stream0)
        del arg103_1
        buf222 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg104_1, buf222, 65536, 9, stream=stream0)
        del arg104_1
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf223 = extern_kernels.convolution(buf221, buf222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf224 = buf202; del buf202  # reuse
        buf225 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf223, arg105_1, buf224, buf225, 960, 2048, stream=stream0)
        buf227 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg109_1, (256, 512), (1, 256), 0), out=buf227)
        del arg109_1
        buf228 = reinterpret_tensor(buf221, (30, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf223, arg105_1, buf224, buf225, buf228, 7680, 256, stream=stream0)
        del arg105_1
        buf229 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, v_32, transpose_61, x_flat_9, v_t_x_8], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf228, (30, 256, 256), (65536, 256, 1), 0), out=buf229)
        buf230 = reinterpret_tensor(buf223, (30, 256, 256), (65536, 256, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [split_8, u_17, mixed_8], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 256, 12), (6400, 12, 1), 0), buf229, out=buf230)
        buf231 = reinterpret_tensor(buf219, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf228, buf230, buf59, buf231, 7680, 256, stream=stream0)
        del buf228
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf232 = extern_kernels.convolution(buf231, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg106_1
        buf233 = reinterpret_tensor(buf231, (30, 256, 256), (65536, 256, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57.run(buf232, arg107_1, buf210, buf233, 1966080, stream=stream0)
        buf234 = reinterpret_tensor(buf230, (7680, 256), (256, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (7680, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), out=buf234)
        del arg108_1
        buf235 = reinterpret_tensor(buf233, (30, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [q_23, view_83, q_24, matmul_16], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf234, buf235, 1966080, stream=stream0)
        buf236 = reinterpret_tensor(buf217, (30, 4, 64, 1), (256, 64, 1, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_84, k_16, transpose_66, matmul_16], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf227, buf236, 7680, stream=stream0)
        buf237 = reinterpret_tensor(buf216, (120, 256, 1), (256, 1, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, q_23, view_83, q_24, matmul_16, view_84, k_16, transpose_66], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (120, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf236, (120, 64, 1), (64, 1, 0), 0), out=buf237)
        buf238 = reinterpret_tensor(buf237, (30, 4, 256, 1), (1024, 256, 1, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [matmul_16, attn_17], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf238, 30720, stream=stream0)
        buf239 = reinterpret_tensor(buf236, (30, 4, 1, 64), (256, 64, 64, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_85, v_34, matmul_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf227, buf239, 7680, stream=stream0)
        buf240 = reinterpret_tensor(buf235, (120, 256, 64), (16384, 64, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, matmul_16, attn_17, matmul_17, view_85, v_34], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf238, (120, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf239, (120, 1, 64), (64, 0, 1), 0), out=buf240)
        buf241 = reinterpret_tensor(buf234, (30, 256, 4, 64), (65536, 256, 64, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf240, buf241, 1966080, stream=stream0)
        buf242 = reinterpret_tensor(buf240, (7680, 256), (256, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38, input_34], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf241, (7680, 256), (256, 1), 0), reinterpret_tensor(arg110_1, (256, 256), (1, 256), 0), out=buf242)
        del arg110_1
        buf243 = reinterpret_tensor(buf242, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58.run(buf243, buf232, arg107_1, buf210, arg111_1, 1966080, stream=stream0)
        del arg111_1
        buf244 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg112_1, buf244, 65536, 9, stream=stream0)
        del arg112_1
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf245 = extern_kernels.convolution(buf243, buf244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf246 = buf225; del buf225  # reuse
        buf247 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf245, arg113_1, buf246, buf247, 960, 2048, stream=stream0)
        buf249 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg117_1, (256, 512), (1, 256), 0), out=buf249)
        del arg117_1
        buf250 = reinterpret_tensor(buf243, (30, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf245, arg113_1, buf246, buf247, buf250, 7680, 256, stream=stream0)
        del arg113_1
        buf251 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, v_36, transpose_69, x_flat_10, v_t_x_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf250, (30, 256, 256), (65536, 256, 1), 0), out=buf251)
        buf252 = reinterpret_tensor(buf245, (30, 256, 256), (65536, 256, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [split_9, u_19, mixed_9], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 256, 12), (6400, 12, 1), 0), buf251, out=buf252)
        buf253 = reinterpret_tensor(buf241, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf250, buf252, buf59, buf253, 7680, 256, stream=stream0)
        del buf250
        del buf252
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf254 = extern_kernels.convolution(buf253, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg114_1
        buf255 = buf254; del buf254  # reuse
        buf256 = reinterpret_tensor(buf253, (30, 256, 256), (65536, 256, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59.run(buf255, arg115_1, buf232, arg107_1, buf210, buf256, 1966080, stream=stream0)
        del arg107_1
        del arg115_1
        del buf210
        buf257 = reinterpret_tensor(buf232, (7680, 256), (256, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (7680, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 256), (1, 256), 0), out=buf257)
        del arg116_1
        buf258 = reinterpret_tensor(buf256, (30, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [q_26, view_93, q_27, matmul_18], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_15.run(buf257, buf258, 1966080, stream=stream0)
        buf259 = reinterpret_tensor(buf239, (30, 4, 64, 1), (256, 64, 1, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_94, k_18, transpose_74, matmul_18], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_16.run(buf249, buf259, 7680, stream=stream0)
        buf260 = reinterpret_tensor(buf238, (120, 256, 1), (256, 1, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, q_26, view_93, q_27, matmul_18, view_94, k_18, transpose_74], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf258, (120, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf259, (120, 64, 1), (64, 1, 0), 0), out=buf260)
        buf261 = reinterpret_tensor(buf260, (30, 4, 256, 1), (1024, 256, 1, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [matmul_18, attn_19], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_17.run(buf261, 30720, stream=stream0)
        buf262 = reinterpret_tensor(buf259, (30, 4, 1, 64), (256, 64, 64, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_95, v_38, matmul_19], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_18.run(buf249, buf262, 7680, stream=stream0)
        del buf249
        buf263 = reinterpret_tensor(buf258, (120, 256, 64), (16384, 64, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, matmul_18, attn_19, matmul_19, view_95, v_38], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (120, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf262, (120, 1, 64), (64, 0, 1), 0), out=buf263)
        del buf261
        buf264 = reinterpret_tensor(buf257, (30, 256, 4, 64), (65536, 256, 64, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_19.run(buf263, buf264, 1966080, stream=stream0)
        buf265 = reinterpret_tensor(buf263, (7680, 256), (256, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42, input_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf264, (7680, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 256), (1, 256), 0), out=buf265)
        del arg118_1
        buf266 = reinterpret_tensor(buf265, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_56.run(buf266, buf255, arg119_1, 1966080, stream=stream0)
        del arg119_1
        buf267 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(arg120_1, buf267, 65536, 9, stream=stream0)
        del arg120_1
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf267
        buf269 = buf247; del buf247  # reuse
        buf270 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_22.run(buf268, arg121_1, buf269, buf270, 960, 2048, stream=stream0)
        buf272 = reinterpret_tensor(buf266, (30, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf268, arg121_1, buf269, buf270, buf272, 7680, 256, stream=stream0)
        del arg121_1
        buf273 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, v_40, transpose_77, x_flat_11, v_t_x_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf272, (30, 256, 256), (65536, 256, 1), 0), out=buf273)
        buf274 = reinterpret_tensor(buf268, (30, 256, 256), (65536, 256, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [split_10, u_21, mixed_10], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (30, 256, 12), (6400, 12, 1), 0), buf273, out=buf274)
        del buf273
        buf275 = reinterpret_tensor(buf264, (30, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_24.run(buf272, buf274, buf59, buf275, 7680, 256, stream=stream0)
        del buf272
        del buf274
        del buf59
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf276 = extern_kernels.convolution(buf275, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (30, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg122_1
        del buf275
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60.run(buf277, arg123_1, buf255, 1966080, stream=stream0)
        del arg123_1
        del buf255
        buf278 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_61.run(arg124_1, buf278, 32768, 16, stream=stream0)
        del arg124_1
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf279 = extern_kernels.convolution(buf277, buf278, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf277
        del buf278
        buf280 = reinterpret_tensor(buf262, (30, 256), (256, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg126_1, (256, 256), (1, 256), 0), out=buf280)
        del arg126_1
        del buf17
        buf281 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf281, arg127_1, 7680, stream=stream0)
        del arg127_1
        buf282 = empty_strided_cuda((30, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf281, reinterpret_tensor(arg128_1, (256, 128), (1, 256), 0), out=buf282)
        del arg128_1
        del buf281
        buf283 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62.run(buf283, arg125_1, buf38, buf282, arg129_1, 3932160, stream=stream0)
        del arg125_1
        del arg129_1
        del buf282
        buf284 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg130_1, buf284, 16384, 9, stream=stream0)
        del arg130_1
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf283, buf284, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf284
        buf286 = buf270; del buf270  # reuse
        buf287 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf285, arg131_1, buf286, buf287, 960, 4096, stream=stream0)
        buf289 = reinterpret_tensor(buf38, (30, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf285, arg131_1, buf286, buf287, buf289, 3840, 1024, stream=stream0)
        del arg131_1
        buf290 = empty_strided_cuda((30, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, v_42, transpose_78, x_flat_12, v_t_x_11], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf289, (30, 128, 1024), (131072, 1024, 1), 0), out=buf290)
        buf291 = reinterpret_tensor(buf285, (30, 128, 1024), (131072, 1024, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [split_11, u_23, mixed_11], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 128, 12), (3200, 12, 1), 0), buf290, out=buf291)
        del buf290
        buf292 = empty_strided_cuda((30, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf289, buf291, buf20, buf292, 3840, 1024, stream=stream0)
        del buf289
        del buf291
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43, x_44], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf293 = extern_kernels.convolution(buf292, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg132_1
        del buf292
        buf294 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43, x_44, add_47, h_18], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf294, arg133_1, buf283, 3932160, stream=stream0)
        del arg133_1
        buf295 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg134_1, buf295, 16384, 9, stream=stream0)
        del arg134_1
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf294, buf295, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf295
        buf297 = buf287; del buf287  # reuse
        buf298 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf296, arg135_1, buf297, buf298, 960, 4096, stream=stream0)
        buf300 = reinterpret_tensor(buf283, (30, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf296, arg135_1, buf297, buf298, buf300, 3840, 1024, stream=stream0)
        del arg135_1
        buf301 = empty_strided_cuda((30, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, v_44, transpose_79, x_flat_13, v_t_x_12], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf300, (30, 128, 1024), (131072, 1024, 1), 0), out=buf301)
        buf302 = reinterpret_tensor(buf296, (30, 128, 1024), (131072, 1024, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [split_12, u_25, mixed_12], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 128, 12), (3200, 12, 1), 0), buf301, out=buf302)
        del buf301
        buf303 = empty_strided_cuda((30, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf300, buf302, buf20, buf303, 3840, 1024, stream=stream0)
        del buf300
        del buf302
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46, x_47], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf304 = extern_kernels.convolution(buf303, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg136_1
        del buf303
        buf305 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46, x_47, add_50, h_19], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf305, arg137_1, buf294, 3932160, stream=stream0)
        del arg137_1
        buf306 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg138_1, buf306, 16384, 9, stream=stream0)
        del arg138_1
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf306
        buf308 = buf298; del buf298  # reuse
        buf309 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf307, arg139_1, buf308, buf309, 960, 4096, stream=stream0)
        buf311 = reinterpret_tensor(buf294, (30, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf307, arg139_1, buf308, buf309, buf311, 3840, 1024, stream=stream0)
        del arg139_1
        buf312 = empty_strided_cuda((30, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, v_46, transpose_80, x_flat_14, v_t_x_13], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf311, (30, 128, 1024), (131072, 1024, 1), 0), out=buf312)
        buf313 = reinterpret_tensor(buf307, (30, 128, 1024), (131072, 1024, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [split_13, u_27, mixed_13], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 128, 12), (3200, 12, 1), 0), buf312, out=buf313)
        del buf312
        buf314 = empty_strided_cuda((30, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf311, buf313, buf20, buf314, 3840, 1024, stream=stream0)
        del buf311
        del buf313
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49, x_50], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf315 = extern_kernels.convolution(buf314, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg140_1
        del buf314
        buf316 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49, x_50, add_53, h_20], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf316, arg141_1, buf305, 3932160, stream=stream0)
        del arg141_1
        buf317 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg142_1, buf317, 16384, 9, stream=stream0)
        del arg142_1
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf317
        buf319 = buf309; del buf309  # reuse
        buf320 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf318, arg143_1, buf319, buf320, 960, 4096, stream=stream0)
        buf322 = reinterpret_tensor(buf305, (30, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf318, arg143_1, buf319, buf320, buf322, 3840, 1024, stream=stream0)
        del arg143_1
        buf323 = empty_strided_cuda((30, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, v_48, transpose_81, x_flat_15, v_t_x_14], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf322, (30, 128, 1024), (131072, 1024, 1), 0), out=buf323)
        buf324 = reinterpret_tensor(buf318, (30, 128, 1024), (131072, 1024, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [split_14, u_29, mixed_14], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 128, 12), (3200, 12, 1), 0), buf323, out=buf324)
        del buf323
        buf325 = empty_strided_cuda((30, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf322, buf324, buf20, buf325, 3840, 1024, stream=stream0)
        del buf322
        del buf324
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52, x_53], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf326 = extern_kernels.convolution(buf325, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg144_1
        del buf325
        buf327 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52, x_53, add_56, h_21], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf327, arg145_1, buf316, 3932160, stream=stream0)
        del arg145_1
        buf328 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg146_1, buf328, 16384, 9, stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf327, buf328, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf328
        buf330 = buf320; del buf320  # reuse
        buf331 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf329, arg147_1, buf330, buf331, 960, 4096, stream=stream0)
        buf333 = reinterpret_tensor(buf316, (30, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_10.run(buf329, arg147_1, buf330, buf331, buf333, 3840, 1024, stream=stream0)
        del arg147_1
        buf334 = empty_strided_cuda((30, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, v_50, transpose_82, x_flat_16, v_t_x_15], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf333, (30, 128, 1024), (131072, 1024, 1), 0), out=buf334)
        buf335 = reinterpret_tensor(buf329, (30, 128, 1024), (131072, 1024, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [split_15, u_31, mixed_15], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (30, 128, 12), (3200, 12, 1), 0), buf334, out=buf335)
        del buf334
        buf336 = empty_strided_cuda((30, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_11.run(buf333, buf335, buf20, buf336, 3840, 1024, stream=stream0)
        del buf20
        del buf333
        del buf335
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf337 = extern_kernels.convolution(buf336, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (30, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg148_1
        del buf336
        buf338 = buf331; del buf331  # reuse
        buf339 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63.run(buf337, arg149_1, buf327, buf338, buf339, 960, 4096, stream=stream0)
        buf341 = buf337; del buf337  # reuse
        buf342 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42, input_43], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_64.run(buf342, arg149_1, buf327, buf338, buf339, arg150_1, arg151_1, 3932160, stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del buf327
        del buf338
        del buf339
        buf343 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_65.run(arg152_1, buf343, 512, 9, stream=stream0)
        del arg152_1
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten.convolution]
        buf344 = extern_kernels.convolution(buf342, buf343, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (30, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
        del buf342
        del buf343
        buf345 = empty_strided_cuda((30, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_66.run(buf344, arg153_1, buf345, 120, 1024, stream=stream0)
        del arg153_1
        del buf344
    return (buf345, )


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
        partition0_args = [arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1]
        del arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1
        (buf345,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf345, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.int64)
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
    arg25_1 = rand_strided((30, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
