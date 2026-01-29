# AOT ID: ['1_inference']
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



# kernel path: ./.compile_cache/7g/c7gmdiqifcwcixyz37xahii7mcx4pkahlz57zcfj6akqnhyegv7s.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h => convolution
# Graph fragment:
#   %arg25_1 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %convolution : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 196608, 'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
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
#   %convolution : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/e2/ce2rg3zqmtyskxvibs3lz2v24nbi27sujcolflydb7g3oktzq5kw.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h => convolution
# Graph fragment:
#   %buf2 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf2]
#   %arg24_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %convolution : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg25_1, %arg23_1, %arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437696}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
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
#   %convolution_1 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/ri/cri3vgsstkeqp3nk32xpfsltwtxnhvc3a5qzk4aucxk4pst4jmfi.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution_1
#   x_norm => var_mean, view
# Graph fragment:
#   %buf5 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %convolution_1 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [6, 32, 4, 1024]), kwargs = {})
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
    size_hints={'x': 256, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072, 'r0_': 3146240}}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: ./.compile_cache/ho/choebbcpt735ht4ahtrwkx5dh7koaq6oureaq6mlkyhdvmgvpoa5.py
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
#   %arg0_1 : Tensor "f32[6][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %unsqueeze : Tensor "f32[6, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, -9.210340371976184), kwargs = {})
#   %div : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 128), kwargs = {})
#   %exp : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 0), kwargs = {})
#   %mul_1 : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %cos : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %sin : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   return %cat
triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5 = async_compile.triton('triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
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


# kernel path: ./.compile_cache/cg/ccgaqdvcwmtqzu2manm4ygcqfayxyczscxseviolcx3p65lnjebu.py
# Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.addmm]
# Source node to ATen node mapping:
#   arange => iota
#   args => mul_1
#   cos => cos
#   emb => cat
#   freqs => exp
#   getitem => unsqueeze
#   getitem_1 => unsqueeze_1
#   input_1 => constant_pad_nd_default_1
#   mul => mul
#   sin => sin
#   truediv => div
# Graph fragment:
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0" = PlaceHolder[target=cat]
#   %unsqueeze : Tensor "f32[6, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, -9.210340371976184), kwargs = {})
#   %div : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 128), kwargs = {})
#   %exp : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 0), kwargs = {})
#   %mul_1 : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %cos : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %sin : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   %constant_pad_nd_default_1 : Tensor "f32[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat, [0, 0, 0, 2]), kwargs = {})
#   return %constant_pad_nd_default_1
triton_poi_fused_addmm_arange_cat_cos_div_exp_mul_sin_unsqueeze_6 = async_compile.triton('triton_poi_fused_addmm_arange_cat_cos_div_exp_mul_sin_unsqueeze_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_arange_cat_cos_div_exp_mul_sin_unsqueeze_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_arange_cat_cos_div_exp_mul_sin_unsqueeze_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 6, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2 & xmask, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/cc/cccisurryypnowczdn2v3zad4f6krhebixs5ygleyy76fwivatqj.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_39, slice_tensor_1
#   input_2 => mul_2, sigmoid
# Graph fragment:
#   %mm_default_39 : Tensor "f32[8, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_default_39]
#   %arg2_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add_tensor_39 : Tensor "f32[8, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_39, %arg2_1), kwargs = {})
#   %slice_tensor_1 : Tensor "f32[6, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_tensor_39, 0, 0, -2), kwargs = {})
#   %sigmoid : Tensor "f32[6, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%slice_tensor_1,), kwargs = {})
#   %mul_2 : Tensor "f32[6, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_tensor_1, %sigmoid), kwargs = {})
#   return %mul_2
triton_poi_fused_addmm_silu_7 = async_compile.triton('triton_poi_fused_addmm_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 77824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/wn/cwna73gwa4fvuxppglx6ddzvntngwc6u7cq7w4xzjurjho33o65w.py
# Topologically Sorted Source Nodes: [s_emb], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   s_emb => embedding
# Graph fragment:
#   %arg6_1 : Tensor "i64[6][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg5_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg6_1), kwargs = {})
#   return %embedding
triton_poi_fused_embedding_8 = async_compile.triton('triton_poi_fused_embedding_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
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


# kernel path: ./.compile_cache/zv/czvqc5g5spvg5of7cyqpxazrbcut3qzv5vddynwcw3pjmpwi4g43.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_4 => add_tensor_38
#   input_5 => mul_3, sigmoid_1
# Graph fragment:
#   %mm_default_38 : Tensor "f32[6, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_38]
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %add_tensor_38 : Tensor "f32[6, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_38, %arg8_1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[6, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_38,), kwargs = {})
#   %mul_3 : Tensor "f32[6, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_38, %sigmoid_1), kwargs = {})
#   return %mul_3
triton_poi_fused_addmm_silu_9 = async_compile.triton('triton_poi_fused_addmm_silu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 38912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
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


# kernel path: ./.compile_cache/gv/cgvcxw5ymcf3xhqpvafb6rvmp3henbflfshrpvexoqqxv2y6e7uu.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_7 => add_tensor_37
#   input_8 => mul_4, sigmoid_2
# Graph fragment:
#   %mm_default_37 : Tensor "f32[6, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_37]
#   %arg12_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_tensor_37 : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_37, %arg12_1), kwargs = {})
#   %sigmoid_2 : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_37,), kwargs = {})
#   %mul_4 : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_37, %sigmoid_2), kwargs = {})
#   return %mul_4
triton_poi_fused_addmm_silu_10 = async_compile.triton('triton_poi_fused_addmm_silu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_silu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 19456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_silu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
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


# kernel path: ./.compile_cache/b5/cb5hr24cxk22skhstsknsauur46ganetzlfs5gzrb4ez5xe6v2mq.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convolution_1
#   x_norm => add, mul_7, rsqrt, sub, var_mean, view
# Graph fragment:
#   %buf5 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %getitem_1 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=buf7]
#   %convolution_1 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   return %mul_7
triton_poi_fused_convolution_native_group_norm_11 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3146240, 'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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


# kernel path: ./.compile_cache/gj/cgjxhz3qhglkjsnk772bpw4hz5iq45v2njb6qasu4qo4cztmqfei.py
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
#   %mul_7 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0" = PlaceHolder[target=mul_7]
#   %bmm_1 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %addmm_5 : Tensor "f32[6, 3200][3200, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %convolution_1 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [6, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [6, 128, -1]), kwargs = {})
#   %add_1 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [6, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "f32[6, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [6, 128, 1, 1]), kwargs = {})
#   %add_2 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   return %mul_8
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 6294528, 'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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


# kernel path: ./.compile_cache/pf/cpfswdscdiwz3hgs5co5xzzuup2rmjxewg7olfbcpq5qrrkxph5u.py
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
#   %buf26 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf26]
#   %arg29_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %convolution : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution]
#   %convolution_1 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg26_1, %arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_1, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_7 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [6, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [6, 128, -1]), kwargs = {})
#   %add_1 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [6, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "f32[6, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [6, 128, 1, 1]), kwargs = {})
#   %add_2 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   %convolution_2 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_8, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution), kwargs = {})
#   %sigmoid_6 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_9 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_6), kwargs = {})
#   return %mul_9
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12583424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
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


# kernel path: ./.compile_cache/2d/c2dod4yxgz7glzupylrrdqxk35l45v2wj5nrbljbcaoubtmhonfq.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   h_3 => convolution_5
# Graph fragment:
#   %arg34_1 : Tensor "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg34_1]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf40
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: ./.compile_cache/ta/ctaabdzs2edxllku6edfezbehxeigoqrvxvmjeqor2q2a3bne3c5.py
# Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   q => permute_12
#   q_1 => clone
#   view_10 => view_14
# Graph fragment:
#   %buf41 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf41]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_14 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_5, [6, 256, 256]), kwargs = {})
#   %permute_12 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_12,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_convolution_transpose_view_15 = async_compile.triton('triton_poi_fused_clone_convolution_transpose_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_transpose_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4719616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_transpose_view_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/xk/cxkpv65tz6agozuoiu536v2xe6jg2hiqloq4khy3lgiimmleb4z5.py
# Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_1
#   q_1 => view_16
#   q_2 => permute_15
#   view_11 => view_17
# Graph fragment:
#   %mm : Tensor "f32[1536, 256][256, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_16 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [6, 256, 256]), kwargs = {})
#   %view_17 : Tensor "f32[6, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [6, 256, 4, 64]), kwargs = {})
#   %permute_15 : Tensor "f32[6, 4, 256, 64][65536, 64, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "f32[6, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_clone_transpose_view_16 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/25/c25luoaxug7rssomn4ecxw6c52fj7znlu2hulft35jn5dipv6uar.py
# Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   k_1 => permute_16
#   kv => unsqueeze_2
#   matmul => clone_2
#   transpose_6 => permute_18
#   view_12 => view_18
# Graph fragment:
#   %mm_1 : Tensor "f32[6, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "f32[6, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_18 : Tensor "f32[6, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_10, [6, 1, 4, 64]), kwargs = {})
#   %permute_16 : Tensor "f32[6, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %permute_18 : Tensor "f32[6, 4, 64, 1][512, 64, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_16, [0, 1, 3, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[6, 4, 64, 1][256, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused_clone_split_transpose_unsqueeze_view_17 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/6j/c6jt3ac2oomorhrmr6moi63ipf4oemn22wvwhqbhwxcwefmea6my.py
# Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div_1, exp_1, sum_1
#   matmul => view_22
# Graph fragment:
#   %bmm_4 : Tensor "f32[24, 256, 1][256, 1, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %view_22 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_4, [6, 4, 256, 1]), kwargs = {})
#   %mul_tensor_38 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, 1), kwargs = {})
#   %amax_default_19 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_38, [-1], True), kwargs = {})
#   %sub_tensor_19 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_38, %amax_default_19), kwargs = {})
#   %mul_tensor_39 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_19, 0.125), kwargs = {})
#   %exp_1 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_39,), kwargs = {})
#   %sum_1 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[6, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_1), kwargs = {})
#   return %expand_2
triton_poi_fused__softmax_amax_mul_sub_view_18 = async_compile.triton('triton_poi_fused__softmax_amax_mul_sub_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_amax_mul_sub_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_amax_mul_sub_view_18(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
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


# kernel path: ./.compile_cache/cv/ccvoiolahsxhtjdxsjgwnshr2aphpf6njnrveltmraxvqrkxqdup.py
# Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   kv => unsqueeze_2
#   matmul_1 => clone_3
#   v_5 => permute_17
#   view_13 => view_19
# Graph fragment:
#   %mm_1 : Tensor "f32[6, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "f32[6, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_19 : Tensor "f32[6, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_11, [6, 1, 4, 64]), kwargs = {})
#   %permute_17 : Tensor "f32[6, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[6, 4, 1, 64][256, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_split_transpose_unsqueeze_view_19 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/gb/cgbkcwx6o5hltrwomj5vflten5g4b62qsy4yzhubwkcf7ynq6fpc.py
# Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => view_25
#   out_4 => clone_4
#   transpose_7 => permute_19
# Graph fragment:
#   %bmm_5 : Tensor "f32[24, 256, 64][16384, 64, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %view_25 : Tensor "f32[6, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [6, 4, 256, 64]), kwargs = {})
#   %permute_19 : Tensor "f32[6, 256, 4, 64][65536, 64, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_25, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "f32[6, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_transpose_view_20 = async_compile.triton('triton_poi_fused_clone_transpose_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/o2/co2fjy7gpoclgzmfx6xgdtnylerqne42jf73vtg6eiwtv37yaxis.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_36, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
# Graph fragment:
#   %buf41 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf41]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %mm_default_36 : Tensor "f32[1536, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_36]
#   %arg39_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   return %add_8
triton_poi_fused_add_addmm_convolution_transpose_view_21 = async_compile.triton('triton_poi_fused_add_addmm_convolution_transpose_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_transpose_view_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6293504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_transpose_view_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/sw/cswkuoup4qz74q3rmbsudkzgsolnjcby3hikqbdp2wjyuihqdxts.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_36, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
# Graph fragment:
#   %arg40_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg40_1]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf53
triton_poi_fused_add_addmm_convolution_transpose_view_22 = async_compile.triton('triton_poi_fused_add_addmm_convolution_transpose_view_22', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_transpose_view_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_transpose_view_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: ./.compile_cache/6f/c6ffqhutgc6iajuikckfrfuke5jbmotidabjzarptyddogu3sg6b.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_36, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
#   x_norm_2 => var_mean_2, view_30
# Graph fragment:
#   %buf54 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf54]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_13,%buf56
triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23 = async_compile.triton('triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072, 'r0_': 1573888}}
)
@triton.jit
def triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: ./.compile_cache/ip/cipgnxuxkimbuatrdqfmix4swjmgostplgiqwdlsshvyf7trcnm5.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_36, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
#   x_7 => convolution_6
#   x_norm_2 => add_9, mul_14, rsqrt_2, sub_3, var_mean_2, view_30
# Graph fragment:
#   %buf54 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf54]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %getitem_13 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf56 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=buf56]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   return %mul_14
triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1573888, 'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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


# kernel path: ./.compile_cache/5i/c5il5pweext6kyrn5ek6zgaikbhgp7tl4fnqrcuiusbjzo7h5c3c.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   h_3 => convolution_5
#   input_16 => add_tensor_36, view_28
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
#   %mul_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0" = PlaceHolder[target=mul_14]
#   %bmm_7 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %addmm_7 : Tensor "f32[6, 6400][6400, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [6, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [6, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [6, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [6, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   return %mul_15
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3151872, 'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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


# kernel path: ./.compile_cache/he/cheuswzp34q6pxjsdwxcmzcvy6kvfipuzmifc4jjadlbrg2vfauq.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   input_16 => add_tensor_36, view_28
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
#   %buf66 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf66]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf41 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf41]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [6, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [6, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [6, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [6, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %view_37 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_16, [6, 256, 256]), kwargs = {})
#   %permute_23 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_37, [0, 2, 1]), kwargs = {})
#   %clone_6 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_6
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6293504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/o4/co4jcwndhe6h45m5pewvczmnqhfbdaaqllzmqmjpzabeu2kupaoo.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   input_16 => add_tensor_36, view_28
#   input_18 => add_tensor_34, view_51
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
#   %buf66 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf66]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf41 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf41]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %mm_default_34 : Tensor "f32[1536, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg47_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [6, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [6, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [6, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [6, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %add_tensor_34 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg47_1), kwargs = {})
#   %view_51 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [6, 256, 256]), kwargs = {})
#   %permute_32 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_51, [0, 2, 1]), kwargs = {})
#   %view_52 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_32, [6, 256, 16, 16]), kwargs = {})
#   %add_13 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %view_52), kwargs = {})
#   return %add_13
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7867392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/k6/ck65m53opwprehds4l3lgoc7dgrnnjvfldlkwjmlsqgzf2lfjmbi.py
# Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13, add_13, h_5], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_13 => add_17
#   add_9 => add_12
#   h_3 => convolution_5
#   h_4 => mul_16, sigmoid_10
#   h_5 => mul_20, sigmoid_12
#   input_16 => add_tensor_36, view_28
#   input_18 => add_tensor_34, view_51
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
#   %buf88 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf88]
#   %arg51_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %buf66 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf66]
#   %arg43_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf41 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf41]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %add_17 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_17]
#   %convolution_5 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_12, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_36 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %arg39_1), kwargs = {})
#   %view_28 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [6, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [6, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convolution_6 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_8, %arg40_1, %arg41_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_30 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_3 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_9 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [6, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [6, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [6, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [6, 256, 1, 1]), kwargs = {})
#   %add_11 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convolution_7 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_15, %arg42_1, %arg43_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %sigmoid_10 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_10), kwargs = {})
#   %add_tensor_34 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg47_1), kwargs = {})
#   %view_51 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [6, 256, 256]), kwargs = {})
#   %permute_32 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_51, [0, 2, 1]), kwargs = {})
#   %view_52 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_32, [6, 256, 16, 16]), kwargs = {})
#   %add_13 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %view_52), kwargs = {})
#   %convolution_8 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_13, %arg48_1, %arg49_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_53 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_8, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_3 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_5 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_53, %getitem_20), kwargs = {})
#   %add_14 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_19, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_18 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %view_54 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_18, [6, 256, 16, 16]), kwargs = {})
#   %view_58 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_54, [6, 256, -1]), kwargs = {})
#   %add_15 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_58, %bmm_11), kwargs = {})
#   %view_59 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_15, [6, 256, 16, 16]), kwargs = {})
#   %view_57 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_23, [6, 256, 1, 1]), kwargs = {})
#   %add_16 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_59, %view_57), kwargs = {})
#   %sigmoid_11 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_16,), kwargs = {})
#   %mul_19 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sigmoid_11), kwargs = {})
#   %convolution_9 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_19, %arg50_1, %arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_17 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %mul_16), kwargs = {})
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   return %add_17,%mul_20
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7867392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/7l/c7lah4xmwr62jf54mh45nwcxlep4ewfixr5lelzayvcdblplm3o5.py
# Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
# Graph fragment:
#   %arg52_1 : Tensor "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg52_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf91
triton_poi_fused_convolution_silu_29 = async_compile.triton('triton_poi_fused_convolution_silu_29', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: ./.compile_cache/e4/ce4la3yl26ucjcevjdfzwf6xpbvrsigxrlr6hhnwtrl5mkzl5sun.py
# Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   q_6 => permute_34
#   q_7 => clone_12
#   view_30 => view_60
# Graph fragment:
#   %buf92 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf92]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_60 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_10, [6, 512, 64]), kwargs = {})
#   %permute_34 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_60, [0, 2, 1]), kwargs = {})
#   %clone_12 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_12
triton_poi_fused_clone_convolution_silu_transpose_view_30 = async_compile.triton('triton_poi_fused_clone_convolution_silu_transpose_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_silu_transpose_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2361344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_silu_transpose_view_30(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/il/cilua77ka3qmcpn6pukzeoqtrrpwtx4dragqqxexb2bpgwu3vz54.py
# Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_4 => clone_13
#   q_7 => view_62
#   q_8 => permute_37
#   view_31 => view_63
# Graph fragment:
#   %mm_4 : Tensor "f32[384, 512][512, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %view_62 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [6, 64, 512]), kwargs = {})
#   %view_63 : Tensor "f32[6, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_62, [6, 64, 8, 64]), kwargs = {})
#   %permute_37 : Tensor "f32[6, 8, 64, 64][32768, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_63, [0, 2, 1, 3]), kwargs = {})
#   %clone_13 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_13
triton_poi_fused__unsafe_view_clone_transpose_view_31 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/q3/cq3imwifzttztbiyndrm6uvr6muzpddelqkfqi3lesn2sibx5vn5.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   k_5 => permute_38
#   kv_2 => unsqueeze_4
#   matmul_4 => clone_14
#   transpose_22 => permute_40
#   view_32 => view_64
# Graph fragment:
#   %mm_5 : Tensor "f32[6, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "f32[6, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_64 : Tensor "f32[6, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_24, [6, 1, 8, 64]), kwargs = {})
#   %permute_38 : Tensor "f32[6, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_64, [0, 2, 1, 3]), kwargs = {})
#   %permute_40 : Tensor "f32[6, 8, 64, 1][1024, 64, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_38, [0, 1, 3, 2]), kwargs = {})
#   %clone_14 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_14
triton_poi_fused_clone_split_transpose_unsqueeze_view_32 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/un/cunh3s3p6k3cl5wwdmmzfaglw24neavwl2hifbrxsiir7hvv6xkz.py
# Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_5 => div_3, exp_3, sum_3
#   matmul_4 => view_68
# Graph fragment:
#   %bmm_12 : Tensor "f32[48, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_12]
#   %view_68 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_12, [6, 8, 64, 1]), kwargs = {})
#   %mul_tensor_34 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_68, 1), kwargs = {})
#   %amax_default_17 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_34, [-1], True), kwargs = {})
#   %sub_tensor_17 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_34, %amax_default_17), kwargs = {})
#   %mul_tensor_35 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_17, 0.125), kwargs = {})
#   %exp_3 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_35,), kwargs = {})
#   %sum_3 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [-1], True), kwargs = {})
#   %div_3 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_3, %sum_3), kwargs = {})
#   return %expand_10
triton_poi_fused__softmax_amax_mul_sub_view_33 = async_compile.triton('triton_poi_fused__softmax_amax_mul_sub_view_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_amax_mul_sub_view_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_amax_mul_sub_view_33(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
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


# kernel path: ./.compile_cache/rm/crmorsvumsn6fn3lvjfmycqfd2oyxdnszlarqtuww4ycpzdu6ls7.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   kv_2 => unsqueeze_4
#   matmul_5 => clone_15
#   v_13 => permute_39
#   view_33 => view_65
# Graph fragment:
#   %mm_5 : Tensor "f32[6, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "f32[6, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_65 : Tensor "f32[6, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_25, [6, 1, 8, 64]), kwargs = {})
#   %permute_39 : Tensor "f32[6, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_65, [0, 2, 1, 3]), kwargs = {})
#   %clone_15 : Tensor "f32[6, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_15
triton_poi_fused_clone_split_transpose_unsqueeze_view_34 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + 1024*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/dw/cdwerdykkqzebobmfar2jmq45zytnrvfyrio3o4sjqqcmf7wyqly.py
# Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_5 => view_71
#   out_12 => clone_16
#   transpose_23 => permute_41
# Graph fragment:
#   %bmm_13 : Tensor "f32[48, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %view_71 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_13, [6, 8, 64, 64]), kwargs = {})
#   %permute_41 : Tensor "f32[6, 64, 8, 64][32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_71, [0, 2, 1, 3]), kwargs = {})
#   %clone_16 : Tensor "f32[6, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_16
triton_poi_fused_clone_transpose_view_35 = async_compile.triton('triton_poi_fused_clone_transpose_view_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/kq/ckqpnjjgwbz4rvlafvl65g2o7xiqdgku22ogwye5ss7ysdphgs6t.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_33, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
# Graph fragment:
#   %buf92 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf92]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %mm_default_33 : Tensor "f32[384, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg57_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   return %add_18
triton_poi_fused_add_addmm_convolution_silu_transpose_view_36 = async_compile.triton('triton_poi_fused_add_addmm_convolution_silu_transpose_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_silu_transpose_view_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3149824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_silu_transpose_view_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/ia/ciahfircwgqy3fydtkays7ix5ka2nvdtr2ajhj4g2f4r6y555hu5.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_33, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
# Graph fragment:
#   %arg58_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg58_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf104
triton_poi_fused_add_addmm_convolution_silu_transpose_view_37 = async_compile.triton('triton_poi_fused_add_addmm_convolution_silu_transpose_view_37', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_silu_transpose_view_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 18874368, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_silu_transpose_view_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: ./.compile_cache/i7/ci7n7gc4hiwog7wvpopmj7t3eqegcwo3vjqwy64cj3wdebwfjww2.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_33, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
#   x_norm_4 => var_mean_4, view_76
# Graph fragment:
#   %buf105 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf105]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [6, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_27,%buf107
triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38 = async_compile.triton('triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072, 'r0_': 788480}}
)
@triton.jit
def triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: ./.compile_cache/zz/czzaqpcmghnd73m3bbfrdon7clcazzoboansfdg6jpnpkvibegsp.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_33, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
#   x_15 => convolution_11
#   x_norm_4 => add_19, mul_22, rsqrt_4, sub_7, var_mean_4, view_76
# Graph fragment:
#   %buf105 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf105]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %getitem_27 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf107 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=buf107]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [6, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   return %mul_22
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 788480, 'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
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


# kernel path: ./.compile_cache/57/c57qq6dox7hy6hti2lptfdwhvlukqzrrzrwjsxbwgtnji4ngwo4m.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   input_20 => add_tensor_33, view_74
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
#   %mul_22 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0" = PlaceHolder[target=mul_22]
#   %bmm_15 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0" = PlaceHolder[target=bmm_15]
#   %addmm_9 : Tensor "f32[6, 12800][12800, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [6, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [6, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [6, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [6, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[6, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [6, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   return %mul_23
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1585152, 'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
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


# kernel path: ./.compile_cache/yo/cyokzylh3tyyk3roalrp5o4si7vwcuk5qfwau67fzvt33ruczgmp.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   input_20 => add_tensor_33, view_74
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
#   %buf117 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf117]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf92 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf92]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [6, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [6, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [6, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [6, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[6, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [6, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %view_83 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_24, [6, 512, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_83, [0, 2, 1]), kwargs = {})
#   %clone_18 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_45,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_18
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3149824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/p6/cp6smf77e4dskx47gcvuob5w2t3g7cymfpodu3vlhd4aqyu5fojy.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   input_20 => add_tensor_33, view_74
#   input_22 => add_tensor_31, view_97
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
#   %buf117 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf117]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf92 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf92]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %mm_default_31 : Tensor "f32[384, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_31]
#   %arg65_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [6, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [6, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [6, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [6, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[6, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [6, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %add_tensor_31 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_31, %arg65_1), kwargs = {})
#   %view_97 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_31, [6, 64, 512]), kwargs = {})
#   %permute_54 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_97, [0, 2, 1]), kwargs = {})
#   %view_98 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_54, [6, 512, 8, 8]), kwargs = {})
#   %add_23 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %view_98), kwargs = {})
#   return %add_23
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_42 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3938304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/6c/c6cyyihutasiiwkwk4oelb6p2rt5tzlbw66dtggqpevqyhvv6zwm.py
# Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_17 => add_22
#   add_21 => add_27
#   h_5 => mul_20, sigmoid_12
#   h_6 => convolution_10
#   h_7 => mul_24, sigmoid_14
#   h_8 => mul_28, sigmoid_16
#   input_20 => add_tensor_33, view_74
#   input_22 => add_tensor_31, view_97
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
#   %buf138 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf138]
#   %arg69_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %buf117 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf117]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %buf92 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf92]
#   %arg53_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %add_27 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %getitem_39 : Tensor "f32[6, 64, 1][64, 1, 384]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf141 : Tensor "f32[6, 64, 1][64, 1, 384]cuda:0" = PlaceHolder[target=buf141]
#   %arg70_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg70_1]
#   %arg71_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg71_1]
#   %sigmoid_12 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_12), kwargs = {})
#   %convolution_10 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_20, %arg52_1, %arg53_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_tensor_33 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg57_1), kwargs = {})
#   %view_74 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [6, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [6, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convolution_11 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_18, %arg58_1, %arg59_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_76 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_11, [6, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_7 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_19 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_22 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [6, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [6, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [6, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "f32[6, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [6, 512, 1, 1]), kwargs = {})
#   %add_21 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convolution_12 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %arg60_1, %arg61_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %sigmoid_14 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_24 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_14), kwargs = {})
#   %add_tensor_31 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_31, %arg65_1), kwargs = {})
#   %view_97 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_31, [6, 64, 512]), kwargs = {})
#   %permute_54 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_97, [0, 2, 1]), kwargs = {})
#   %view_98 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_54, [6, 512, 8, 8]), kwargs = {})
#   %add_23 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %view_98), kwargs = {})
#   %convolution_13 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_23, %arg66_1, %arg67_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_99 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_13, [6, 32, 16, 64]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_99, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_5 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_9 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_99, %getitem_34), kwargs = {})
#   %add_24 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_33, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %mul_26 : Tensor "f32[6, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_5), kwargs = {})
#   %view_100 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_26, [6, 512, 8, 8]), kwargs = {})
#   %view_104 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_100, [6, 512, -1]), kwargs = {})
#   %add_25 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_104, %bmm_19), kwargs = {})
#   %view_105 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_25, [6, 512, 8, 8]), kwargs = {})
#   %view_103 : Tensor "f32[6, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_37, [6, 512, 1, 1]), kwargs = {})
#   %add_26 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_105, %view_103), kwargs = {})
#   %sigmoid_15 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_26,), kwargs = {})
#   %mul_27 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %sigmoid_15), kwargs = {})
#   %convolution_14 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_27, %arg68_1, %arg69_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_27 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %mul_24), kwargs = {})
#   %sigmoid_16 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [6, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %clone_24 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_56,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_24, %getitem_39), kwargs = {})
#   %add_28 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_29 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_6), kwargs = {})
#   %mul_30 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %arg70_1), kwargs = {})
#   %add_29 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %arg71_1), kwargs = {})
#   return %add_27,%getitem_39,%buf141,%add_29
triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_43 = async_compile.triton('triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 5515264}}
)
@triton.jit
def triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: ./.compile_cache/sw/cswnoxvfyneg5ysf5tc6xn2gztubbr7t5hc4epzkblv4q7t76psq.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   matmul_8 => clone_25
#   q_12 => select
#   qkv => add_tensor_30, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
# Graph fragment:
#   %mm_default_30 : Tensor "f32[384, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_30]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_30 : Tensor "f32[384, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_30, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[6, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_30, [6, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[6, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [6, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 6, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select : Tensor "f32[6, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 0), kwargs = {})
#   %clone_25 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_16,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_25
triton_poi_fused_addmm_clone_permute_select_view_44 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_view_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2361344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_view_44(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/2e/c2ehzuavlqohivuzcnroa7g27tjwxgntvzfb4fkkkc45owsuq5wt.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   k_8 => select_1
#   matmul_8 => clone_26
#   qkv => add_tensor_30, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   transpose_35 => permute_59
# Graph fragment:
#   %mm_default_30 : Tensor "f32[384, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_30]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_30 : Tensor "f32[384, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_30, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[6, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_30, [6, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[6, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [6, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 6, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_1 : Tensor "f32[6, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 1), kwargs = {})
#   %permute_59 : Tensor "f32[6, 8, 64, 64][98304, 64, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_26 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_17,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_26
triton_poi_fused_addmm_clone_permute_select_transpose_view_45 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_transpose_view_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_transpose_view_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 788480, 'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_transpose_view_45(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
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


# kernel path: ./.compile_cache/uu/cuuaqsafvpmcuy3fwzsodun5f3xtc7ljrrnuhbumbtlbr6ioohia.py
# Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_9 => div_5, exp_5, sum_5
#   matmul_8 => view_112
# Graph fragment:
#   %bmm_20 : Tensor "f32[48, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %amax_default_15 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 3072]cuda:0" = PlaceHolder[target=amax_default_15]
#   %sum_5 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 3072]cuda:0" = PlaceHolder[target=sum_5]
#   %view_112 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_20, [6, 8, 64, 64]), kwargs = {})
#   %mul_tensor_30 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_112, 1), kwargs = {})
#   %amax_default_15 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_30, [-1], True), kwargs = {})
#   %sub_tensor_15 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_30, %amax_default_15), kwargs = {})
#   %mul_tensor_31 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_15, 0.125), kwargs = {})
#   %exp_5 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_31,), kwargs = {})
#   %sum_5 : Tensor "f32[6, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1], True), kwargs = {})
#   %div_5 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_5), kwargs = {})
#   return %amax_default_15,%sum_5,%expand_18
triton_per_fused__softmax_amax_mul_sub_view_46 = async_compile.triton('triton_per_fused__softmax_amax_mul_sub_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_amax_mul_sub_view_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 2359296}}
)
@triton.jit
def triton_per_fused__softmax_amax_mul_sub_view_46(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3072
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


# kernel path: ./.compile_cache/i6/ci6joducozsb6jg22s222vs6ny6vjtml5xf7grsgqqvjp2wshlb6.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   out_20 => clone_27
#   qkv => add_tensor_30, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   v_20 => select_2
# Graph fragment:
#   %mm_default_30 : Tensor "f32[384, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_30]
#   %arg73_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %add_tensor_30 : Tensor "f32[384, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_30, %arg73_1), kwargs = {})
#   %view_108 : Tensor "f32[6, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_30, [6, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "f32[6, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [6, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "f32[3, 6, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_2 : Tensor "f32[6, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 2), kwargs = {})
#   %clone_27 : Tensor "f32[6, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_27
triton_poi_fused_addmm_clone_permute_select_view_47 = async_compile.triton('triton_poi_fused_addmm_clone_permute_select_view_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_clone_permute_select_view_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2361344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_clone_permute_select_view_47(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: ./.compile_cache/el/cellflr4svsaimiqlfwjhtcdnkk474uolppcj5oq43ruxfyhh7b5.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_8 => mul_28, sigmoid_16
#   out_22 => add_tensor_29, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %add_27 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %mm_default_29 : Tensor "f32[384, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_29]
#   %arg75_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %getitem_41 : Tensor "f32[6, 64, 1][64, 1, 384]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf156 : Tensor "f32[6, 64, 1][64, 1, 384]cuda:0" = PlaceHolder[target=buf156]
#   %arg76_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg76_1]
#   %arg77_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %sigmoid_16 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [6, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_29 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_29, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_29, [6, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   return %getitem_41,%buf156,%add_32
triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_48 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 3151872}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: ./.compile_cache/az/cazsbjcythtsqi37avxegodc2kuonln3fuw2jqpi4jlfbru7ddxg.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
# Source node to ATen node mapping:
#   h_8 => mul_28, sigmoid_16
#   h_9 => convolution_15
#   out_22 => add_tensor_29, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %arg78_1 : Tensor "f32[512, 256, 4, 4][4096, 16, 4, 1]cuda:0" = PlaceHolder[target=arg78_1]
#   %sigmoid_16 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [6, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_29 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_29, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_29, [6, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [6, 512, 8, 8]), kwargs = {})
#   %convolution_15 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_119, %arg78_1, %arg79_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf160
triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_49 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_49', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 16777216, 'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_49(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: ./.compile_cache/tj/ctjuvdbyebks52wpu5c4zudq7upsf23beirrbvj35uydj2l6xqae.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   gate => unsqueeze_7
#   h_10 => add_33
#   h_16_gated => mul_35
#   h_8 => mul_28, sigmoid_16
#   h_9 => convolution_15
#   input_26 => add_tensor_27
#   input_27 => sigmoid_18
#   out_22 => add_tensor_29, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, clone_29, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   unsqueeze_4 => unsqueeze_6
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %buf161 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf161]
#   %arg79_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %mul_20 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=mul_20]
#   %mm_default_27 : Tensor "f32[6, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_27]
#   %arg83_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %sigmoid_16 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[6, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_16), kwargs = {})
#   %view_106 : Tensor "f32[6, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [6, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %add_tensor_29 : Tensor "f32[384, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_29, %arg75_1), kwargs = {})
#   %view_118 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_29, [6, 64, 512]), kwargs = {})
#   %add_30 : Tensor "f32[6, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %clone_29 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_30,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_29, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_29, %getitem_41), kwargs = {})
#   %add_31 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[6, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg76_1), kwargs = {})
#   %add_32 : Tensor "f32[6, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg77_1), kwargs = {})
#   %permute_62 : Tensor "f32[6, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[6, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [6, 512, 8, 8]), kwargs = {})
#   %convolution_15 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_119, %arg78_1, %arg79_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_tensor_27 : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_27, %arg83_1), kwargs = {})
#   %sigmoid_18 : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_27,), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[6, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_18, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[6, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %mul_35 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_7), kwargs = {})
#   %add_33 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %mul_35), kwargs = {})
#   return %add_33
triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50 = async_compile.triton('triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3145728, 'x': 3153920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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


# kernel path: ./.compile_cache/r3/cr3db6cy4vl4kyyfzd5u7g7ozi3zw6dmfoprhxv7uqhl26hjtawc.py
# Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
# Source node to ATen node mapping:
#   q_13 => permute_65
#   q_14 => mm_8, permute_66, view_121
#   view_52 => view_120
# Graph fragment:
#   %add_33 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %view_120 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_33, [6, 256, 256]), kwargs = {})
#   %permute_65 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_120, [0, 2, 1]), kwargs = {})
#   %view_121 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_65, [1536, 256]), kwargs = {})
#   %permute_66 : Tensor "f32[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg84_1, [1, 0]), kwargs = {})
#   %mm_8 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_121, %permute_66), kwargs = {})
#   return %buf166
triton_poi_fused_mm_t_transpose_view_51 = async_compile.triton('triton_poi_fused_mm_t_transpose_view_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_t_transpose_view_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_t_transpose_view_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256*x1 + 65536*(x0 // 256) + ((x0 % 256))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: ./.compile_cache/hg/chg7arh5ab36mrt3tjgwnznyx6gmcgu4jgaoasfgdu7ceu6y6xvq.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   input_28 => add_tensor_26, view_134
#   out_27 => view_135
#   transpose_44 => permute_74
#   x_22 => add_34
# Graph fragment:
#   %add_33 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_default_26 : Tensor "f32[1536, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_26]
#   %arg87_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %add_tensor_26 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_26, [6, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [6, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   return %add_34
triton_poi_fused_add_addmm_transpose_view_52 = async_compile.triton('triton_poi_fused_add_addmm_transpose_view_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_transpose_view_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1572864, 'x': 4719616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_transpose_view_52(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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


# kernel path: ./.compile_cache/wi/cwihghjjutbnhlyrkz7gvn3ulhcj52thqmdnliutv4nfhbvtpaqg.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   h_11 => mul_39, sigmoid_20
#   input_28 => add_tensor_26, view_134
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
#   %buf187 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf187]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %add_tensor_26 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_26, [6, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [6, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [6, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [6, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [6, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [6, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %view_143 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [6, 256, 256]), kwargs = {})
#   %permute_76 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_143, [0, 2, 1]), kwargs = {})
#   %clone_36 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_76,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_36
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1572864, 'x': 4719616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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


# kernel path: ./.compile_cache/hh/chhojp3btl63i7a7pz7dm2lmzprvebxuhldaawsqitqgsrpo5ing.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   h_11 => mul_39, sigmoid_20
#   input_28 => add_tensor_26, view_134
#   input_30 => add_tensor_25, view_157
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
#   %buf187 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf187]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_default_25 : Tensor "f32[1536, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_25]
#   %arg95_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg95_1]
#   %add_tensor_26 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_26, [6, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [6, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [6, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [6, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [6, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [6, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %add_tensor_25 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_25, %arg95_1), kwargs = {})
#   %view_157 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_25, [6, 256, 256]), kwargs = {})
#   %permute_85 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1]), kwargs = {})
#   %view_158 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_85, [6, 256, 16, 16]), kwargs = {})
#   %add_39 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %view_158), kwargs = {})
#   return %add_39
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1572864, 'x': 6293504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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


# kernel path: ./.compile_cache/e5/ce5q2tvqev4ytbaiaa2xzh6h3hc7la5b4lohrmd6eofbta42xtwn.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_38
#   add_31 => add_43
#   h_11 => mul_39, sigmoid_20
#   h_12 => mul_43, sigmoid_22
#   input_28 => add_tensor_26, view_134
#   input_30 => add_tensor_25, view_157
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
#   %buf209 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf209]
#   %arg99_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg99_1]
#   %buf187 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf187]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %add_33 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_33]
#   %add_43 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %add_tensor_26 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %arg87_1), kwargs = {})
#   %view_134 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_26, [6, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [6, 256, 16, 16]), kwargs = {})
#   %add_34 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_135), kwargs = {})
#   %convolution_16 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_34, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_35 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_16,), kwargs = {memory_format: torch.contiguous_format})
#   %view_136 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_35, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_136, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_6 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_14 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_136, %getitem_45), kwargs = {})
#   %add_35 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_8), kwargs = {})
#   %view_137 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_37, [6, 256, 16, 16]), kwargs = {})
#   %view_141 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_137, [6, 256, -1]), kwargs = {})
#   %add_36 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_141, %bmm_25), kwargs = {})
#   %view_142 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_36, [6, 256, 16, 16]), kwargs = {})
#   %view_140 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [6, 256, 1, 1]), kwargs = {})
#   %add_37 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_142, %view_140), kwargs = {})
#   %sigmoid_19 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_37,), kwargs = {})
#   %mul_38 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sigmoid_19), kwargs = {})
#   %convolution_17 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_33), kwargs = {})
#   %sigmoid_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_39 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_20), kwargs = {})
#   %add_tensor_25 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_25, %arg95_1), kwargs = {})
#   %view_157 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_25, [6, 256, 256]), kwargs = {})
#   %permute_85 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [0, 2, 1]), kwargs = {})
#   %view_158 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_85, [6, 256, 16, 16]), kwargs = {})
#   %add_39 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %view_158), kwargs = {})
#   %convolution_18 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_39, %arg96_1, %arg97_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_159 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_18, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_159, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_7 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_16 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_159, %getitem_52), kwargs = {})
#   %add_40 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_51, 1e-06), kwargs = {})
#   %rsqrt_9 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
#   %mul_41 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %rsqrt_9), kwargs = {})
#   %view_160 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_41, [6, 256, 16, 16]), kwargs = {})
#   %view_164 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_160, [6, 256, -1]), kwargs = {})
#   %add_41 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_164, %bmm_29), kwargs = {})
#   %view_165 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_41, [6, 256, 16, 16]), kwargs = {})
#   %view_163 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_55, [6, 256, 1, 1]), kwargs = {})
#   %add_42 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_165, %view_163), kwargs = {})
#   %sigmoid_21 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_42,), kwargs = {})
#   %mul_42 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %sigmoid_21), kwargs = {})
#   %convolution_19 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_42, %arg98_1, %arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_43 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %mul_39), kwargs = {})
#   %sigmoid_22 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %view_166 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_43, [6, 256, 256]), kwargs = {})
#   %permute_87 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_166, [0, 2, 1]), kwargs = {})
#   %clone_42 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_87,), kwargs = {memory_format: torch.contiguous_format})
#   return %add_43,%clone_42
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9439232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/qx/cqx7bomncf677vkvdvjf2oobxcjeseclpr7lwqofcwgokdstkhjm.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#   h_12 => mul_43, sigmoid_22
#   input_32 => add_tensor_24, view_180
#   out_35 => view_181
#   transpose_60 => permute_96
#   x_30 => add_44
# Graph fragment:
#   %add_43 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %mm_default_24 : Tensor "f32[1536, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_24]
#   %arg103_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg103_1]
#   %sigmoid_22 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_24 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [6, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [6, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   return %add_44
triton_poi_fused_add_addmm_silu_transpose_view_56 = async_compile.triton('triton_poi_fused_add_addmm_silu_transpose_view_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_silu_transpose_view_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6292480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_silu_transpose_view_56(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/qv/cqvcblgqypcjesowdhnca55zb64zb5r3cjceqmrli4poelobnjaq.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_35 => add_48
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   input_32 => add_tensor_24, view_180
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
#   %buf232 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf232]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %sigmoid_22 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_24 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [6, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [6, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [6, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [6, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [6, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [6, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %view_189 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_47, [6, 256, 256]), kwargs = {})
#   %permute_98 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_189, [0, 2, 1]), kwargs = {})
#   %clone_48 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_98,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_48
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6292480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/qm/cqm7nucb2lcq7azpito3bh5qewd75l2rfwzry3bryswnervbjm6j.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_35 => add_48
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   input_32 => add_tensor_24, view_180
#   input_34 => add_tensor_23, view_203
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
#   %buf232 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf232]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %mm_default_23 : Tensor "f32[1536, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_23]
#   %arg111_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg111_1]
#   %sigmoid_22 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_24 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [6, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [6, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [6, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [6, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [6, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [6, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %add_tensor_23 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %arg111_1), kwargs = {})
#   %view_203 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [6, 256, 256]), kwargs = {})
#   %permute_107 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_203, [0, 2, 1]), kwargs = {})
#   %view_204 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_107, [6, 256, 16, 16]), kwargs = {})
#   %add_49 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %view_204), kwargs = {})
#   return %add_49
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7866368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/lb/clbkg4jwt5ngvhjtxsmtnohvjzstfghfoeklljpeyipnkgqf3u23.py
# Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
# Source node to ATen node mapping:
#   add_35 => add_48
#   add_39 => add_53
#   h_12 => mul_43, sigmoid_22
#   h_13 => mul_47, sigmoid_24
#   h_14 => mul_51, sigmoid_26
#   input_32 => add_tensor_24, view_180
#   input_34 => add_tensor_23, view_203
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
#   %buf254 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf254]
#   %arg115_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg115_1]
#   %buf232 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf232]
#   %arg107_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg107_1]
#   %add_43 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_43]
#   %add_53 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_53]
#   %sigmoid_22 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_43 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_22), kwargs = {})
#   %add_tensor_24 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %arg103_1), kwargs = {})
#   %view_180 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_24, [6, 256, 256]), kwargs = {})
#   %permute_96 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_180, [0, 2, 1]), kwargs = {})
#   %view_181 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_96, [6, 256, 16, 16]), kwargs = {})
#   %add_44 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %view_181), kwargs = {})
#   %convolution_20 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %arg104_1, %arg105_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_182 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_182, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_8 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_18 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_182, %getitem_59), kwargs = {})
#   %add_45 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_10 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_45 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_10), kwargs = {})
#   %view_183 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_45, [6, 256, 16, 16]), kwargs = {})
#   %view_187 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [6, 256, -1]), kwargs = {})
#   %add_46 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_187, %bmm_33), kwargs = {})
#   %view_188 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_46, [6, 256, 16, 16]), kwargs = {})
#   %view_186 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_62, [6, 256, 1, 1]), kwargs = {})
#   %add_47 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_188, %view_186), kwargs = {})
#   %sigmoid_23 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_47,), kwargs = {})
#   %mul_46 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %sigmoid_23), kwargs = {})
#   %convolution_21 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_46, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_48 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %mul_43), kwargs = {})
#   %sigmoid_24 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_47 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_24), kwargs = {})
#   %add_tensor_23 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %arg111_1), kwargs = {})
#   %view_203 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_23, [6, 256, 256]), kwargs = {})
#   %permute_107 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_203, [0, 2, 1]), kwargs = {})
#   %view_204 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_107, [6, 256, 16, 16]), kwargs = {})
#   %add_49 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %view_204), kwargs = {})
#   %convolution_22 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_49, %arg112_1, %arg113_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_205 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_22, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_205, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_9 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_20 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_205, %getitem_66), kwargs = {})
#   %add_50 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_65, 1e-06), kwargs = {})
#   %rsqrt_11 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_50,), kwargs = {})
#   %mul_49 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_11), kwargs = {})
#   %view_206 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_49, [6, 256, 16, 16]), kwargs = {})
#   %view_210 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_206, [6, 256, -1]), kwargs = {})
#   %add_51 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_210, %bmm_37), kwargs = {})
#   %view_211 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_51, [6, 256, 16, 16]), kwargs = {})
#   %view_209 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_69, [6, 256, 1, 1]), kwargs = {})
#   %add_52 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_211, %view_209), kwargs = {})
#   %sigmoid_25 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_52,), kwargs = {})
#   %mul_50 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_52, %sigmoid_25), kwargs = {})
#   %convolution_23 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_50, %arg114_1, %arg115_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_53 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_23, %mul_47), kwargs = {})
#   %sigmoid_26 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %view_212 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_51, [6, 256, 256]), kwargs = {})
#   %permute_109 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_212, [0, 2, 1]), kwargs = {})
#   %clone_54 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_109,), kwargs = {memory_format: torch.contiguous_format})
#   return %add_53,%clone_54
triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59 = async_compile.triton('triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11012096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: ./.compile_cache/m6/cm6dslqspps6ye3xg7vfd22vkgbfjesmsyy2jn33utofuqghngvc.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
# Source node to ATen node mapping:
#   add_43 => add_58
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   input_36 => add_tensor_22, view_226
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
#   %buf276 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf276]
#   %arg123_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg123_1]
#   %add_53 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=add_53]
#   %sigmoid_26 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_22 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [6, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [6, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [6, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [6, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [6, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [6, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   return %mul_55
triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60 = async_compile.triton('triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6292480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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
#   input_36 => add_tensor_22, view_226
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
#   %sigmoid_26 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_22 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [6, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [6, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [6, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [6, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [6, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [6, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   %convolution_26 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_55, %arg124_1, %arg125_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
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


# kernel path: ./.compile_cache/v3/cv3qxk3j6eb4yfofjooxprkstilazmj5i3ueygkaqwzfh26ytung.py
# Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   add_43 => add_58
#   gate_1 => unsqueeze_14
#   h_14 => mul_51, sigmoid_26
#   h_15 => mul_55, sigmoid_28
#   h_16 => convolution_26
#   h_17 => add_59
#   h_32_gated => mul_57
#   input_36 => add_tensor_22, view_226
#   input_40 => add_tensor_20
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
#   %buf279 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf279]
#   %arg125_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg125_1]
#   %mul_12 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_12]
#   %mm_default_20 : Tensor "f32[6, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_20]
#   %arg129_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg129_1]
#   %sigmoid_26 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_51 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_26), kwargs = {})
#   %add_tensor_22 : Tensor "f32[1536, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %arg119_1), kwargs = {})
#   %view_226 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_22, [6, 256, 256]), kwargs = {})
#   %permute_118 : Tensor "f32[6, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_226, [0, 2, 1]), kwargs = {})
#   %view_227 : Tensor "f32[6, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [6, 256, 16, 16]), kwargs = {})
#   %add_54 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %view_227), kwargs = {})
#   %convolution_24 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_54, %arg120_1, %arg121_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_228 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_24, [6, 32, 8, 256]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_10 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_22 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_228, %getitem_73), kwargs = {})
#   %add_55 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_55,), kwargs = {})
#   %mul_53 : Tensor "f32[6, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_12), kwargs = {})
#   %view_229 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_53, [6, 256, 16, 16]), kwargs = {})
#   %view_233 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_229, [6, 256, -1]), kwargs = {})
#   %add_56 : Tensor "f32[6, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_233, %bmm_41), kwargs = {})
#   %view_234 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_56, [6, 256, 16, 16]), kwargs = {})
#   %view_232 : Tensor "f32[6, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_76, [6, 256, 1, 1]), kwargs = {})
#   %add_57 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_234, %view_232), kwargs = {})
#   %sigmoid_27 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_54 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_27), kwargs = {})
#   %convolution_25 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_54, %arg122_1, %arg123_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %mul_51), kwargs = {})
#   %sigmoid_28 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %mul_55 : Tensor "f32[6, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sigmoid_28), kwargs = {})
#   %convolution_26 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_55, %arg124_1, %arg125_1, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_tensor_20 : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_20, %arg129_1), kwargs = {})
#   %sigmoid_30 : Tensor "f32[6, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_20,), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[6, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_30, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[6, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_57 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_14), kwargs = {})
#   %add_59 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %mul_57), kwargs = {})
#   return %add_59
triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62 = async_compile.triton('triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12587008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
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


# kernel path: ./.compile_cache/ii/ciinsajnbxgial2tlnnrfxyuk2zwczuc6bj67hx3bj2f4i6tz5w3.py
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
#   %buf337 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf337]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %mul_69 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_69]
#   %convolution_35 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_69, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_263 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_35, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_263, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_15 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub_27 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_263, %getitem_98), kwargs = {})
#   %add_76 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_97, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_70 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_17), kwargs = {})
#   %view_264 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [6, 128, 32, 32]), kwargs = {})
#   %view_268 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_264, [6, 128, -1]), kwargs = {})
#   %add_77 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_268, %bmm_51), kwargs = {})
#   %view_269 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [6, 128, 32, 32]), kwargs = {})
#   %view_267 : Tensor "f32[6, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_101, [6, 128, 1, 1]), kwargs = {})
#   %add_78 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_269, %view_267), kwargs = {})
#   %sigmoid_39 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convolution_36 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_71, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %mul_69), kwargs = {})
#   %sigmoid_40 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_72 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_40), kwargs = {})
#   %view_270 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_72, [6, 32, 4, 1024]), kwargs = {})
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
    size_hints={'x': 256, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072, 'r0_': 6291968}}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: ./.compile_cache/vg/cvgvcyzgv6uduz6225sw5o6coqsjwztt2k42m5hkkiosftj2yyo3.py
# Topologically Sorted Source Nodes: [unsqueeze_13, s_emb_1, cat_3], Original ATen: [aten.unsqueeze, aten.expand, aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_3
#   s_emb_1 => expand_40
#   unsqueeze_13 => unsqueeze_23
# Graph fragment:
#   %arg154_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg154_1]
#   %unsqueeze_23 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg154_1, 0), kwargs = {})
#   %expand_40 : Tensor "f32[6, 256][0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_23, [6, -1]), kwargs = {})
#   %cat_3 : Tensor "f32[6, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%addmm_26, %expand_40], -1), kwargs = {})
#   return %buf355
triton_poi_fused_cat_expand_unsqueeze_64 = async_compile.triton('triton_poi_fused_cat_expand_unsqueeze_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_expand_unsqueeze_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_expand_unsqueeze_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + 512*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/3k/c3kmpklirg63ntbpvolu3arilhpw537zfqb254cvfcb5a5wc75tr.py
# Topologically Sorted Source Nodes: [lt, zeros_like, full_like, sigma], Original ATen: [aten.lt, aten.zeros_like, aten.full_like, aten.where]
# Source node to ATen node mapping:
#   full_like => full_default_1
#   lt => lt
#   sigma => where
#   zeros_like => full_default
# Graph fragment:
#   %arg0_1 : Tensor "f32[6][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %lt : Tensor "b8[6][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, 0.5), kwargs = {})
#   %full_default : Tensor "f32[6][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "f32[6][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6], 0.3), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[6][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%lt, %full_default, %full_default_1), kwargs = {})
#   return %where
triton_poi_fused_full_like_lt_where_zeros_like_65 = async_compile.triton('triton_poi_fused_full_like_lt_where_zeros_like_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_full_like_lt_where_zeros_like_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 54}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_full_like_lt_where_zeros_like_65(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 < tmp1
    tmp3 = 0.0
    tmp4 = 0.3
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/x5/cx5wjrpi44hf43dxa4eden4grcdmty5x5wuet7b22oxkpg6fvqbb.py
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
#   %buf337 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf337]
#   %arg149_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %mul_69 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_69]
#   %getitem_103 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=getitem_103]
#   %buf339 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=buf339]
#   %arg150_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg150_1]
#   %arg151_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg151_1]
#   %add_81 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_81]
#   %buf678 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf678]
#   %mul_145 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=mul_145]
#   %getitem_207 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=getitem_207]
#   %buf680 : Tensor "f32[6, 32, 1, 1][32, 1, 192, 192]cuda:0" = PlaceHolder[target=buf680]
#   %add_163 : Tensor "f32[6, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_163]
#   %convolution_35 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_69, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_263 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_35, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_263, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_15 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub_27 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_263, %getitem_98), kwargs = {})
#   %add_76 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_97, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_76,), kwargs = {})
#   %mul_70 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_17), kwargs = {})
#   %view_264 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [6, 128, 32, 32]), kwargs = {})
#   %view_268 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_264, [6, 128, -1]), kwargs = {})
#   %add_77 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_268, %bmm_51), kwargs = {})
#   %view_269 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [6, 128, 32, 32]), kwargs = {})
#   %view_267 : Tensor "f32[6, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_101, [6, 128, 1, 1]), kwargs = {})
#   %add_78 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_269, %view_267), kwargs = {})
#   %sigmoid_39 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_78,), kwargs = {})
#   %mul_71 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %sigmoid_39), kwargs = {})
#   %convolution_36 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_71, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %mul_69), kwargs = {})
#   %sigmoid_40 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_72 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_40), kwargs = {})
#   %view_270 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_72, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %convolution_73 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_145, %arg146_1, %arg147_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_535 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_73, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean_36 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_535, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_31 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_30, [1536, 1536, 128], 1), kwargs = {})
#   %sub_56 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_535, %getitem_202), kwargs = {})
#   %add_158 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_201, 1e-06), kwargs = {})
#   %rsqrt_36 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_158,), kwargs = {})
#   %mul_146 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %rsqrt_36), kwargs = {})
#   %view_536 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_146, [6, 128, 32, 32]), kwargs = {})
#   %view_540 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_536, [6, 128, -1]), kwargs = {})
#   %add_159 : Tensor "f32[6, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_540, %bmm_103), kwargs = {})
#   %view_541 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_159, [6, 128, 32, 32]), kwargs = {})
#   %view_539 : Tensor "f32[6, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_205, [6, 128, 1, 1]), kwargs = {})
#   %add_160 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_541, %view_539), kwargs = {})
#   %sigmoid_81 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_160,), kwargs = {})
#   %mul_147 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_160, %sigmoid_81), kwargs = {})
#   %convolution_74 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_147, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_161 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_74, %mul_145), kwargs = {})
#   %sigmoid_82 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_161,), kwargs = {})
#   %mul_148 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %sigmoid_82), kwargs = {})
#   %view_542 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_148, [6, 32, 4, 1024]), kwargs = {})
#   %var_mean_37 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_542, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %getitem_103), kwargs = {})
#   %add_80 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_102, 1e-05), kwargs = {})
#   %rsqrt_18 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   %mul_73 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_18), kwargs = {})
#   %view_271 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_73, [6, 128, 32, 32]), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, 0), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, 2), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 3), kwargs = {})
#   %mul_74 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_271, %unsqueeze_17), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg151_1, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 3), kwargs = {})
#   %add_81 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_20), kwargs = {})
#   %sigmoid_41 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %sub_57 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_542, %getitem_207), kwargs = {})
#   %add_162 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_206, 1e-05), kwargs = {})
#   %rsqrt_37 : Tensor "f32[6, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_162,), kwargs = {})
#   %mul_149 : Tensor "f32[6, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %rsqrt_37), kwargs = {})
#   %view_543 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_149, [6, 128, 32, 32]), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, 0), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_37, 2), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, 3), kwargs = {})
#   %mul_150 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_543, %unsqueeze_39), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg151_1, 0), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, 2), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_41, 3), kwargs = {})
#   %add_163 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %unsqueeze_42), kwargs = {})
#   %sigmoid_83 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_151 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_83), kwargs = {})
#   return %add_81,%mul_75,%add_163,%mul_151
triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_66 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_66', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25167360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_66(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
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


# kernel path: ./.compile_cache/wn/cwnoiiocon6u6xuhhwt5h6uzop3qhjjulpfowfk2hbxt6wqntrek.py
# Topologically Sorted Source Nodes: [input_43, input_44, input_87, input_88], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_75, sigmoid_41
#   input_44 => convolution_37
#   input_87 => mul_151, sigmoid_83
#   input_88 => convolution_75
# Graph fragment:
#   %arg152_1 : Tensor "f32[4, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg152_1]
#   %sigmoid_41 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convolution_37 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_151 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_83), kwargs = {})
#   %convolution_75 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_151, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf685,%buf692
triton_poi_fused_convolution_silu_67 = async_compile.triton('triton_poi_fused_convolution_silu_67', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 73728, 'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_67(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: ./.compile_cache/qe/cqe7jad4ubfpegwuskrijom4unlhxxe6sye4wrpw4kindognbwix.py
# Topologically Sorted Source Nodes: [input_43, input_44, std_target, input_87, input_88, sub, mul_28, v_pred, std_pred, add_121, rescale_factor, v_pred_rescaled, mul_30, mul_31, v_guided, mul_32], Original ATen: [aten.silu, aten.convolution, aten.std, aten.sub, aten.mul, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_121 => add_165
#   input_43 => mul_75, sigmoid_41
#   input_44 => convolution_37
#   input_87 => mul_151, sigmoid_83
#   input_88 => convolution_75
#   mul_28 => mul_152
#   mul_30 => mul_154
#   mul_31 => mul_155
#   mul_32 => mul_156
#   rescale_factor => div_22
#   std_pred => sqrt_1, var_1
#   std_target => sqrt, var
#   sub => sub_58
#   v_guided => add_166
#   v_pred => add_164
#   v_pred_rescaled => mul_153
# Graph fragment:
#   %buf686 : Tensor "f32[6, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf686]
#   %arg153_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %buf693 : Tensor "f32[6, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf693]
#   %buf688 : Tensor "f32[6, 1, 1, 1][1, 6, 6, 6]cuda:0" = PlaceHolder[target=buf688]
#   %buf695 : Tensor "f32[6, 1, 1, 1][1, 6, 6, 6]cuda:0" = PlaceHolder[target=buf695]
#   %sigmoid_41 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convolution_37 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var : Tensor "f32[6, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%convolution_37, [1, 2, 3]), kwargs = {correction: 1.0, keepdim: True})
#   %sqrt : Tensor "f32[6, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%var,), kwargs = {})
#   %sigmoid_83 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_151 : Tensor "f32[6, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_83), kwargs = {})
#   %convolution_75 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_151, %arg152_1, %arg153_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_58 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %convolution_75), kwargs = {})
#   %mul_152 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, 5.0), kwargs = {})
#   %add_164 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_75, %mul_152), kwargs = {})
#   %var_1 : Tensor "f32[6, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.var.correction](args = (%add_164, [1, 2, 3]), kwargs = {correction: 1.0, keepdim: True})
#   %sqrt_1 : Tensor "f32[6, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%var_1,), kwargs = {})
#   %add_165 : Tensor "f32[6, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt_1, 1e-08), kwargs = {})
#   %div_22 : Tensor "f32[6, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt, %add_165), kwargs = {})
#   %mul_153 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, %div_22), kwargs = {})
#   %mul_154 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, 0.7), kwargs = {})
#   %mul_155 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_164, 0.30000000000000004), kwargs = {})
#   %add_166 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_154, %mul_155), kwargs = {})
#   %mul_156 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_166, 0.06666666666666667), kwargs = {})
#   return %buf688,%buf695,%mul_156
triton_red_fused_add_convolution_div_mul_silu_std_sub_68 = async_compile.triton('triton_red_fused_add_convolution_div_mul_silu_std_sub_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_mul_silu_std_sub_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 393232}}
)
@triton.jit
def triton_red_fused_add_convolution_div_mul_silu_std_sub_68(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 6
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
    tmp17_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        r0_1 = (r0_index % 4)
        tmp0 = tl.load(in_ptr0 + (r0_3 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_out_ptr0 + (r0_3 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp13 = 5.0
        tmp14 = tmp12 * tmp13
        tmp15 = tmp11 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(r0_mask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(r0_mask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(r0_mask & xmask, tmp17_weight_next, tmp17_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp18, tmp19, tmp20 = triton_helpers.welford(tmp17_mean, tmp17_m2, tmp17_weight, 1)
    tmp17 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tmp22 = tmp20[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        r0_1 = (r0_index % 4)
        tmp23 = tl.load(in_out_ptr0 + (r0_3 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr0 + (r0_3 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tmp23 + tmp24
        tmp27 = tmp26 + tmp24
        tmp28 = tmp27 - tmp25
        tmp29 = 5.0
        tmp30 = tmp28 * tmp29
        tmp31 = tmp25 + tmp30
        tmp32 = 4095.0
        tmp33 = (tmp8 / tmp32)
        tmp34 = libdevice.sqrt(tmp33)
        tmp35 = (tmp21 / tmp32)
        tmp36 = libdevice.sqrt(tmp35)
        tmp37 = 1e-08
        tmp38 = tmp36 + tmp37
        tmp39 = (tmp34 / tmp38)
        tmp40 = tmp31 * tmp39
        tmp41 = 0.7
        tmp42 = tmp40 * tmp41
        tmp43 = 0.30000000000000004
        tmp44 = tmp31 * tmp43
        tmp45 = tmp42 + tmp44
        tmp46 = 0.06666666666666667
        tmp47 = tmp45 * tmp46
        tl.store(in_out_ptr0 + (r0_3 + 4096*x0), tmp47, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ./.compile_cache/sc/cscb42ywdh3b72pmmgyok3oi36kezienpjvwoizsdaztpn4rhybi.py
# Topologically Sorted Source Nodes: [x_next], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_next => add_167
# Graph fragment:
#   %arg25_1 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg25_1]
#   %mul_156 : Tensor "f32[6, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=mul_156]
#   %add_167 : Tensor "f32[6, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg25_1, %mul_156), kwargs = {})
#   return %add_167
triton_poi_fused_add_69 = async_compile.triton('triton_poi_fused_add_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 98304, 'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_69(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
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
    tmp1 = tl.load(in_ptr1 + (y0 + 4*x2 + 4096*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1024*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: ./.compile_cache/hu/chuda4ufm4irfzanphtzyu7drpsbx45kxrebdhzujedycnzafpdz.py
# Topologically Sorted Source Nodes: [gt, add_noise], Original ATen: [aten.gt, aten.any]
# Source node to ATen node mapping:
#   add_noise => any_1
#   gt => gt
# Graph fragment:
#   %where : Tensor "f32[6][1]cuda:0" = PlaceHolder[target=where]
#   %gt : Tensor "b8[6][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
#   %any_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.default](args = (%gt,), kwargs = {})
#   return %any_1
triton_poi_fused_any_gt_70 = async_compile.triton('triton_poi_fused_any_gt_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_any_gt_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_any_gt_70(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (1))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (2))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr0 + (3))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (4))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (5))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp6 = tmp5 > tmp2
    tmp7 = tmp3 | tmp6
    tmp10 = tmp9 > tmp2
    tmp11 = tmp7 | tmp10
    tmp14 = tmp13 > tmp2
    tmp15 = tmp11 | tmp14
    tmp18 = tmp17 > tmp2
    tmp19 = tmp15 | tmp18
    tmp22 = tmp21 > tmp2
    tmp23 = tmp19 | tmp22
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp23, None)
''', device_str='cuda')

def partition_0(args):
    arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg154_1, arg150_1, arg151_1, arg152_1, arg153_1 = args
    args.clear()
    assert_size_stride(arg25_1, (6, 4, 32, 32), (4096, 1024, 32, 1))
    assert_size_stride(arg23_1, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg0_1, (6, ), (1, ))
    assert_size_stride(arg1_1, (1024, 256), (256, 1))
    assert_size_stride(arg2_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg3_1, (256, 1024), (1024, 1))
    assert_size_stride(arg6_1, (6, ), (1, ))
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
        buf0 = empty_strided_cuda((6, 4, 32, 32), (4096, 1, 128, 4), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg25_1, buf0, 24, 1024, stream=stream0)
        buf1 = empty_strided_cuda((128, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg23_1, buf1, 512, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf3, arg24_1, 786432, stream=stream0)
        buf4 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg26_1, buf4, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf6 = empty_strided_cuda((6, 32, 1, 1), (32, 1, 192, 192), torch.float32)
        buf7 = empty_strided_cuda((6, 32, 1, 1), (32, 1, 192, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf5, arg27_1, buf6, buf7, 192, 4096, stream=stream0)
        buf9 = empty_strided_cuda((6, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5.run(arg0_1, buf9, 1536, stream=stream0)
        buf10 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_arange_cat_cos_div_exp_mul_sin_unsqueeze_6.run(buf9, buf10, 2048, stream=stream0)
        buf11 = empty_strided_cuda((8, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.addmm, aten.t]
        extern_kernels.mm(buf10, reinterpret_tensor(arg1_1, (256, 1024), (1, 256), 0), out=buf11)
        buf12 = empty_strided_cuda((6, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_7.run(buf11, arg2_1, buf12, 6144, stream=stream0)
        buf15 = empty_strided_cuda((6, 512), (512, 1), torch.float32)
        buf13 = reinterpret_tensor(buf15, (6, 256), (512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg4_1, buf12, reinterpret_tensor(arg3_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf13)
        buf14 = reinterpret_tensor(buf15, (6, 256), (512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [s_emb], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_8.run(arg6_1, arg5_1, buf14, 1536, stream=stream0)
        del arg5_1
        del arg6_1
        del buf13
        del buf14
        buf16 = empty_strided_cuda((6, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf15, reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf16)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf17, arg8_1, 3072, stream=stream0)
        buf18 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg10_1, buf17, reinterpret_tensor(arg9_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf18)
        buf19 = empty_strided_cuda((6, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), out=buf19)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf20, arg12_1, 1536, stream=stream0)
        buf21 = empty_strided_cuda((6, 3200), (3200, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg14_1, buf20, reinterpret_tensor(arg13_1, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf21)
        buf22 = empty_strided_cuda((6, 32, 4, 1024), (131072, 4096, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf5, arg27_1, buf6, buf7, buf22, 768, 1024, stream=stream0)
        buf23 = empty_strided_cuda((6, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm, split, v_1, transpose, x_flat, v_t_x], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf22, (6, 128, 1024), (131072, 1024, 1), 0), out=buf23)
        buf24 = reinterpret_tensor(buf5, (6, 128, 1024), (131072, 1024, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [split, u_1, mixed], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 128, 12), (3200, 12, 1), 0), buf23, out=buf24)
        buf25 = empty_strided_cuda((6, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf22, buf24, buf21, buf25, 768, 1024, stream=stream0)
        del buf22
        del buf24
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf26 = extern_kernels.convolution(buf25, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2, add_2, h_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf27, arg29_1, buf3, 786432, stream=stream0)
        buf28 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg30_1, buf28, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf30 = buf7; del buf7  # reuse
        buf31 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf29, arg31_1, buf30, buf31, 192, 4096, stream=stream0)
        buf33 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg37_1, (256, 512), (1, 256), 0), out=buf33)
        buf34 = reinterpret_tensor(buf3, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf29, arg31_1, buf30, buf31, buf34, 768, 1024, stream=stream0)
        buf35 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, v_3, transpose_1, x_flat_1, v_t_x_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf34, (6, 128, 1024), (131072, 1024, 1), 0), out=buf35)
        buf36 = reinterpret_tensor(buf29, (6, 128, 1024), (131072, 1024, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [split_1, u_3, mixed_1], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 128, 12), (3200, 12, 1), 0), buf35, out=buf36)
        buf37 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf34, buf36, buf21, buf37, 768, 1024, stream=stream0)
        del buf34
        del buf36
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4, x_5], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf38 = extern_kernels.convolution(buf37, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4, x_5, add_5, h_2], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf39, arg33_1, buf27, 786432, stream=stream0)
        del buf27
        buf40 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(arg34_1, buf40, 32768, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf39, buf40, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf42 = empty_strided_cuda((6, 256, 256), (65536, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_transpose_view_15.run(buf41, arg35_1, buf42, 393216, stream=stream0)
        buf43 = empty_strided_cuda((1536, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (1536, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 256), (1, 256), 0), out=buf43)
        buf44 = reinterpret_tensor(buf42, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf43, buf44, 393216, stream=stream0)
        buf45 = reinterpret_tensor(buf20, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf33, buf45, 1536, stream=stream0)
        buf46 = reinterpret_tensor(buf12, (24, 256, 1), (256, 1, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, q_1, view_11, q_2, matmul, view_12, k_1, transpose_6], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf45, (24, 64, 1), (64, 1, 0), 0), out=buf46)
        buf47 = reinterpret_tensor(buf46, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [matmul, attn_1], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf47, 6144, stream=stream0)
        buf48 = reinterpret_tensor(buf45, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf33, buf48, 1536, stream=stream0)
        buf49 = reinterpret_tensor(buf44, (24, 256, 64), (16384, 64, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [kv, chunk, matmul, attn_1, matmul_1, view_13, v_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf48, (24, 1, 64), (64, 0, 1), 0), out=buf49)
        buf50 = reinterpret_tensor(buf43, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf49, buf50, 393216, stream=stream0)
        buf51 = reinterpret_tensor(buf49, (1536, 256), (256, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4, input_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf50, (1536, 256), (256, 1), 0), reinterpret_tensor(arg38_1, (256, 256), (1, 256), 0), out=buf51)
        buf52 = reinterpret_tensor(buf51, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(buf52, buf41, arg35_1, arg39_1, 393216, stream=stream0)
        buf53 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg40_1, buf53, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf55 = buf31; del buf31  # reuse
        buf56 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf54, arg41_1, buf55, buf56, 192, 2048, stream=stream0)
        buf58 = reinterpret_tensor(buf48, (6, 256), (256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg15_1, (256, 256), (1, 256), 0), out=buf58)
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf59, arg16_1, 1536, stream=stream0)
        buf60 = empty_strided_cuda((6, 6400), (6400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg18_1, buf59, reinterpret_tensor(arg17_1, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf60)
        buf61 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg45_1, (256, 512), (1, 256), 0), out=buf61)
        buf62 = reinterpret_tensor(buf52, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf54, arg41_1, buf55, buf56, buf62, 1536, 256, stream=stream0)
        buf63 = empty_strided_cuda((6, 12, 256), (3072, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, v_7, transpose_9, x_flat_2, v_t_x_2], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf62, (6, 256, 256), (65536, 256, 1), 0), out=buf63)
        buf64 = reinterpret_tensor(buf54, (6, 256, 256), (65536, 256, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [split_2, u_5, mixed_2], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 256, 12), (6400, 12, 1), 0), buf63, out=buf64)
        buf65 = reinterpret_tensor(buf50, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf62, buf64, buf60, buf65, 1536, 256, stream=stream0)
        del buf62
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf66 = extern_kernels.convolution(buf65, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf67 = reinterpret_tensor(buf65, (6, 256, 256), (65536, 256, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26.run(buf66, arg43_1, buf41, arg35_1, buf67, 393216, stream=stream0)
        buf68 = reinterpret_tensor(buf64, (1536, 256), (256, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (1536, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 256), (1, 256), 0), out=buf68)
        buf69 = reinterpret_tensor(buf67, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [q_4, view_21, q_5, matmul_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf68, buf69, 393216, stream=stream0)
        buf70 = reinterpret_tensor(buf59, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_22, k_3, transpose_14, matmul_2], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf61, buf70, 1536, stream=stream0)
        buf71 = reinterpret_tensor(buf47, (24, 256, 1), (256, 1, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, q_4, view_21, q_5, matmul_2, view_22, k_3, transpose_14], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf70, (24, 64, 1), (64, 1, 0), 0), out=buf71)
        buf72 = reinterpret_tensor(buf71, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [matmul_2, attn_3], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf72, 6144, stream=stream0)
        buf73 = reinterpret_tensor(buf70, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_23, v_9, matmul_3], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf61, buf73, 1536, stream=stream0)
        buf74 = reinterpret_tensor(buf69, (24, 256, 64), (16384, 64, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [kv_1, chunk_1, matmul_2, attn_3, matmul_3, view_23, v_9], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf73, (24, 1, 64), (64, 0, 1), 0), out=buf74)
        buf75 = reinterpret_tensor(buf68, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf74, buf75, 393216, stream=stream0)
        buf76 = reinterpret_tensor(buf74, (1536, 256), (256, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8, input_18], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf75, (1536, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 256), (1, 256), 0), out=buf76)
        buf77 = reinterpret_tensor(buf76, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27.run(buf77, buf66, arg43_1, buf41, arg35_1, arg47_1, 393216, stream=stream0)
        buf78 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg48_1, buf78, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf79 = extern_kernels.convolution(buf77, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf80 = buf56; del buf56  # reuse
        buf81 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf79, arg49_1, buf80, buf81, 192, 2048, stream=stream0)
        buf83 = reinterpret_tensor(buf72, (6, 1024), (1024, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg55_1, (256, 1024), (1, 256), 0), out=buf83)
        buf84 = reinterpret_tensor(buf77, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf79, arg49_1, buf80, buf81, buf84, 1536, 256, stream=stream0)
        buf85 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, v_11, transpose_17, x_flat_3, v_t_x_3], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf84, (6, 256, 256), (65536, 256, 1), 0), out=buf85)
        buf86 = reinterpret_tensor(buf79, (6, 256, 256), (65536, 256, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [split_3, u_7, mixed_3], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 256, 12), (6400, 12, 1), 0), buf85, out=buf86)
        buf87 = reinterpret_tensor(buf75, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf84, buf86, buf60, buf87, 1536, 256, stream=stream0)
        del buf84
        del buf86
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf88 = extern_kernels.convolution(buf87, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf87
        buf89 = buf88; del buf88  # reuse
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [h_3, input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9, add_9, h_4, input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13, add_13, h_5], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf90, arg51_1, buf66, arg43_1, buf41, arg35_1, 393216, stream=stream0)
        del buf41
        buf91 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_29.run(arg52_1, buf91, 131072, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_5, h_6], Original ATen: [aten.silu, aten.convolution]
        buf92 = extern_kernels.convolution(buf90, buf91, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf91
        buf93 = empty_strided_cuda((6, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_silu_transpose_view_30.run(buf92, arg53_1, buf93, 196608, stream=stream0)
        buf94 = empty_strided_cuda((384, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, view_30, q_6, q_7], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (384, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf94)
        buf95 = reinterpret_tensor(buf93, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_31.run(buf94, buf95, 196608, stream=stream0)
        buf96 = reinterpret_tensor(buf61, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_32.run(buf83, buf96, 3072, stream=stream0)
        buf97 = reinterpret_tensor(buf15, (48, 64, 1), (64, 1, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, q_7, view_31, q_8, matmul_4, view_32, k_5, transpose_22], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf95, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf96, (48, 64, 1), (64, 1, 0), 0), out=buf97)
        buf98 = reinterpret_tensor(buf97, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [matmul_4, attn_5], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_33.run(buf98, 3072, stream=stream0)
        buf99 = reinterpret_tensor(buf96, (6, 8, 1, 64), (512, 64, 64, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_34.run(buf83, buf99, 3072, stream=stream0)
        buf100 = reinterpret_tensor(buf95, (48, 64, 64), (4096, 64, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [kv_2, chunk_2, matmul_4, attn_5, matmul_5, view_33, v_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (48, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf99, (48, 1, 64), (64, 0, 1), 0), out=buf100)
        buf101 = reinterpret_tensor(buf94, (6, 64, 8, 64), (32768, 512, 64, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_35.run(buf100, buf101, 196608, stream=stream0)
        buf102 = reinterpret_tensor(buf100, (384, 512), (512, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12, input_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf101, (384, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 512), (1, 512), 0), out=buf102)
        buf103 = reinterpret_tensor(buf102, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(buf103, buf92, arg53_1, arg57_1, 196608, stream=stream0)
        buf104 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_37.run(arg58_1, buf104, 262144, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf106 = buf81; del buf81  # reuse
        buf107 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf105, arg59_1, buf106, buf107, 192, 1024, stream=stream0)
        buf109 = reinterpret_tensor(buf73, (6, 256), (256, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg19_1, (256, 256), (1, 256), 0), out=buf109)
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf110, arg20_1, 1536, stream=stream0)
        buf111 = empty_strided_cuda((6, 12800), (12800, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg22_1, buf110, reinterpret_tensor(arg21_1, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf111)
        buf112 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg63_1, (256, 1024), (1, 256), 0), out=buf112)
        buf113 = reinterpret_tensor(buf103, (6, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39.run(buf105, arg59_1, buf106, buf107, buf113, 3072, 64, stream=stream0)
        buf114 = reinterpret_tensor(buf1, (6, 12, 64), (768, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, v_15, transpose_25, x_flat_4, v_t_x_4], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (6, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf113, (6, 512, 64), (32768, 64, 1), 0), out=buf114)
        buf115 = reinterpret_tensor(buf105, (6, 512, 64), (32768, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [split_4, u_9, mixed_4], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (6, 512, 12), (12800, 12, 1), 0), buf114, out=buf115)
        buf116 = reinterpret_tensor(buf101, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40.run(buf113, buf115, buf111, buf116, 3072, 64, stream=stream0)
        del buf113
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf117 = extern_kernels.convolution(buf116, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf118 = reinterpret_tensor(buf116, (6, 64, 512), (32768, 512, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41.run(buf117, arg61_1, buf92, arg53_1, buf118, 196608, stream=stream0)
        buf119 = reinterpret_tensor(buf115, (384, 512), (512, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (384, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 512), (1, 512), 0), out=buf119)
        buf120 = reinterpret_tensor(buf118, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [q_10, view_41, q_11, matmul_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_31.run(buf119, buf120, 196608, stream=stream0)
        buf121 = reinterpret_tensor(buf99, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_42, k_7, transpose_30, matmul_6], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_32.run(buf112, buf121, 3072, stream=stream0)
        buf122 = reinterpret_tensor(buf98, (48, 64, 1), (64, 1, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, q_10, view_41, q_11, matmul_6, view_42, k_7, transpose_30], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf121, (48, 64, 1), (64, 1, 0), 0), out=buf122)
        buf123 = reinterpret_tensor(buf122, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [matmul_6, attn_7], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_33.run(buf123, 3072, stream=stream0)
        buf124 = reinterpret_tensor(buf121, (6, 8, 1, 64), (512, 64, 64, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_43, v_17, matmul_7], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_34.run(buf112, buf124, 3072, stream=stream0)
        buf125 = reinterpret_tensor(buf120, (48, 64, 64), (4096, 64, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [kv_3, chunk_3, matmul_6, attn_7, matmul_7, view_43, v_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf123, (48, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf124, (48, 1, 64), (64, 0, 1), 0), out=buf125)
        buf126 = reinterpret_tensor(buf119, (6, 64, 8, 64), (32768, 512, 64, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_35.run(buf125, buf126, 196608, stream=stream0)
        buf127 = reinterpret_tensor(buf125, (384, 512), (512, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16, input_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf126, (384, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf127)
        buf128 = reinterpret_tensor(buf127, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_42.run(buf128, buf117, arg61_1, buf92, arg53_1, arg65_1, 196608, stream=stream0)
        buf129 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_37.run(arg66_1, buf129, 262144, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf130 = extern_kernels.convolution(buf128, buf129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf129
        buf131 = buf107; del buf107  # reuse
        buf132 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf130, arg67_1, buf131, buf132, 192, 1024, stream=stream0)
        buf134 = reinterpret_tensor(buf128, (6, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39.run(buf130, arg67_1, buf131, buf132, buf134, 3072, 64, stream=stream0)
        buf135 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, v_19, transpose_33, x_flat_5, v_t_x_5], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (6, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf134, (6, 512, 64), (32768, 64, 1), 0), out=buf135)
        buf136 = reinterpret_tensor(buf130, (6, 512, 64), (32768, 64, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [split_5, u_11, mixed_5], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (6, 512, 12), (12800, 12, 1), 0), buf135, out=buf136)
        buf137 = reinterpret_tensor(buf126, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40.run(buf134, buf136, buf111, buf137, 3072, 64, stream=stream0)
        del buf134
        del buf136
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf138 = extern_kernels.convolution(buf137, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf139 = buf138; del buf138  # reuse
        buf143 = reinterpret_tensor(buf137, (6, 64, 512), (32768, 512, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [h_5, h_6, input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17, add_17, h_7, input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_43.run(buf139, arg69_1, buf117, arg61_1, buf92, arg53_1, arg70_1, arg71_1, buf143, 384, 512, stream=stream0)
        buf144 = reinterpret_tensor(buf78, (384, 1536), (1536, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf143, (384, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 1536), (1, 512), 0), out=buf144)
        buf145 = reinterpret_tensor(buf143, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_44.run(buf144, arg73_1, buf145, 196608, stream=stream0)
        buf146 = reinterpret_tensor(buf92, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_transpose_view_45.run(buf144, arg73_1, buf146, 3072, 64, stream=stream0)
        buf147 = reinterpret_tensor(buf117, (48, 64, 64), (4096, 64, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8, k_8, transpose_35], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone, aten._unsafe_view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf146, (48, 64, 64), (4096, 64, 1), 0), out=buf147)
        buf150 = reinterpret_tensor(buf147, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [matmul_8, attn_9], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_amax_mul_sub_view_46.run(buf150, 3072, 64, stream=stream0)
        buf151 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_47.run(buf144, arg73_1, buf151, 196608, stream=stream0)
        buf152 = reinterpret_tensor(buf145, (48, 64, 64), (4096, 64, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, matmul_8, attn_9, out_20, v_20], Original ATen: [aten.addmm, aten.view, aten.permute, aten.mul, aten.sub, aten._softmax, aten.select, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf150, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf151, (48, 64, 64), (4096, 64, 1), 0), out=buf152)
        del buf150
        buf153 = reinterpret_tensor(buf151, (6, 64, 8, 64), (32768, 512, 64, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_35.run(buf152, buf153, 196608, stream=stream0)
        buf154 = reinterpret_tensor(buf152, (384, 512), (512, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21, out_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf153, (384, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf154)
        del buf153
        buf159 = reinterpret_tensor(buf139, (6, 64, 512), (32768, 512, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_48.run(buf159, buf154, arg75_1, arg76_1, arg77_1, 384, 512, stream=stream0)
        del buf154
        buf158 = reinterpret_tensor(buf124, (6, 512), (512, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg85_1, (256, 512), (1, 256), 0), out=buf158)
        buf160 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_49.run(arg78_1, buf160, 131072, 16, stream=stream0)
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        buf161 = extern_kernels.convolution(reinterpret_tensor(buf159, (6, 512, 8, 8), (32768, 1, 4096, 512), 0), buf160, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf159
        del buf160
        buf162 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg80_1, (256, 256), (1, 256), 0), out=buf162)
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf163, arg81_1, 1536, stream=stream0)
        buf164 = empty_strided_cuda((6, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf163, reinterpret_tensor(arg82_1, (256, 256), (1, 256), 0), out=buf164)
        buf165 = reinterpret_tensor(buf66, (6, 256, 16, 16), (65536, 256, 16, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50.run(buf161, arg79_1, buf90, buf164, arg83_1, buf165, 1536, 256, stream=stream0)
        buf166 = reinterpret_tensor(buf90, (1536, 256), (1, 1536), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_t_transpose_view_51.run(buf165, buf166, 393216, stream=stream0)
        buf167 = reinterpret_tensor(buf161, (1536, 256), (256, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        extern_kernels.mm(buf166, reinterpret_tensor(arg84_1, (256, 256), (1, 256), 0), out=buf167)
        buf168 = reinterpret_tensor(buf166, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [q_14, view_53, q_15, matmul_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf167, buf168, 393216, stream=stream0)
        buf169 = reinterpret_tensor(buf164, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_54, k_10, transpose_42, matmul_10], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf158, buf169, 1536, stream=stream0)
        buf170 = reinterpret_tensor(buf112, (24, 256, 1), (256, 1, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, q_14, view_53, q_15, matmul_10, view_54, k_10, transpose_42], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf169, (24, 64, 1), (64, 1, 0), 0), out=buf170)
        buf171 = reinterpret_tensor(buf170, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [matmul_10, attn_11], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf171, 6144, stream=stream0)
        buf172 = reinterpret_tensor(buf169, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_55, v_22, matmul_11], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf158, buf172, 1536, stream=stream0)
        buf173 = reinterpret_tensor(buf168, (24, 256, 64), (16384, 64, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [kv_4, chunk_4, matmul_10, attn_11, matmul_11, view_55, v_22], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf172, (24, 1, 64), (64, 0, 1), 0), out=buf173)
        buf174 = reinterpret_tensor(buf167, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf173, buf174, 393216, stream=stream0)
        buf175 = reinterpret_tensor(buf173, (1536, 256), (256, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26, input_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf174, (1536, 256), (256, 1), 0), reinterpret_tensor(arg86_1, (256, 256), (1, 256), 0), out=buf175)
        buf176 = reinterpret_tensor(buf175, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_transpose_view_52.run(buf176, buf165, arg87_1, 1536, 256, stream=stream0)
        buf177 = reinterpret_tensor(buf144, (256, 256, 3, 3), (2304, 1, 768, 256), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg88_1, buf177, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf179 = buf132; del buf132  # reuse
        buf180 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf178, arg89_1, buf179, buf180, 192, 2048, stream=stream0)
        buf182 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg93_1, (256, 512), (1, 256), 0), out=buf182)
        buf183 = reinterpret_tensor(buf176, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf178, arg89_1, buf179, buf180, buf183, 1536, 256, stream=stream0)
        buf184 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, v_24, transpose_45, x_flat_7, v_t_x_6], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf183, (6, 256, 256), (65536, 256, 1), 0), out=buf184)
        buf185 = reinterpret_tensor(buf178, (6, 256, 256), (65536, 256, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [split_6, u_13, mixed_6], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 256, 12), (6400, 12, 1), 0), buf184, out=buf185)
        buf186 = reinterpret_tensor(buf174, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf183, buf185, buf60, buf186, 1536, 256, stream=stream0)
        del buf183
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf187 = extern_kernels.convolution(buf186, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf188 = reinterpret_tensor(buf186, (6, 256, 256), (65536, 256, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53.run(buf187, arg91_1, buf165, buf188, 1536, 256, stream=stream0)
        buf189 = reinterpret_tensor(buf185, (1536, 256), (256, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (1536, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf189)
        buf190 = reinterpret_tensor(buf188, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [q_17, view_63, q_18, matmul_12], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf189, buf190, 393216, stream=stream0)
        buf191 = reinterpret_tensor(buf172, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_64, k_12, transpose_50, matmul_12], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf182, buf191, 1536, stream=stream0)
        buf192 = reinterpret_tensor(buf171, (24, 256, 1), (256, 1, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, q_17, view_63, q_18, matmul_12, view_64, k_12, transpose_50], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf191, (24, 64, 1), (64, 1, 0), 0), out=buf192)
        buf193 = reinterpret_tensor(buf192, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [matmul_12, attn_13], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf193, 6144, stream=stream0)
        buf194 = reinterpret_tensor(buf191, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_65, v_26, matmul_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf182, buf194, 1536, stream=stream0)
        buf195 = reinterpret_tensor(buf190, (24, 256, 64), (16384, 64, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [kv_5, chunk_5, matmul_12, attn_13, matmul_13, view_65, v_26], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf194, (24, 1, 64), (64, 0, 1), 0), out=buf195)
        buf196 = reinterpret_tensor(buf189, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf195, buf196, 393216, stream=stream0)
        buf197 = reinterpret_tensor(buf195, (1536, 256), (256, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30, input_30], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf196, (1536, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 256), (1, 256), 0), out=buf197)
        buf198 = reinterpret_tensor(buf197, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54.run(buf198, buf187, arg91_1, buf165, arg95_1, 1536, 256, stream=stream0)
        buf199 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg96_1, buf199, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf200 = extern_kernels.convolution(buf198, buf199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf201 = buf180; del buf180  # reuse
        buf202 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf200, arg97_1, buf201, buf202, 192, 2048, stream=stream0)
        buf204 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg101_1, (256, 512), (1, 256), 0), out=buf204)
        buf205 = reinterpret_tensor(buf198, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf200, arg97_1, buf201, buf202, buf205, 1536, 256, stream=stream0)
        buf206 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, v_28, transpose_53, x_flat_8, v_t_x_7], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf205, (6, 256, 256), (65536, 256, 1), 0), out=buf206)
        buf207 = reinterpret_tensor(buf200, (6, 256, 256), (65536, 256, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [split_7, u_15, mixed_7], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 256, 12), (6400, 12, 1), 0), buf206, out=buf207)
        buf208 = reinterpret_tensor(buf196, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf205, buf207, buf60, buf208, 1536, 256, stream=stream0)
        del buf205
        del buf207
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf209 = extern_kernels.convolution(buf208, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf210 = buf209; del buf209  # reuse
        buf211 = reinterpret_tensor(buf208, (6, 256, 256), (65536, 256, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25, add_27, h_11, input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55.run(buf210, arg99_1, buf187, arg91_1, buf165, buf211, 393216, stream=stream0)
        del buf165
        buf212 = reinterpret_tensor(buf187, (1536, 256), (256, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [h_12, view_72, q_19, q_20], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (1536, 256), (256, 1), 0), reinterpret_tensor(arg100_1, (256, 256), (1, 256), 0), out=buf212)
        buf213 = reinterpret_tensor(buf211, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [q_20, view_73, q_21, matmul_14], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf212, buf213, 393216, stream=stream0)
        buf214 = reinterpret_tensor(buf194, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_74, k_14, transpose_58, matmul_14], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf204, buf214, 1536, stream=stream0)
        buf215 = reinterpret_tensor(buf193, (24, 256, 1), (256, 1, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, q_20, view_73, q_21, matmul_14, view_74, k_14, transpose_58], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf214, (24, 64, 1), (64, 1, 0), 0), out=buf215)
        buf216 = reinterpret_tensor(buf215, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [matmul_14, attn_15], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf216, 6144, stream=stream0)
        buf217 = reinterpret_tensor(buf214, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_75, v_30, matmul_15], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf204, buf217, 1536, stream=stream0)
        buf218 = reinterpret_tensor(buf213, (24, 256, 64), (16384, 64, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [kv_6, chunk_6, matmul_14, attn_15, matmul_15, view_75, v_30], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf217, (24, 1, 64), (64, 0, 1), 0), out=buf218)
        buf219 = reinterpret_tensor(buf212, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf218, buf219, 393216, stream=stream0)
        buf220 = reinterpret_tensor(buf218, (1536, 256), (256, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34, input_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf219, (1536, 256), (256, 1), 0), reinterpret_tensor(arg102_1, (256, 256), (1, 256), 0), out=buf220)
        buf221 = reinterpret_tensor(buf220, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_56.run(buf221, buf210, arg103_1, 393216, stream=stream0)
        buf222 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg104_1, buf222, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf223 = extern_kernels.convolution(buf221, buf222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf224 = buf202; del buf202  # reuse
        buf225 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf223, arg105_1, buf224, buf225, 192, 2048, stream=stream0)
        buf227 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg109_1, (256, 512), (1, 256), 0), out=buf227)
        buf228 = reinterpret_tensor(buf221, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf223, arg105_1, buf224, buf225, buf228, 1536, 256, stream=stream0)
        buf229 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, v_32, transpose_61, x_flat_9, v_t_x_8], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf228, (6, 256, 256), (65536, 256, 1), 0), out=buf229)
        buf230 = reinterpret_tensor(buf223, (6, 256, 256), (65536, 256, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [split_8, u_17, mixed_8], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 256, 12), (6400, 12, 1), 0), buf229, out=buf230)
        buf231 = reinterpret_tensor(buf219, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf228, buf230, buf60, buf231, 1536, 256, stream=stream0)
        del buf228
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf232 = extern_kernels.convolution(buf231, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf233 = reinterpret_tensor(buf231, (6, 256, 256), (65536, 256, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57.run(buf232, arg107_1, buf210, buf233, 393216, stream=stream0)
        buf234 = reinterpret_tensor(buf230, (1536, 256), (256, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (1536, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), out=buf234)
        buf235 = reinterpret_tensor(buf233, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [q_23, view_83, q_24, matmul_16], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf234, buf235, 393216, stream=stream0)
        buf236 = reinterpret_tensor(buf217, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_84, k_16, transpose_66, matmul_16], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf227, buf236, 1536, stream=stream0)
        buf237 = reinterpret_tensor(buf216, (24, 256, 1), (256, 1, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, q_23, view_83, q_24, matmul_16, view_84, k_16, transpose_66], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf236, (24, 64, 1), (64, 1, 0), 0), out=buf237)
        buf238 = reinterpret_tensor(buf237, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [matmul_16, attn_17], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf238, 6144, stream=stream0)
        buf239 = reinterpret_tensor(buf236, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_85, v_34, matmul_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf227, buf239, 1536, stream=stream0)
        buf240 = reinterpret_tensor(buf235, (24, 256, 64), (16384, 64, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [kv_7, chunk_7, matmul_16, attn_17, matmul_17, view_85, v_34], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf238, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf239, (24, 1, 64), (64, 0, 1), 0), out=buf240)
        buf241 = reinterpret_tensor(buf234, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf240, buf241, 393216, stream=stream0)
        buf242 = reinterpret_tensor(buf240, (1536, 256), (256, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38, input_34], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf241, (1536, 256), (256, 1), 0), reinterpret_tensor(arg110_1, (256, 256), (1, 256), 0), out=buf242)
        buf243 = reinterpret_tensor(buf242, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58.run(buf243, buf232, arg107_1, buf210, arg111_1, 393216, stream=stream0)
        buf244 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg112_1, buf244, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf245 = extern_kernels.convolution(buf243, buf244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf246 = buf225; del buf225  # reuse
        buf247 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf245, arg113_1, buf246, buf247, 192, 2048, stream=stream0)
        buf249 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg117_1, (256, 512), (1, 256), 0), out=buf249)
        buf250 = reinterpret_tensor(buf243, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf245, arg113_1, buf246, buf247, buf250, 1536, 256, stream=stream0)
        buf251 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, v_36, transpose_69, x_flat_10, v_t_x_9], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf250, (6, 256, 256), (65536, 256, 1), 0), out=buf251)
        buf252 = reinterpret_tensor(buf245, (6, 256, 256), (65536, 256, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [split_9, u_19, mixed_9], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 256, 12), (6400, 12, 1), 0), buf251, out=buf252)
        buf253 = reinterpret_tensor(buf241, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf250, buf252, buf60, buf253, 1536, 256, stream=stream0)
        del buf250
        del buf252
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf254 = extern_kernels.convolution(buf253, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf255 = buf254; del buf254  # reuse
        buf256 = reinterpret_tensor(buf253, (6, 256, 256), (65536, 256, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [h_12, input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33, add_35, h_13, input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59.run(buf255, arg115_1, buf232, arg107_1, buf210, buf256, 393216, stream=stream0)
        del buf210
        buf257 = reinterpret_tensor(buf232, (1536, 256), (256, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (1536, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 256), (1, 256), 0), out=buf257)
        buf258 = reinterpret_tensor(buf256, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [q_26, view_93, q_27, matmul_18], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf257, buf258, 393216, stream=stream0)
        buf259 = reinterpret_tensor(buf239, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_94, k_18, transpose_74, matmul_18], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf249, buf259, 1536, stream=stream0)
        buf260 = reinterpret_tensor(buf238, (24, 256, 1), (256, 1, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, q_26, view_93, q_27, matmul_18, view_94, k_18, transpose_74], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf258, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf259, (24, 64, 1), (64, 1, 0), 0), out=buf260)
        buf261 = reinterpret_tensor(buf260, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [matmul_18, attn_19], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf261, 6144, stream=stream0)
        buf262 = reinterpret_tensor(buf259, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_95, v_38, matmul_19], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf249, buf262, 1536, stream=stream0)
        buf263 = reinterpret_tensor(buf258, (24, 256, 64), (16384, 64, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [kv_8, chunk_8, matmul_18, attn_19, matmul_19, view_95, v_38], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf262, (24, 1, 64), (64, 0, 1), 0), out=buf263)
        buf264 = reinterpret_tensor(buf257, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf263, buf264, 393216, stream=stream0)
        buf265 = reinterpret_tensor(buf263, (1536, 256), (256, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42, input_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf264, (1536, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 256), (1, 256), 0), out=buf265)
        buf266 = reinterpret_tensor(buf265, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_56.run(buf266, buf255, arg119_1, 393216, stream=stream0)
        buf267 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg120_1, buf267, 65536, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf267
        buf269 = buf247; del buf247  # reuse
        buf270 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf268, arg121_1, buf269, buf270, 192, 2048, stream=stream0)
        buf272 = reinterpret_tensor(buf266, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf268, arg121_1, buf269, buf270, buf272, 1536, 256, stream=stream0)
        buf273 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, v_40, transpose_77, x_flat_11, v_t_x_10], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf272, (6, 256, 256), (65536, 256, 1), 0), out=buf273)
        buf274 = reinterpret_tensor(buf268, (6, 256, 256), (65536, 256, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [split_10, u_21, mixed_10], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (6, 256, 12), (6400, 12, 1), 0), buf273, out=buf274)
        buf275 = reinterpret_tensor(buf264, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf272, buf274, buf60, buf275, 1536, 256, stream=stream0)
        del buf272
        del buf274
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf276 = extern_kernels.convolution(buf275, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf275
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60.run(buf277, arg123_1, buf255, 393216, stream=stream0)
        buf278 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_61.run(arg124_1, buf278, 32768, 16, stream=stream0)
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf279 = extern_kernels.convolution(buf277, buf278, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf278
        buf280 = reinterpret_tensor(buf262, (6, 256), (256, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg126_1, (256, 256), (1, 256), 0), out=buf280)
        buf281 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf281, arg127_1, 1536, stream=stream0)
        buf282 = empty_strided_cuda((6, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf281, reinterpret_tensor(arg128_1, (256, 128), (1, 256), 0), out=buf282)
        buf283 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [h_14, input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62.run(buf283, arg125_1, buf39, buf282, arg129_1, 786432, stream=stream0)
        buf284 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg130_1, buf284, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf283, buf284, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf286 = buf270; del buf270  # reuse
        buf287 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf285, arg131_1, buf286, buf287, 192, 4096, stream=stream0)
        buf289 = reinterpret_tensor(buf39, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf285, arg131_1, buf286, buf287, buf289, 768, 1024, stream=stream0)
        buf290 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, v_42, transpose_78, x_flat_12, v_t_x_11], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf289, (6, 128, 1024), (131072, 1024, 1), 0), out=buf290)
        buf291 = reinterpret_tensor(buf285, (6, 128, 1024), (131072, 1024, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [split_11, u_23, mixed_11], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 128, 12), (3200, 12, 1), 0), buf290, out=buf291)
        buf292 = empty_strided_cuda((6, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf289, buf291, buf21, buf292, 768, 1024, stream=stream0)
        del buf289
        del buf291
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43, x_44], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf293 = extern_kernels.convolution(buf292, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf294 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43, x_44, add_47, h_18], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf294, arg133_1, buf283, 786432, stream=stream0)
        buf295 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg134_1, buf295, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf294, buf295, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf297 = buf287; del buf287  # reuse
        buf298 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf296, arg135_1, buf297, buf298, 192, 4096, stream=stream0)
        buf300 = reinterpret_tensor(buf283, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf296, arg135_1, buf297, buf298, buf300, 768, 1024, stream=stream0)
        buf301 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, v_44, transpose_79, x_flat_13, v_t_x_12], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf300, (6, 128, 1024), (131072, 1024, 1), 0), out=buf301)
        buf302 = reinterpret_tensor(buf296, (6, 128, 1024), (131072, 1024, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [split_12, u_25, mixed_12], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 128, 12), (3200, 12, 1), 0), buf301, out=buf302)
        buf303 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf300, buf302, buf21, buf303, 768, 1024, stream=stream0)
        del buf300
        del buf302
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46, x_47], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf304 = extern_kernels.convolution(buf303, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf305 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46, x_47, add_50, h_19], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf305, arg137_1, buf294, 786432, stream=stream0)
        buf306 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg138_1, buf306, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf308 = buf298; del buf298  # reuse
        buf309 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf307, arg139_1, buf308, buf309, 192, 4096, stream=stream0)
        buf311 = reinterpret_tensor(buf294, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf307, arg139_1, buf308, buf309, buf311, 768, 1024, stream=stream0)
        buf312 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, v_46, transpose_80, x_flat_14, v_t_x_13], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf311, (6, 128, 1024), (131072, 1024, 1), 0), out=buf312)
        buf313 = reinterpret_tensor(buf307, (6, 128, 1024), (131072, 1024, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [split_13, u_27, mixed_13], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 128, 12), (3200, 12, 1), 0), buf312, out=buf313)
        buf314 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf311, buf313, buf21, buf314, 768, 1024, stream=stream0)
        del buf311
        del buf313
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49, x_50], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf315 = extern_kernels.convolution(buf314, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf316 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49, x_50, add_53, h_20], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf316, arg141_1, buf305, 786432, stream=stream0)
        buf317 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg142_1, buf317, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf319 = buf309; del buf309  # reuse
        buf320 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf318, arg143_1, buf319, buf320, 192, 4096, stream=stream0)
        buf322 = reinterpret_tensor(buf305, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf318, arg143_1, buf319, buf320, buf322, 768, 1024, stream=stream0)
        buf323 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, v_48, transpose_81, x_flat_15, v_t_x_14], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf322, (6, 128, 1024), (131072, 1024, 1), 0), out=buf323)
        buf324 = reinterpret_tensor(buf318, (6, 128, 1024), (131072, 1024, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [split_14, u_29, mixed_14], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 128, 12), (3200, 12, 1), 0), buf323, out=buf324)
        buf325 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf322, buf324, buf21, buf325, 768, 1024, stream=stream0)
        del buf322
        del buf324
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52, x_53], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf326 = extern_kernels.convolution(buf325, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf327 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52, x_53, add_56, h_21], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf327, arg145_1, buf316, 786432, stream=stream0)
        buf328 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg146_1, buf328, 16384, 9, stream=stream0)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf327, buf328, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf330 = buf320; del buf320  # reuse
        buf331 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf329, arg147_1, buf330, buf331, 192, 4096, stream=stream0)
        buf333 = reinterpret_tensor(buf316, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf329, arg147_1, buf330, buf331, buf333, 768, 1024, stream=stream0)
        buf334 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, v_50, transpose_82, x_flat_16, v_t_x_15], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf333, (6, 128, 1024), (131072, 1024, 1), 0), out=buf334)
        buf335 = reinterpret_tensor(buf329, (6, 128, 1024), (131072, 1024, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [split_15, u_31, mixed_15], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (6, 128, 12), (3200, 12, 1), 0), buf334, out=buf335)
        buf336 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf333, buf335, buf21, buf336, 768, 1024, stream=stream0)
        del buf333
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf337 = extern_kernels.convolution(buf336, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf338 = buf331; del buf331  # reuse
        buf339 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63.run(buf337, arg149_1, buf327, buf338, buf339, 192, 4096, stream=stream0)
        buf341 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg25_1, buf341, 24, 1024, stream=stream0)
        buf342 = reinterpret_tensor(buf135, (128, 4, 3, 3), (36, 1, 12, 4), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(arg23_1, buf342, 512, 9, stream=stream0)
        del arg23_1
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf341, buf342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf341
        buf344 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [h_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf344, arg24_1, 786432, stream=stream0)
        del arg24_1
        buf345 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg26_1, buf345, 16384, 9, stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf344, buf345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf347 = empty_strided_cuda((6, 32, 1, 1), (32, 1, 192, 192), torch.float32)
        buf348 = empty_strided_cuda((6, 32, 1, 1), (32, 1, 192, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_57, x_norm_17], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf346, arg27_1, buf347, buf348, 192, 4096, stream=stream0)
        buf350 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [getitem_71, arange_1, mul_14, truediv_1, freqs_1, getitem_72, args_1, cos_1, sin_1, emb_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_cat_cos_div_exp_mul_sin_unsqueeze_5.run(arg0_1, buf350, 1536, stream=stream0)
        buf351 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [getitem_71, arange_1, mul_14, truediv_1, freqs_1, getitem_72, args_1, cos_1, sin_1, emb_1, input_45], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_arange_cat_cos_div_exp_mul_sin_unsqueeze_6.run(buf350, buf351, 2048, stream=stream0)
        buf352 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [getitem_71, arange_1, mul_14, truediv_1, freqs_1, getitem_72, args_1, cos_1, sin_1, emb_1, input_45], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten.addmm, aten.t]
        extern_kernels.mm(buf351, reinterpret_tensor(arg1_1, (256, 1024), (1, 256), 0), out=buf352)
        del arg1_1
        del buf351
        buf353 = reinterpret_tensor(buf261, (6, 1024), (1024, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_7.run(buf352, arg2_1, buf353, 6144, stream=stream0)
        del arg2_1
        del buf352
        buf356 = buf249; del buf249  # reuse
        buf354 = reinterpret_tensor(buf356, (6, 256), (512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_45, input_46, input_47], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg4_1, buf353, reinterpret_tensor(arg3_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf354)
        del arg3_1
        del arg4_1
        buf355 = reinterpret_tensor(buf356, (6, 256), (512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [unsqueeze_13, s_emb_1, cat_3], Original ATen: [aten.unsqueeze, aten.expand, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_expand_unsqueeze_64.run(arg154_1, buf355, 1536, stream=stream0)
        del arg154_1
        del buf354
        del buf355
        buf357 = reinterpret_tensor(buf123, (6, 512), (512, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf356, reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf357)
        del arg7_1
        buf358 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_9.run(buf358, arg8_1, 3072, stream=stream0)
        del arg8_1
        buf359 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49, input_50], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg10_1, buf358, reinterpret_tensor(arg9_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf359)
        del arg10_1
        del arg9_1
        buf360 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg11_1, (256, 256), (1, 256), 0), out=buf360)
        del arg11_1
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf361, arg12_1, 1536, stream=stream0)
        del arg12_1
        buf362 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52, input_53], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg14_1, buf361, reinterpret_tensor(arg13_1, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf362)
        del arg13_1
        del arg14_1
        buf363 = reinterpret_tensor(buf336, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf346, arg27_1, buf347, buf348, buf363, 768, 1024, stream=stream0)
        del arg27_1
        buf364 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, v_52, transpose_83, x_flat_17, v_t_x_16], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf363, (6, 128, 1024), (131072, 1024, 1), 0), out=buf364)
        buf365 = reinterpret_tensor(buf346, (6, 128, 1024), (131072, 1024, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [split_16, u_33, mixed_16], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 12), (3200, 12, 1), 0), buf364, out=buf365)
        buf366 = reinterpret_tensor(buf335, (6, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, x_flat_17, out_56, view_131, shift_33, out_57, x_58], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf363, buf365, buf362, buf366, 768, 1024, stream=stream0)
        del buf363
        del buf365
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, x_flat_17, out_56, view_131, shift_33, out_57, x_58, x_59], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf367 = extern_kernels.convolution(buf366, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg28_1
        buf368 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [x_57, x_norm_17, split_16, x_flat_17, out_56, view_131, shift_33, out_57, x_58, x_59, add_62, h_24], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf368, arg29_1, buf344, 786432, stream=stream0)
        del arg29_1
        buf369 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg30_1, buf369, 16384, 9, stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf368, buf369, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf369
        buf371 = buf348; del buf348  # reuse
        buf372 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf370, arg31_1, buf371, buf372, 192, 4096, stream=stream0)
        buf374 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg37_1, (256, 512), (1, 256), 0), out=buf374)
        del arg37_1
        buf375 = reinterpret_tensor(buf344, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf370, arg31_1, buf371, buf372, buf375, 768, 1024, stream=stream0)
        del arg31_1
        buf376 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, v_54, transpose_84, x_flat_18, v_t_x_17], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf375, (6, 128, 1024), (131072, 1024, 1), 0), out=buf376)
        buf377 = reinterpret_tensor(buf370, (6, 128, 1024), (131072, 1024, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [split_17, u_35, mixed_17], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 12), (3200, 12, 1), 0), buf376, out=buf377)
        del buf376
        buf378 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, x_flat_18, out_58, view_136, shift_35, out_59, x_61], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf375, buf377, buf362, buf378, 768, 1024, stream=stream0)
        del buf375
        del buf377
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, x_flat_18, out_58, view_136, shift_35, out_59, x_61, x_62], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf379 = extern_kernels.convolution(buf378, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg32_1
        del buf378
        buf380 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [x_60, x_norm_18, split_17, x_flat_18, out_58, view_136, shift_35, out_59, x_61, x_62, add_65, h_25], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf380, arg33_1, buf368, 786432, stream=stream0)
        del arg33_1
        del buf368
        buf381 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [h_26], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(arg34_1, buf381, 32768, 9, stream=stream0)
        del arg34_1
        # Topologically Sorted Source Nodes: [h_26], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf380, buf381, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf381
        buf383 = reinterpret_tensor(buf277, (6, 256, 256), (65536, 256, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [h_26, view_137, q_28, q_29], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_transpose_view_15.run(buf382, arg35_1, buf383, 393216, stream=stream0)
        buf384 = reinterpret_tensor(buf255, (1536, 256), (256, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [h_26, view_137, q_28, q_29], Original ATen: [aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (1536, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 256), (1, 256), 0), out=buf384)
        del arg36_1
        buf385 = reinterpret_tensor(buf383, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [q_29, view_138, q_30, matmul_20], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf384, buf385, 393216, stream=stream0)
        buf386 = reinterpret_tensor(buf361, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, view_139, k_20, transpose_89, matmul_20], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf374, buf386, 1536, stream=stream0)
        buf387 = reinterpret_tensor(buf353, (24, 256, 1), (256, 1, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, q_29, view_138, q_30, matmul_20, view_139, k_20, transpose_89], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf385, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf386, (24, 64, 1), (64, 1, 0), 0), out=buf387)
        buf388 = reinterpret_tensor(buf387, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [matmul_20, attn_21], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf388, 6144, stream=stream0)
        buf389 = reinterpret_tensor(buf386, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, view_140, v_56, matmul_21], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf374, buf389, 1536, stream=stream0)
        buf390 = reinterpret_tensor(buf385, (24, 256, 64), (16384, 64, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [kv_9, chunk_9, matmul_20, attn_21, matmul_21, view_140, v_56], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf388, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf389, (24, 1, 64), (64, 0, 1), 0), out=buf390)
        buf391 = reinterpret_tensor(buf384, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [matmul_21, transpose_90, out_60], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf390, buf391, 393216, stream=stream0)
        buf392 = reinterpret_tensor(buf390, (1536, 256), (256, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [matmul_21, transpose_90, out_60, input_60], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf391, (1536, 256), (256, 1), 0), reinterpret_tensor(arg38_1, (256, 256), (1, 256), 0), out=buf392)
        del arg38_1
        buf393 = reinterpret_tensor(buf392, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_21.run(buf393, buf382, arg35_1, arg39_1, 393216, stream=stream0)
        del arg39_1
        buf394 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg40_1, buf394, 65536, 9, stream=stream0)
        del arg40_1
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf395 = extern_kernels.convolution(buf393, buf394, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf396 = buf372; del buf372  # reuse
        buf397 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf395, arg41_1, buf396, buf397, 192, 2048, stream=stream0)
        buf399 = reinterpret_tensor(buf389, (6, 256), (256, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg15_1, (256, 256), (1, 256), 0), out=buf399)
        del arg15_1
        buf400 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf400, arg16_1, 1536, stream=stream0)
        del arg16_1
        buf401 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55, input_56], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg18_1, buf400, reinterpret_tensor(arg17_1, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf401)
        del arg17_1
        del arg18_1
        buf402 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg45_1, (256, 512), (1, 256), 0), out=buf402)
        del arg45_1
        buf403 = reinterpret_tensor(buf393, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf395, arg41_1, buf396, buf397, buf403, 1536, 256, stream=stream0)
        del arg41_1
        buf404 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, v_58, transpose_92, x_flat_19, v_t_x_18], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf403, (6, 256, 256), (65536, 256, 1), 0), out=buf404)
        buf405 = reinterpret_tensor(buf395, (6, 256, 256), (65536, 256, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [split_18, u_37, mixed_18], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 256, 12), (6400, 12, 1), 0), buf404, out=buf405)
        buf406 = reinterpret_tensor(buf391, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf403, buf405, buf401, buf406, 1536, 256, stream=stream0)
        del buf403
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf407 = extern_kernels.convolution(buf406, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg42_1
        buf408 = reinterpret_tensor(buf406, (6, 256, 256), (65536, 256, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, view_147, q_31, q_32], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_26.run(buf407, arg43_1, buf382, arg35_1, buf408, 393216, stream=stream0)
        buf409 = reinterpret_tensor(buf405, (1536, 256), (256, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, view_147, q_31, q_32], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (1536, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 256), (1, 256), 0), out=buf409)
        del arg44_1
        buf410 = reinterpret_tensor(buf408, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [q_32, view_148, q_33, matmul_22], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf409, buf410, 393216, stream=stream0)
        buf411 = reinterpret_tensor(buf400, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, view_149, k_22, transpose_97, matmul_22], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf402, buf411, 1536, stream=stream0)
        buf412 = reinterpret_tensor(buf388, (24, 256, 1), (256, 1, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, q_32, view_148, q_33, matmul_22, view_149, k_22, transpose_97], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf410, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf411, (24, 64, 1), (64, 1, 0), 0), out=buf412)
        buf413 = reinterpret_tensor(buf412, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [matmul_22, attn_23], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf413, 6144, stream=stream0)
        buf414 = reinterpret_tensor(buf411, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, view_150, v_60, matmul_23], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf402, buf414, 1536, stream=stream0)
        buf415 = reinterpret_tensor(buf410, (24, 256, 64), (16384, 64, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [kv_10, chunk_10, matmul_22, attn_23, matmul_23, view_150, v_60], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf413, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf414, (24, 1, 64), (64, 0, 1), 0), out=buf415)
        buf416 = reinterpret_tensor(buf409, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [matmul_23, transpose_98, out_64], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf415, buf416, 393216, stream=stream0)
        buf417 = reinterpret_tensor(buf415, (1536, 256), (256, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [matmul_23, transpose_98, out_64, input_62], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf416, (1536, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 256), (1, 256), 0), out=buf417)
        del arg46_1
        buf418 = reinterpret_tensor(buf417, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_27.run(buf418, buf407, arg43_1, buf382, arg35_1, arg47_1, 393216, stream=stream0)
        del arg47_1
        buf419 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg48_1, buf419, 65536, 9, stream=stream0)
        del arg48_1
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf420 = extern_kernels.convolution(buf418, buf419, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf419
        buf421 = buf397; del buf397  # reuse
        buf422 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf420, arg49_1, buf421, buf422, 192, 2048, stream=stream0)
        buf424 = reinterpret_tensor(buf413, (6, 1024), (1024, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg55_1, (256, 1024), (1, 256), 0), out=buf424)
        del arg55_1
        buf425 = reinterpret_tensor(buf418, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf420, arg49_1, buf421, buf422, buf425, 1536, 256, stream=stream0)
        del arg49_1
        buf426 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, v_62, transpose_100, x_flat_20, v_t_x_19], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf425, (6, 256, 256), (65536, 256, 1), 0), out=buf426)
        buf427 = reinterpret_tensor(buf420, (6, 256, 256), (65536, 256, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [split_19, u_39, mixed_19], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 256, 12), (6400, 12, 1), 0), buf426, out=buf427)
        buf428 = reinterpret_tensor(buf416, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, x_flat_20, out_66, view_156, shift_39, out_67, x_69], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf425, buf427, buf401, buf428, 1536, 256, stream=stream0)
        del buf425
        del buf427
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, x_flat_20, out_66, view_156, shift_39, out_67, x_69, x_70], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf429 = extern_kernels.convolution(buf428, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg50_1
        del buf428
        buf430 = buf429; del buf429  # reuse
        buf431 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [h_26, input_60, transpose_91, out_61, x_63, x_64, x_norm_19, split_18, x_flat_19, out_62, view_146, shift_37, out_63, x_65, x_66, add_69, h_27, input_62, transpose_99, out_65, x_67, x_68, x_norm_20, split_19, x_flat_20, out_66, view_156, shift_39, out_67, x_69, x_70, add_73, h_28], Original ATen: [aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_28.run(buf431, arg51_1, buf407, arg43_1, buf382, arg35_1, 393216, stream=stream0)
        del arg35_1
        del arg43_1
        del arg51_1
        del buf382
        del buf407
        buf432 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_28, h_29], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_29.run(arg52_1, buf432, 131072, 9, stream=stream0)
        del arg52_1
        # Topologically Sorted Source Nodes: [h_28, h_29], Original ATen: [aten.silu, aten.convolution]
        buf433 = extern_kernels.convolution(buf431, buf432, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf432
        buf434 = empty_strided_cuda((6, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_28, h_29, view_157, q_34, q_35], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_silu_transpose_view_30.run(buf433, arg53_1, buf434, 196608, stream=stream0)
        buf435 = empty_strided_cuda((384, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_28, h_29, view_157, q_34, q_35], Original ATen: [aten.silu, aten.convolution, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (384, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf435)
        del arg54_1
        buf436 = reinterpret_tensor(buf434, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [q_35, view_158, q_36, matmul_24], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_31.run(buf435, buf436, 196608, stream=stream0)
        buf437 = reinterpret_tensor(buf402, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, view_159, k_24, transpose_105, matmul_24], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_32.run(buf424, buf437, 3072, stream=stream0)
        buf438 = reinterpret_tensor(buf356, (48, 64, 1), (64, 1, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, q_35, view_158, q_36, matmul_24, view_159, k_24, transpose_105], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf437, (48, 64, 1), (64, 1, 0), 0), out=buf438)
        buf439 = reinterpret_tensor(buf438, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [matmul_24, attn_25], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_33.run(buf439, 3072, stream=stream0)
        buf440 = reinterpret_tensor(buf437, (6, 8, 1, 64), (512, 64, 64, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, view_160, v_64, matmul_25], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_34.run(buf424, buf440, 3072, stream=stream0)
        buf441 = reinterpret_tensor(buf436, (48, 64, 64), (4096, 64, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [kv_11, chunk_11, matmul_24, attn_25, matmul_25, view_160, v_64], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf439, (48, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf440, (48, 1, 64), (64, 0, 1), 0), out=buf441)
        buf442 = reinterpret_tensor(buf435, (6, 64, 8, 64), (32768, 512, 64, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [matmul_25, transpose_106, out_68], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_35.run(buf441, buf442, 196608, stream=stream0)
        buf443 = reinterpret_tensor(buf441, (384, 512), (512, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [matmul_25, transpose_106, out_68, input_64], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf442, (384, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 512), (1, 512), 0), out=buf443)
        del arg56_1
        buf444 = reinterpret_tensor(buf443, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_36.run(buf444, buf433, arg53_1, arg57_1, 196608, stream=stream0)
        del arg57_1
        buf445 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_37.run(arg58_1, buf445, 262144, 9, stream=stream0)
        del arg58_1
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add]
        buf446 = extern_kernels.convolution(buf444, buf445, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        buf447 = buf422; del buf422  # reuse
        buf448 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf446, arg59_1, buf447, buf448, 192, 1024, stream=stream0)
        buf450 = reinterpret_tensor(buf414, (6, 256), (256, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg19_1, (256, 256), (1, 256), 0), out=buf450)
        del arg19_1
        buf451 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf451, arg20_1, 1536, stream=stream0)
        del arg20_1
        buf452 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58, input_59], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.addmm(arg22_1, buf451, reinterpret_tensor(arg21_1, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf452)
        del arg21_1
        del arg22_1
        buf453 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg63_1, (256, 1024), (1, 256), 0), out=buf453)
        del arg63_1
        buf454 = reinterpret_tensor(buf444, (6, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39.run(buf446, arg59_1, buf447, buf448, buf454, 3072, 64, stream=stream0)
        del arg59_1
        buf455 = reinterpret_tensor(buf342, (6, 12, 64), (768, 64, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, v_66, transpose_108, x_flat_21, v_t_x_20], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf452, (6, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf454, (6, 512, 64), (32768, 64, 1), 0), out=buf455)
        buf456 = reinterpret_tensor(buf446, (6, 512, 64), (32768, 64, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [split_20, u_41, mixed_20], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf452, (6, 512, 12), (12800, 12, 1), 0), buf455, out=buf456)
        buf457 = reinterpret_tensor(buf442, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40.run(buf454, buf456, buf452, buf457, 3072, 64, stream=stream0)
        del buf454
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf458 = extern_kernels.convolution(buf457, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf458, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del arg60_1
        buf459 = reinterpret_tensor(buf457, (6, 64, 512), (32768, 512, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, view_167, q_37, q_38], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_41.run(buf458, arg61_1, buf433, arg53_1, buf459, 196608, stream=stream0)
        buf460 = reinterpret_tensor(buf456, (384, 512), (512, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, view_167, q_37, q_38], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (384, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 512), (1, 512), 0), out=buf460)
        del arg62_1
        buf461 = reinterpret_tensor(buf459, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [q_38, view_168, q_39, matmul_26], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_31.run(buf460, buf461, 196608, stream=stream0)
        buf462 = reinterpret_tensor(buf440, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, view_169, k_26, transpose_113, matmul_26], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_32.run(buf453, buf462, 3072, stream=stream0)
        buf463 = reinterpret_tensor(buf439, (48, 64, 1), (64, 1, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, q_38, view_168, q_39, matmul_26, view_169, k_26, transpose_113], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf461, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf462, (48, 64, 1), (64, 1, 0), 0), out=buf463)
        buf464 = reinterpret_tensor(buf463, (6, 8, 64, 1), (512, 64, 1, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [matmul_26, attn_27], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_33.run(buf464, 3072, stream=stream0)
        buf465 = reinterpret_tensor(buf462, (6, 8, 1, 64), (512, 64, 64, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, view_170, v_68, matmul_27], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_34.run(buf453, buf465, 3072, stream=stream0)
        buf466 = reinterpret_tensor(buf461, (48, 64, 64), (4096, 64, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [kv_12, chunk_12, matmul_26, attn_27, matmul_27, view_170, v_68], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf464, (48, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf465, (48, 1, 64), (64, 0, 1), 0), out=buf466)
        del buf464
        buf467 = reinterpret_tensor(buf460, (6, 64, 8, 64), (32768, 512, 64, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [matmul_27, transpose_114, out_72], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_35.run(buf466, buf467, 196608, stream=stream0)
        buf468 = reinterpret_tensor(buf466, (384, 512), (512, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [matmul_27, transpose_114, out_72, input_66], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf467, (384, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf468)
        del arg64_1
        buf469 = reinterpret_tensor(buf468, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_42.run(buf469, buf458, arg61_1, buf433, arg53_1, arg65_1, 196608, stream=stream0)
        del arg65_1
        buf470 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_silu_transpose_view_37.run(arg66_1, buf470, 262144, 9, stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf471 = extern_kernels.convolution(buf469, buf470, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del buf470
        buf472 = buf448; del buf448  # reuse
        buf473 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_38.run(buf471, arg67_1, buf472, buf473, 192, 1024, stream=stream0)
        buf475 = reinterpret_tensor(buf469, (6, 32, 16, 64), (32768, 1024, 64, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_transpose_view_39.run(buf471, arg67_1, buf472, buf473, buf475, 3072, 64, stream=stream0)
        del arg67_1
        buf476 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, v_70, transpose_116, x_flat_22, v_t_x_21], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf452, (6, 12, 512), (12800, 1, 12), 6144), reinterpret_tensor(buf475, (6, 512, 64), (32768, 64, 1), 0), out=buf476)
        buf477 = reinterpret_tensor(buf471, (6, 512, 64), (32768, 64, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [split_21, u_43, mixed_21], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf452, (6, 512, 12), (12800, 12, 1), 0), buf476, out=buf477)
        del buf476
        buf478 = reinterpret_tensor(buf467, (6, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, x_flat_22, out_74, view_176, shift_43, out_75, x_77], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_40.run(buf475, buf477, buf452, buf478, 3072, 64, stream=stream0)
        del buf452
        del buf475
        del buf477
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, x_flat_22, out_74, view_176, shift_43, out_75, x_77, x_78], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes]
        buf479 = extern_kernels.convolution(buf478, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (6, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
        del arg68_1
        buf480 = buf479; del buf479  # reuse
        buf484 = reinterpret_tensor(buf478, (6, 64, 512), (32768, 512, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [h_28, h_29, input_64, transpose_107, out_69, x_71, x_72, x_norm_21, split_20, x_flat_21, out_70, view_166, shift_41, out_71, x_73, x_74, add_77, h_30, input_66, transpose_115, out_73, x_75, x_76, x_norm_22, split_21, x_flat_22, out_74, view_176, shift_43, out_75, x_77, x_78, add_81, h_31, view_177, x_flat_23, x_norm_23], Original ATen: [aten.silu, aten.convolution, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_group_norm, aten.split_with_sizes, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_43.run(buf480, arg69_1, buf458, arg61_1, buf433, arg53_1, arg70_1, arg71_1, buf484, 384, 512, stream=stream0)
        del arg53_1
        del arg61_1
        del arg69_1
        del arg70_1
        del arg71_1
        buf485 = empty_strided_cuda((384, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, x_norm_23, qkv_3], Original ATen: [aten.silu, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf484, (384, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 1536), (1, 512), 0), out=buf485)
        del arg72_1
        buf486 = reinterpret_tensor(buf484, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, q_40, matmul_28], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_44.run(buf485, arg73_1, buf486, 196608, stream=stream0)
        buf487 = reinterpret_tensor(buf458, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, k_27, transpose_118, matmul_28], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_transpose_view_45.run(buf485, arg73_1, buf487, 3072, 64, stream=stream0)
        buf488 = reinterpret_tensor(buf433, (48, 64, 64), (4096, 64, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, q_40, matmul_28, k_27, transpose_118], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone, aten._unsafe_view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf486, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf487, (48, 64, 64), (4096, 64, 1), 0), out=buf488)
        buf491 = reinterpret_tensor(buf488, (6, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [matmul_28, attn_29], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_amax_mul_sub_view_46.run(buf491, 3072, 64, stream=stream0)
        buf492 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, v_71, out_76], Original ATen: [aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_permute_select_view_47.run(buf485, arg73_1, buf492, 196608, stream=stream0)
        del arg73_1
        buf493 = reinterpret_tensor(buf486, (48, 64, 64), (4096, 64, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [qkv_3, qkv_4, qkv_5, matmul_28, attn_29, out_76, v_71], Original ATen: [aten.addmm, aten.view, aten.permute, aten.mul, aten.sub, aten._softmax, aten.select, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf491, (48, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf492, (48, 64, 64), (4096, 64, 1), 0), out=buf493)
        del buf491
        buf494 = reinterpret_tensor(buf492, (6, 64, 8, 64), (32768, 512, 64, 1), 0); del buf492  # reuse
        # Topologically Sorted Source Nodes: [out_76, transpose_119, out_77], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_35.run(buf493, buf494, 196608, stream=stream0)
        buf495 = reinterpret_tensor(buf493, (384, 512), (512, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [out_76, transpose_119, out_77, out_78], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf494, (384, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf495)
        del arg74_1
        del buf494
        buf500 = reinterpret_tensor(buf480, (6, 64, 512), (32768, 512, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_silu_transpose_view_48.run(buf500, buf495, arg75_1, arg76_1, arg77_1, 384, 512, stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        del buf495
        buf499 = reinterpret_tensor(buf465, (6, 512), (512, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg85_1, (256, 512), (1, 256), 0), out=buf499)
        del arg85_1
        buf501 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80, transpose_120, out_81, h_32], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_layer_norm_silu_transpose_view_49.run(arg78_1, buf501, 131072, 16, stream=stream0)
        del arg78_1
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80, transpose_120, out_81, h_32], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution]
        buf502 = extern_kernels.convolution(reinterpret_tensor(buf500, (6, 512, 8, 8), (32768, 1, 4096, 512), 0), buf501, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf500
        del buf501
        buf503 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg80_1, (256, 256), (1, 256), 0), out=buf503)
        del arg80_1
        buf504 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf504, arg81_1, 1536, stream=stream0)
        del arg81_1
        buf505 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69, input_70], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf504, reinterpret_tensor(arg82_1, (256, 256), (1, 256), 0), out=buf505)
        del arg82_1
        del buf504
        buf506 = empty_strided_cuda((6, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_31, view_177, x_flat_23, out_78, out_79, out_80, transpose_120, out_81, h_32, input_70, input_71, unsqueeze_18, gate_2, h_16_gated_1, h_33], Original ATen: [aten.silu, aten.view, aten.transpose, aten.addmm, aten.add, aten.native_layer_norm, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_layer_norm_sigmoid_silu_transpose_unsqueeze_view_50.run(buf502, arg79_1, buf431, buf505, arg83_1, buf506, 1536, 256, stream=stream0)
        del arg79_1
        del arg83_1
        buf507 = reinterpret_tensor(buf502, (1536, 256), (1, 1536), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [view_179, q_41, q_42], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_t_transpose_view_51.run(buf506, buf507, 393216, stream=stream0)
        buf508 = reinterpret_tensor(buf431, (1536, 256), (256, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [view_179, q_41, q_42], Original ATen: [aten.view, aten.transpose, aten.t, aten.mm]
        extern_kernels.mm(buf507, reinterpret_tensor(arg84_1, (256, 256), (1, 256), 0), out=buf508)
        del arg84_1
        buf509 = reinterpret_tensor(buf507, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [q_42, view_180, q_43, matmul_30], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf508, buf509, 393216, stream=stream0)
        buf510 = reinterpret_tensor(buf505, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, view_181, k_29, transpose_125, matmul_30], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf499, buf510, 1536, stream=stream0)
        buf511 = reinterpret_tensor(buf453, (24, 256, 1), (256, 1, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, q_42, view_180, q_43, matmul_30, view_181, k_29, transpose_125], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf509, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf510, (24, 64, 1), (64, 1, 0), 0), out=buf511)
        buf512 = reinterpret_tensor(buf511, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [matmul_30, attn_31], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf512, 6144, stream=stream0)
        buf513 = reinterpret_tensor(buf510, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, view_182, v_73, matmul_31], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf499, buf513, 1536, stream=stream0)
        buf514 = reinterpret_tensor(buf509, (24, 256, 64), (16384, 64, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [kv_13, chunk_13, matmul_30, attn_31, matmul_31, view_182, v_73], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf512, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf513, (24, 1, 64), (64, 0, 1), 0), out=buf514)
        buf515 = reinterpret_tensor(buf508, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [matmul_31, transpose_126, out_82], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf514, buf515, 393216, stream=stream0)
        buf516 = reinterpret_tensor(buf514, (1536, 256), (256, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [matmul_31, transpose_126, out_82, input_72], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf515, (1536, 256), (256, 1), 0), reinterpret_tensor(arg86_1, (256, 256), (1, 256), 0), out=buf516)
        del arg86_1
        buf517 = reinterpret_tensor(buf516, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf516  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_transpose_view_52.run(buf517, buf506, arg87_1, 1536, 256, stream=stream0)
        del arg87_1
        buf518 = reinterpret_tensor(buf485, (256, 256, 3, 3), (2304, 1, 768, 256), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg88_1, buf518, 65536, 9, stream=stream0)
        del arg88_1
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf519 = extern_kernels.convolution(buf517, buf518, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf520 = buf473; del buf473  # reuse
        buf521 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf519, arg89_1, buf520, buf521, 192, 2048, stream=stream0)
        buf523 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg93_1, (256, 512), (1, 256), 0), out=buf523)
        del arg93_1
        buf524 = reinterpret_tensor(buf517, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf519, arg89_1, buf520, buf521, buf524, 1536, 256, stream=stream0)
        del arg89_1
        buf525 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, v_75, transpose_128, x_flat_24, v_t_x_22], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf524, (6, 256, 256), (65536, 256, 1), 0), out=buf525)
        buf526 = reinterpret_tensor(buf519, (6, 256, 256), (65536, 256, 1), 0); del buf519  # reuse
        # Topologically Sorted Source Nodes: [split_22, u_45, mixed_22], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 256, 12), (6400, 12, 1), 0), buf525, out=buf526)
        buf527 = reinterpret_tensor(buf515, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf524, buf526, buf401, buf527, 1536, 256, stream=stream0)
        del buf524
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf528 = extern_kernels.convolution(buf527, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg90_1
        buf529 = reinterpret_tensor(buf527, (6, 256, 256), (65536, 256, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, view_189, q_44, q_45], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_53.run(buf528, arg91_1, buf506, buf529, 1536, 256, stream=stream0)
        buf530 = reinterpret_tensor(buf526, (1536, 256), (256, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, view_189, q_44, q_45], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (1536, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf530)
        del arg92_1
        buf531 = reinterpret_tensor(buf529, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [q_45, view_190, q_46, matmul_32], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf530, buf531, 393216, stream=stream0)
        buf532 = reinterpret_tensor(buf513, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, view_191, k_31, transpose_133, matmul_32], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf523, buf532, 1536, stream=stream0)
        buf533 = reinterpret_tensor(buf512, (24, 256, 1), (256, 1, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, q_45, view_190, q_46, matmul_32, view_191, k_31, transpose_133], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf531, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf532, (24, 64, 1), (64, 1, 0), 0), out=buf533)
        buf534 = reinterpret_tensor(buf533, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [matmul_32, attn_33], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf534, 6144, stream=stream0)
        buf535 = reinterpret_tensor(buf532, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, view_192, v_77, matmul_33], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf523, buf535, 1536, stream=stream0)
        buf536 = reinterpret_tensor(buf531, (24, 256, 64), (16384, 64, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [kv_14, chunk_14, matmul_32, attn_33, matmul_33, view_192, v_77], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf534, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf535, (24, 1, 64), (64, 0, 1), 0), out=buf536)
        buf537 = reinterpret_tensor(buf530, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [matmul_33, transpose_134, out_86], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf536, buf537, 393216, stream=stream0)
        buf538 = reinterpret_tensor(buf536, (1536, 256), (256, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [matmul_33, transpose_134, out_86, input_74], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf537, (1536, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 256), (1, 256), 0), out=buf538)
        del arg94_1
        buf539 = reinterpret_tensor(buf538, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf538  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_54.run(buf539, buf528, arg91_1, buf506, arg95_1, 1536, 256, stream=stream0)
        del arg95_1
        buf540 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg96_1, buf540, 65536, 9, stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf541 = extern_kernels.convolution(buf539, buf540, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf542 = buf521; del buf521  # reuse
        buf543 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf541, arg97_1, buf542, buf543, 192, 2048, stream=stream0)
        buf545 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg101_1, (256, 512), (1, 256), 0), out=buf545)
        del arg101_1
        buf546 = reinterpret_tensor(buf539, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf541, arg97_1, buf542, buf543, buf546, 1536, 256, stream=stream0)
        del arg97_1
        buf547 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, v_79, transpose_136, x_flat_25, v_t_x_23], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf546, (6, 256, 256), (65536, 256, 1), 0), out=buf547)
        buf548 = reinterpret_tensor(buf541, (6, 256, 256), (65536, 256, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [split_23, u_47, mixed_23], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 256, 12), (6400, 12, 1), 0), buf547, out=buf548)
        buf549 = reinterpret_tensor(buf537, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf537  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, x_flat_25, out_88, view_198, shift_47, out_89, x_85], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf546, buf548, buf401, buf549, 1536, 256, stream=stream0)
        del buf546
        del buf548
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, x_flat_25, out_88, view_198, shift_47, out_89, x_85, x_86], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        buf550 = extern_kernels.convolution(buf549, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg98_1
        buf551 = buf550; del buf550  # reuse
        buf552 = reinterpret_tensor(buf549, (6, 256, 256), (65536, 256, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [input_72, transpose_127, out_83, x_79, x_80, x_norm_24, split_22, x_flat_24, out_84, view_188, shift_45, out_85, x_81, x_82, add_87, h_34, input_74, transpose_135, out_87, x_83, x_84, x_norm_25, split_23, x_flat_25, out_88, view_198, shift_47, out_89, x_85, x_86, add_91, h_35, view_199, q_47, q_48], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_55.run(buf551, arg99_1, buf528, arg91_1, buf506, buf552, 393216, stream=stream0)
        del arg91_1
        del arg99_1
        del buf506
        buf553 = reinterpret_tensor(buf528, (1536, 256), (256, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [h_35, view_199, q_47, q_48], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (1536, 256), (256, 1), 0), reinterpret_tensor(arg100_1, (256, 256), (1, 256), 0), out=buf553)
        del arg100_1
        buf554 = reinterpret_tensor(buf552, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [q_48, view_200, q_49, matmul_34], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf553, buf554, 393216, stream=stream0)
        buf555 = reinterpret_tensor(buf535, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, view_201, k_33, transpose_141, matmul_34], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf545, buf555, 1536, stream=stream0)
        buf556 = reinterpret_tensor(buf534, (24, 256, 1), (256, 1, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, q_48, view_200, q_49, matmul_34, view_201, k_33, transpose_141], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf554, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf555, (24, 64, 1), (64, 1, 0), 0), out=buf556)
        buf557 = reinterpret_tensor(buf556, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [matmul_34, attn_35], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf557, 6144, stream=stream0)
        buf558 = reinterpret_tensor(buf555, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, view_202, v_81, matmul_35], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf545, buf558, 1536, stream=stream0)
        buf559 = reinterpret_tensor(buf554, (24, 256, 64), (16384, 64, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [kv_15, chunk_15, matmul_34, attn_35, matmul_35, view_202, v_81], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf558, (24, 1, 64), (64, 0, 1), 0), out=buf559)
        buf560 = reinterpret_tensor(buf553, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [matmul_35, transpose_142, out_90], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf559, buf560, 393216, stream=stream0)
        buf561 = reinterpret_tensor(buf559, (1536, 256), (256, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [matmul_35, transpose_142, out_90, input_76], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf560, (1536, 256), (256, 1), 0), reinterpret_tensor(arg102_1, (256, 256), (1, 256), 0), out=buf561)
        del arg102_1
        buf562 = reinterpret_tensor(buf561, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_56.run(buf562, buf551, arg103_1, 393216, stream=stream0)
        del arg103_1
        buf563 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg104_1, buf563, 65536, 9, stream=stream0)
        del arg104_1
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf564 = extern_kernels.convolution(buf562, buf563, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf564, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf565 = buf543; del buf543  # reuse
        buf566 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf564, arg105_1, buf565, buf566, 192, 2048, stream=stream0)
        buf568 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg109_1, (256, 512), (1, 256), 0), out=buf568)
        del arg109_1
        buf569 = reinterpret_tensor(buf562, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf564, arg105_1, buf565, buf566, buf569, 1536, 256, stream=stream0)
        del arg105_1
        buf570 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, v_83, transpose_144, x_flat_26, v_t_x_24], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf569, (6, 256, 256), (65536, 256, 1), 0), out=buf570)
        buf571 = reinterpret_tensor(buf564, (6, 256, 256), (65536, 256, 1), 0); del buf564  # reuse
        # Topologically Sorted Source Nodes: [split_24, u_49, mixed_24], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 256, 12), (6400, 12, 1), 0), buf570, out=buf571)
        buf572 = reinterpret_tensor(buf560, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf569, buf571, buf401, buf572, 1536, 256, stream=stream0)
        del buf569
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf573 = extern_kernels.convolution(buf572, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf573, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg106_1
        buf574 = reinterpret_tensor(buf572, (6, 256, 256), (65536, 256, 1), 0); del buf572  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, view_209, q_50, q_51], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_57.run(buf573, arg107_1, buf551, buf574, 393216, stream=stream0)
        buf575 = reinterpret_tensor(buf571, (1536, 256), (256, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, view_209, q_50, q_51], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (1536, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), out=buf575)
        del arg108_1
        buf576 = reinterpret_tensor(buf574, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [q_51, view_210, q_52, matmul_36], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf575, buf576, 393216, stream=stream0)
        buf577 = reinterpret_tensor(buf558, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, view_211, k_35, transpose_149, matmul_36], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf568, buf577, 1536, stream=stream0)
        buf578 = reinterpret_tensor(buf557, (24, 256, 1), (256, 1, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, q_51, view_210, q_52, matmul_36, view_211, k_35, transpose_149], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf576, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf577, (24, 64, 1), (64, 1, 0), 0), out=buf578)
        buf579 = reinterpret_tensor(buf578, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [matmul_36, attn_37], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf579, 6144, stream=stream0)
        buf580 = reinterpret_tensor(buf577, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, view_212, v_85, matmul_37], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf568, buf580, 1536, stream=stream0)
        buf581 = reinterpret_tensor(buf576, (24, 256, 64), (16384, 64, 1), 0); del buf576  # reuse
        # Topologically Sorted Source Nodes: [kv_16, chunk_16, matmul_36, attn_37, matmul_37, view_212, v_85], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf579, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf580, (24, 1, 64), (64, 0, 1), 0), out=buf581)
        buf582 = reinterpret_tensor(buf575, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf575  # reuse
        # Topologically Sorted Source Nodes: [matmul_37, transpose_150, out_94], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf581, buf582, 393216, stream=stream0)
        buf583 = reinterpret_tensor(buf581, (1536, 256), (256, 1), 0); del buf581  # reuse
        # Topologically Sorted Source Nodes: [matmul_37, transpose_150, out_94, input_78], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf582, (1536, 256), (256, 1), 0), reinterpret_tensor(arg110_1, (256, 256), (1, 256), 0), out=buf583)
        del arg110_1
        buf584 = reinterpret_tensor(buf583, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_58.run(buf584, buf573, arg107_1, buf551, arg111_1, 393216, stream=stream0)
        del arg111_1
        buf585 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg112_1, buf585, 65536, 9, stream=stream0)
        del arg112_1
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf586 = extern_kernels.convolution(buf584, buf585, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        buf587 = buf566; del buf566  # reuse
        buf588 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf586, arg113_1, buf587, buf588, 192, 2048, stream=stream0)
        buf590 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.t, aten.mm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg117_1, (256, 512), (1, 256), 0), out=buf590)
        del arg117_1
        buf591 = reinterpret_tensor(buf584, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf586, arg113_1, buf587, buf588, buf591, 1536, 256, stream=stream0)
        del arg113_1
        buf592 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, v_87, transpose_152, x_flat_27, v_t_x_25], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf591, (6, 256, 256), (65536, 256, 1), 0), out=buf592)
        buf593 = reinterpret_tensor(buf586, (6, 256, 256), (65536, 256, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [split_25, u_51, mixed_25], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 256, 12), (6400, 12, 1), 0), buf592, out=buf593)
        buf594 = reinterpret_tensor(buf582, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, x_flat_27, out_96, view_218, shift_51, out_97, x_93], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf591, buf593, buf401, buf594, 1536, 256, stream=stream0)
        del buf591
        del buf593
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, x_flat_27, out_96, view_218, shift_51, out_97, x_93, x_94], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf595 = extern_kernels.convolution(buf594, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf595, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg114_1
        buf596 = buf595; del buf595  # reuse
        buf597 = reinterpret_tensor(buf594, (6, 256, 256), (65536, 256, 1), 0); del buf594  # reuse
        # Topologically Sorted Source Nodes: [h_35, input_76, transpose_143, out_91, x_87, x_88, x_norm_26, split_24, x_flat_26, out_92, view_208, shift_49, out_93, x_89, x_90, add_95, h_36, input_78, transpose_151, out_95, x_91, x_92, x_norm_27, split_25, x_flat_27, out_96, view_218, shift_51, out_97, x_93, x_94, add_99, h_37, view_219, q_53, q_54], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_59.run(buf596, arg115_1, buf573, arg107_1, buf551, buf597, 393216, stream=stream0)
        del arg107_1
        del arg115_1
        del buf551
        buf598 = reinterpret_tensor(buf573, (1536, 256), (256, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [h_37, view_219, q_53, q_54], Original ATen: [aten.silu, aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (1536, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 256), (1, 256), 0), out=buf598)
        del arg116_1
        buf599 = reinterpret_tensor(buf597, (6, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [q_54, view_220, q_55, matmul_38], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_transpose_view_16.run(buf598, buf599, 393216, stream=stream0)
        buf600 = reinterpret_tensor(buf580, (6, 4, 64, 1), (256, 64, 1, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, view_221, k_37, transpose_157, matmul_38], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_17.run(buf590, buf600, 1536, stream=stream0)
        buf601 = reinterpret_tensor(buf579, (24, 256, 1), (256, 1, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, q_54, view_220, q_55, matmul_38, view_221, k_37, transpose_157], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.clone, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf599, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf600, (24, 64, 1), (64, 1, 0), 0), out=buf601)
        buf602 = reinterpret_tensor(buf601, (6, 4, 256, 1), (1024, 256, 1, 1), 0); del buf601  # reuse
        # Topologically Sorted Source Nodes: [matmul_38, attn_39], Original ATen: [aten.view, aten.mul, aten.amax, aten.sub, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_amax_mul_sub_view_18.run(buf602, 6144, stream=stream0)
        buf603 = reinterpret_tensor(buf600, (6, 4, 1, 64), (256, 64, 64, 1), 0); del buf600  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, view_222, v_89, matmul_39], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_split_transpose_unsqueeze_view_19.run(buf590, buf603, 1536, stream=stream0)
        del buf590
        buf604 = reinterpret_tensor(buf599, (24, 256, 64), (16384, 64, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [kv_17, chunk_17, matmul_38, attn_39, matmul_39, view_222, v_89], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.mul, aten.amax, aten.sub, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf602, (24, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf603, (24, 1, 64), (64, 0, 1), 0), out=buf604)
        del buf602
        buf605 = reinterpret_tensor(buf598, (6, 256, 4, 64), (65536, 256, 64, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [matmul_39, transpose_158, out_98], Original ATen: [aten.view, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_transpose_view_20.run(buf604, buf605, 393216, stream=stream0)
        buf606 = reinterpret_tensor(buf604, (1536, 256), (256, 1), 0); del buf604  # reuse
        # Topologically Sorted Source Nodes: [matmul_39, transpose_158, out_98, input_80], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf605, (1536, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 256), (1, 256), 0), out=buf606)
        del arg118_1
        buf607 = reinterpret_tensor(buf606, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_silu_transpose_view_56.run(buf607, buf596, arg119_1, 393216, stream=stream0)
        del arg119_1
        buf608 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_transpose_view_22.run(arg120_1, buf608, 65536, 9, stream=stream0)
        del arg120_1
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
        buf609 = extern_kernels.convolution(buf607, buf608, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del buf608
        buf610 = buf588; del buf588  # reuse
        buf611 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_convolution_native_group_norm_transpose_view_23.run(buf609, arg121_1, buf610, buf611, 192, 2048, stream=stream0)
        buf613 = reinterpret_tensor(buf607, (6, 32, 8, 256), (65536, 2048, 256, 1), 0); del buf607  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_transpose_view_24.run(buf609, arg121_1, buf610, buf611, buf613, 1536, 256, stream=stream0)
        del arg121_1
        buf614 = buf592; del buf592  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, v_91, transpose_160, x_flat_28, v_t_x_26], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 12, 256), (6400, 1, 12), 3072), reinterpret_tensor(buf613, (6, 256, 256), (65536, 256, 1), 0), out=buf614)
        buf615 = reinterpret_tensor(buf609, (6, 256, 256), (65536, 256, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [split_26, u_53, mixed_26], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (6, 256, 12), (6400, 12, 1), 0), buf614, out=buf615)
        del buf614
        buf616 = reinterpret_tensor(buf605, (6, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_25.run(buf613, buf615, buf401, buf616, 1536, 256, stream=stream0)
        del buf401
        del buf613
        del buf615
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf617 = extern_kernels.convolution(buf616, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (6, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
        del arg122_1
        del buf616
        buf618 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_60.run(buf618, arg123_1, buf596, 393216, stream=stream0)
        del arg123_1
        del buf596
        buf619 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38, h_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_native_group_norm_silu_split_with_sizes_transpose_view_61.run(arg124_1, buf619, 32768, 16, stream=stream0)
        del arg124_1
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38, h_39], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes]
        buf620 = extern_kernels.convolution(buf618, buf619, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf618
        del buf619
        buf621 = reinterpret_tensor(buf603, (6, 256), (256, 1), 0); del buf603  # reuse
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf359, reinterpret_tensor(arg126_1, (256, 256), (1, 256), 0), out=buf621)
        del arg126_1
        del buf359
        buf622 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten.addmm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_silu_10.run(buf622, arg127_1, 1536, stream=stream0)
        del arg127_1
        buf623 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten.addmm, aten.silu, aten.t]
        extern_kernels.mm(buf622, reinterpret_tensor(arg128_1, (256, 128), (1, 256), 0), out=buf623)
        del arg128_1
        del buf622
        buf624 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [h_37, input_80, transpose_159, out_99, x_95, x_96, x_norm_28, split_26, x_flat_28, out_100, view_228, shift_53, out_101, x_97, x_98, add_103, h_38, h_39, input_84, input_85, unsqueeze_25, gate_3, h_32_gated_1, h_40], Original ATen: [aten.silu, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.sigmoid, aten.unsqueeze, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_mul_native_group_norm_sigmoid_silu_split_with_sizes_transpose_unsqueeze_view_62.run(buf624, arg125_1, buf380, buf623, arg129_1, 786432, stream=stream0)
        del arg125_1
        del arg129_1
        del buf623
        buf625 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg130_1, buf625, 16384, 9, stream=stream0)
        del arg130_1
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf624, buf625, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf626, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf627 = buf611; del buf611  # reuse
        buf628 = buf610; del buf610  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf626, arg131_1, buf627, buf628, 192, 4096, stream=stream0)
        buf630 = reinterpret_tensor(buf380, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf626, arg131_1, buf627, buf628, buf630, 768, 1024, stream=stream0)
        del arg131_1
        buf631 = empty_strided_cuda((6, 12, 1024), (12288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, v_93, transpose_161, x_flat_29, v_t_x_27], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf630, (6, 128, 1024), (131072, 1024, 1), 0), out=buf631)
        buf632 = reinterpret_tensor(buf626, (6, 128, 1024), (131072, 1024, 1), 0); del buf626  # reuse
        # Topologically Sorted Source Nodes: [split_27, u_55, mixed_27], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 12), (3200, 12, 1), 0), buf631, out=buf632)
        buf633 = empty_strided_cuda((6, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, x_flat_29, out_102, view_233, shift_55, out_103, x_100], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf630, buf632, buf362, buf633, 768, 1024, stream=stream0)
        del buf630
        del buf632
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, x_flat_29, out_102, view_233, shift_55, out_103, x_100, x_101], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf634 = extern_kernels.convolution(buf633, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf634, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg132_1
        buf635 = buf634; del buf634  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_norm_29, split_27, x_flat_29, out_102, view_233, shift_55, out_103, x_100, x_101, add_107, h_41], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf635, arg133_1, buf624, 786432, stream=stream0)
        del arg133_1
        buf636 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg134_1, buf636, 16384, 9, stream=stream0)
        del arg134_1
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf637 = extern_kernels.convolution(buf635, buf636, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf637, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf638 = buf628; del buf628  # reuse
        buf639 = buf627; del buf627  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf637, arg135_1, buf638, buf639, 192, 4096, stream=stream0)
        buf641 = reinterpret_tensor(buf624, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf637, arg135_1, buf638, buf639, buf641, 768, 1024, stream=stream0)
        del arg135_1
        buf642 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, v_95, transpose_162, x_flat_30, v_t_x_28], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf641, (6, 128, 1024), (131072, 1024, 1), 0), out=buf642)
        buf643 = reinterpret_tensor(buf637, (6, 128, 1024), (131072, 1024, 1), 0); del buf637  # reuse
        # Topologically Sorted Source Nodes: [split_28, u_57, mixed_28], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 12), (3200, 12, 1), 0), buf642, out=buf643)
        buf644 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, x_flat_30, out_104, view_238, shift_57, out_105, x_103], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf641, buf643, buf362, buf644, 768, 1024, stream=stream0)
        del buf641
        del buf643
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, x_flat_30, out_104, view_238, shift_57, out_105, x_103, x_104], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf645 = extern_kernels.convolution(buf644, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg136_1
        buf646 = buf645; del buf645  # reuse
        # Topologically Sorted Source Nodes: [x_102, x_norm_30, split_28, x_flat_30, out_104, view_238, shift_57, out_105, x_103, x_104, add_110, h_42], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf646, arg137_1, buf635, 786432, stream=stream0)
        del arg137_1
        buf647 = buf636; del buf636  # reuse
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg138_1, buf647, 16384, 9, stream=stream0)
        del arg138_1
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf648 = extern_kernels.convolution(buf646, buf647, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf648, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf649 = buf639; del buf639  # reuse
        buf650 = buf638; del buf638  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf648, arg139_1, buf649, buf650, 192, 4096, stream=stream0)
        buf652 = reinterpret_tensor(buf635, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf648, arg139_1, buf649, buf650, buf652, 768, 1024, stream=stream0)
        del arg139_1
        buf653 = buf642; del buf642  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, v_97, transpose_163, x_flat_31, v_t_x_29], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf652, (6, 128, 1024), (131072, 1024, 1), 0), out=buf653)
        buf654 = reinterpret_tensor(buf648, (6, 128, 1024), (131072, 1024, 1), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [split_29, u_59, mixed_29], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 12), (3200, 12, 1), 0), buf653, out=buf654)
        buf655 = buf644; del buf644  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, x_flat_31, out_106, view_243, shift_59, out_107, x_106], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf652, buf654, buf362, buf655, 768, 1024, stream=stream0)
        del buf652
        del buf654
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, x_flat_31, out_106, view_243, shift_59, out_107, x_106, x_107], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf656 = extern_kernels.convolution(buf655, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg140_1
        buf657 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [x_105, x_norm_31, split_29, x_flat_31, out_106, view_243, shift_59, out_107, x_106, x_107, add_113, h_43], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf657, arg141_1, buf646, 786432, stream=stream0)
        del arg141_1
        buf658 = buf647; del buf647  # reuse
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg142_1, buf658, 16384, 9, stream=stream0)
        del arg142_1
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(buf657, buf658, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf659, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        buf660 = buf650; del buf650  # reuse
        buf661 = buf649; del buf649  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf659, arg143_1, buf660, buf661, 192, 4096, stream=stream0)
        buf663 = reinterpret_tensor(buf646, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf646  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf659, arg143_1, buf660, buf661, buf663, 768, 1024, stream=stream0)
        del arg143_1
        buf664 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, v_99, transpose_164, x_flat_32, v_t_x_30], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf663, (6, 128, 1024), (131072, 1024, 1), 0), out=buf664)
        buf665 = reinterpret_tensor(buf659, (6, 128, 1024), (131072, 1024, 1), 0); del buf659  # reuse
        # Topologically Sorted Source Nodes: [split_30, u_61, mixed_30], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 12), (3200, 12, 1), 0), buf664, out=buf665)
        buf666 = buf655; del buf655  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, x_flat_32, out_108, view_248, shift_61, out_109, x_109], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf663, buf665, buf362, buf666, 768, 1024, stream=stream0)
        del buf663
        del buf665
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, x_flat_32, out_108, view_248, shift_61, out_109, x_109, x_110], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf667 = extern_kernels.convolution(buf666, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf667, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg144_1
        buf668 = buf667; del buf667  # reuse
        # Topologically Sorted Source Nodes: [x_108, x_norm_32, split_30, x_flat_32, out_108, view_248, shift_61, out_109, x_109, x_110, add_116, h_44], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_13.run(buf668, arg145_1, buf657, 786432, stream=stream0)
        del arg145_1
        buf669 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(arg146_1, buf669, 16384, 9, stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf670 = extern_kernels.convolution(buf668, buf669, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf670, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del buf669
        buf671 = buf661; del buf661  # reuse
        buf672 = buf660; del buf660  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf670, arg147_1, buf671, buf672, 192, 4096, stream=stream0)
        buf674 = reinterpret_tensor(buf657, (6, 32, 4, 1024), (131072, 4096, 1024, 1), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_11.run(buf670, arg147_1, buf671, buf672, buf674, 768, 1024, stream=stream0)
        del arg147_1
        buf675 = buf664; del buf664  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, v_101, transpose_165, x_flat_33, v_t_x_31], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 12, 128), (3200, 1, 12), 1536), reinterpret_tensor(buf674, (6, 128, 1024), (131072, 1024, 1), 0), out=buf675)
        buf676 = reinterpret_tensor(buf670, (6, 128, 1024), (131072, 1024, 1), 0); del buf670  # reuse
        # Topologically Sorted Source Nodes: [split_31, u_63, mixed_31], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 12), (3200, 12, 1), 0), buf675, out=buf676)
        del buf675
        buf677 = buf666; del buf666  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_12.run(buf674, buf676, buf362, buf677, 768, 1024, stream=stream0)
        del buf362
        del buf674
        del buf676
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112, x_113], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        buf678 = extern_kernels.convolution(buf677, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf678, (6, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
        del arg148_1
        del buf677
        buf679 = buf672; del buf672  # reuse
        buf680 = buf671; del buf671  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112, x_113, add_119, h_45, input_86], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_63.run(buf678, arg149_1, buf668, buf679, buf680, 192, 4096, stream=stream0)
        buf682 = empty_strided_cuda((6, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [lt, zeros_like, full_like, sigma], Original ATen: [aten.lt, aten.zeros_like, aten.full_like, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_like_lt_where_zeros_like_65.run(arg0_1, buf682, 6, stream=stream0)
        del arg0_1
        buf683 = buf337; del buf337  # reuse
        buf684 = buf683; del buf683  # reuse
        buf690 = buf678; del buf678  # reuse
        buf691 = buf690; del buf690  # reuse
        # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56, add_59, h_22, input_42, x_111, x_norm_33, split_31, x_flat_33, out_110, view_253, shift_63, out_111, x_112, x_113, add_119, h_45, input_86, input_43, input_87], Original ATen: [aten.convolution, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_66.run(buf684, buf691, arg149_1, buf327, buf338, buf339, arg150_1, arg151_1, buf668, buf679, buf680, 786432, stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del buf327
        del buf338
        del buf339
        del buf668
        del buf679
        del buf680
        buf685 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        buf692 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44, input_87, input_88], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_67.run(arg152_1, buf685, buf692, 512, 9, stream=stream0)
        del arg152_1
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten.convolution]
        buf686 = extern_kernels.convolution(buf684, buf685, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (6, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
        del buf684
        del buf685
        # Topologically Sorted Source Nodes: [input_87, input_88], Original ATen: [aten.silu, aten.convolution]
        buf693 = extern_kernels.convolution(buf691, buf692, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (6, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
        del buf691
        del buf692
        buf697 = buf693; del buf693  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44, std_target, input_87, input_88, sub, mul_28, v_pred, std_pred, add_121, rescale_factor, v_pred_rescaled, mul_30, mul_31, v_guided, mul_32], Original ATen: [aten.silu, aten.convolution, aten.std, aten.sub, aten.mul, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_mul_silu_std_sub_68.run(buf697, buf686, arg153_1, 6, 4096, stream=stream0)
        del arg153_1
        buf698 = reinterpret_tensor(buf686, (6, 4, 32, 32), (4096, 1024, 32, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [x_next], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_69.run(arg25_1, buf697, buf698, 24, 1024, stream=stream0)
        del arg25_1
        del buf697
        buf699 = empty_strided_cuda((), (), torch.bool)
        # Topologically Sorted Source Nodes: [gt, add_noise], Original ATen: [aten.gt, aten.any]
        stream0 = get_raw_stream(0)
        triton_poi_fused_any_gt_70.run(buf682, buf699, 1, stream=stream0)
    return (buf682, buf698, buf699, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1 = args
        args.clear()
        partition0_args = [arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg154_1, arg150_1, arg151_1, arg152_1, arg153_1]
        del arg25_1, arg23_1, arg24_1, arg26_1, arg27_1, arg0_1, arg1_1, arg2_1, arg4_1, arg3_1, arg6_1, arg5_1, arg7_1, arg8_1, arg10_1, arg9_1, arg11_1, arg12_1, arg14_1, arg13_1, arg28_1, arg29_1, arg30_1, arg31_1, arg37_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg15_1, arg16_1, arg18_1, arg17_1, arg45_1, arg42_1, arg43_1, arg44_1, arg46_1, arg47_1, arg48_1, arg49_1, arg55_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg56_1, arg57_1, arg58_1, arg59_1, arg19_1, arg20_1, arg22_1, arg21_1, arg63_1, arg60_1, arg61_1, arg62_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg85_1, arg78_1, arg80_1, arg81_1, arg82_1, arg79_1, arg83_1, arg84_1, arg86_1, arg87_1, arg88_1, arg89_1, arg93_1, arg90_1, arg91_1, arg92_1, arg94_1, arg95_1, arg96_1, arg97_1, arg101_1, arg98_1, arg99_1, arg100_1, arg102_1, arg103_1, arg104_1, arg105_1, arg109_1, arg106_1, arg107_1, arg108_1, arg110_1, arg111_1, arg112_1, arg113_1, arg117_1, arg114_1, arg115_1, arg116_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg126_1, arg127_1, arg128_1, arg125_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg154_1, arg150_1, arg151_1, arg152_1, arg153_1
        (buf682, buf698, buf699) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf699, buf698, buf682, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.int64)
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
    arg25_1 = rand_strided((6, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
