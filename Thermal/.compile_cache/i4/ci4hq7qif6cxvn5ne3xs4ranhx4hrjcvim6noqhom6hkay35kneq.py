# AOT ID: ['4_inference']
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/jq/cjq34lx2ug3cctttdh4vzh5gxcjoipk2pxloa72vspka3qp732l5.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h => convert_element_type_62, convert_element_type_63, convert_element_type_64, convolution
# Graph fragment:
#   %arg27_1 : Tensor "f32[s13, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg27_1]
#   %buf0 : Tensor "bf16[s13, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=buf0]
#   %convert_element_type_64 : Tensor "bf16[s13, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_63 : Tensor "bf16[128, 4, 3, 3][36, 1, 12, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_64, %convert_element_type_63, %convert_element_type_62, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0,%buf3
triton_poi_fused__to_copy_convolution_0 = async_compile.triton('triton_poi_fused__to_copy_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_0(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 1024
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 4)
    y3 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x1 + 1024*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr1 + (y2 + 4*x1 + 4096*y3), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/nr/cnrlbxaqa2tzaywgyuzcoy3quabh3p7wwgin7xpol4i2y5qqigbe.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h => convert_element_type_62, convert_element_type_63, convert_element_type_64, convolution
# Graph fragment:
#   %arg25_1 : Tensor "f32[128, 4, 3, 3][36, 1, 12, 4]cuda:0" = PlaceHolder[target=arg25_1]
#   %convert_element_type_64 : Tensor "bf16[s13, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_63 : Tensor "bf16[128, 4, 3, 3][36, 1, 12, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_64, %convert_element_type_63, %convert_element_type_62, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused__to_copy_convolution_1 = async_compile.triton('triton_poi_fused__to_copy_convolution_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ms/cmsjsdojujsrwiixs22t3smiv3dulplf3dxxvf56ncwuxh5xcgz4.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h => convert_element_type_62, convert_element_type_63, convert_element_type_64, convolution
# Graph fragment:
#   %arg26_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %convert_element_type_64 : Tensor "bf16[s13, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_63 : Tensor "bf16[128, 4, 3, 3][36, 1, 12, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_64, %convert_element_type_63, %convert_element_type_62, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf2
triton_poi_fused__to_copy_convolution_2 = async_compile.triton('triton_poi_fused__to_copy_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/m3/cm3anany5y2afk5qtdauuqoydhbistqzyuvnlqxqt2ofpdxo4pt4.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h => convert_element_type_62, convert_element_type_63, convert_element_type_64, convolution
# Graph fragment:
#   %buf4 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf4]
#   %buf2 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf2]
#   %convert_element_type_64 : Tensor "bf16[s13, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_63 : Tensor "bf16[128, 4, 3, 3][36, 1, 12, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_64, %convert_element_type_63, %convert_element_type_62, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused__to_copy_convolution_3 = async_compile.triton('triton_poi_fused__to_copy_convolution_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ld/cldpnlefv4d24f2s4n4yp2lw34meuhlgnydgha5rogjusf5rtl2q.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x => convert_element_type_65, convert_element_type_66, convolution_1
# Graph fragment:
#   %arg28_1 : Tensor "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=arg28_1]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %convert_element_type_66, %convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf6
triton_poi_fused__to_copy_convolution_4 = async_compile.triton('triton_poi_fused__to_copy_convolution_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dl/cdl6ki4oro7woepvpp5eby7fdnunjradljdcxt3m3noxdjznmz4e.py
# Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   x => convert_element_type_65, convert_element_type_66, convolution_1
#   x_norm => clone, convert_element_type_67, var_mean, view
# Graph fragment:
#   %buf8 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf8]
#   %buf7 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf7]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %convert_element_type_66, %convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_67 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [%arg7_1, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf10
triton_red_fused__to_copy_clone_convolution_native_group_norm_5 = async_compile.triton('triton_red_fused__to_copy_clone_convolution_native_group_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_convolution_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_clone_convolution_native_group_norm_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2a/c2a55nnw2dgfx3two77cxvvr5kpcgfii7imtawgwxbohmibdndof.py
# Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   arange => iota
#   args => mul_3
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
#   %arg1_1 : Tensor "f32[s13][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %unsqueeze : Tensor "f32[s13, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, 1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, -9.210340371976184), kwargs = {})
#   %div : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 128), kwargs = {})
#   %exp : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 0), kwargs = {})
#   %mul_3 : Tensor "f32[s13, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %cos : Tensor "f32[s13, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_3,), kwargs = {})
#   %sin : Tensor "f32[s13, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_3,), kwargs = {})
#   %cat : Tensor "f32[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_6 = async_compile.triton('triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ow/cowx27eqxrb2nus5iaewhhzmc3xeosk42y2ud43y64st76de3lka.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_1
# Graph fragment:
#   %arg2_1 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type_1 : Tensor "bf16[1024, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/oj/cojenija6sykhbfk7goypzisajpklcn53jjegj63gigcdmhskelc.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_19, convert_element_type
#   input_2 => convert_element_type_6, convert_element_type_7, mul_16, sigmoid
# Graph fragment:
#   %mm_default_19 : Tensor "bf16[s13, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_default_19]
#   %arg3_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "bf16[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bfloat16), kwargs = {})
#   %add_tensor_19 : Tensor "bf16[s13, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_19, %convert_element_type), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[s13, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_19, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[s13, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul_16 : Tensor "f32[s13, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[s13, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
#   return %convert_element_type_7
triton_poi_fused__to_copy_addmm_silu_8 = async_compile.triton('triton_poi_fused__to_copy_addmm_silu_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/w7/cw7aet5k5hmwy6fpmctwgu6dk7mn7r5ncmig343cnnkgtml7rw6v.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_8
# Graph fragment:
#   %arg5_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_8 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_8
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7u/c7uigpg4dcseoimmrtyij4wnhuxvi72vk7rde6zvn6bv2q7fhplr.py
# Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   cat_1 => cat_1
#   input_4 => convert_element_type_15
#   s_emb => embedding
# Graph fragment:
#   %addmm_1 : Tensor "bf16[s13, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %arg8_1 : Tensor "i64[s13][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg6_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %embedding : Tensor "f32[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg6_1, %arg8_1), kwargs = {})
#   %cat_1 : Tensor "f32[s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%addmm_1, %embedding], -1), kwargs = {})
#   %convert_element_type_15 : Tensor "bf16[s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_15
triton_poi_fused__to_copy_cat_embedding_10 = async_compile.triton('triton_poi_fused__to_copy_cat_embedding_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_embedding_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_embedding_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full([XBLOCK], 4, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tl.broadcast_to(tmp16, [XBLOCK])) & (tl.broadcast_to(tmp16, [XBLOCK]) < 4)) | ~(tmp9 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp16, [XBLOCK]) < 4")
    tmp18 = tl.load(in_ptr2 + (256*tmp16 + ((-256) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp8, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/y5/cy57ibifd4mer32jyro6rqh2s3ntc4p2l5uloofdeyj254mtxjev.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_4 => add_tensor_18, convert_element_type_13
#   input_5 => convert_element_type_19, convert_element_type_20, mul_29, sigmoid_1
# Graph fragment:
#   %mm_default_18 : Tensor "bf16[s13, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_18]
#   %arg10_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convert_element_type_13 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.bfloat16), kwargs = {})
#   %add_tensor_18 : Tensor "bf16[s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %convert_element_type_13), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[s13, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_18, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_19,), kwargs = {})
#   %mul_29 : Tensor "f32[s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %sigmoid_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.bfloat16), kwargs = {})
#   return %convert_element_type_20
triton_poi_fused__to_copy_addmm_silu_11 = async_compile.triton('triton_poi_fused__to_copy_addmm_silu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_silu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_silu_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/x5/cx5gbc2ih7aynjrk5eycyl4jbvanovsoebq4mxjj6ke4u4co37m3.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_6 => convert_element_type_22
# Graph fragment:
#   %arg11_1 : Tensor "f32[256, 512][512, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convert_element_type_22 : Tensor "bf16[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_22
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/l7/cl725fju4ab64pvamfhmjwft2xhccahrqlwscvf6opr4mldrlee2.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_7 => convert_element_type_27
# Graph fragment:
#   %arg13_1 : Tensor "f32[256, 256][256, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convert_element_type_27 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_27
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/me/cme5bvblthwopbgxbar4cefesxgy2x7iu42ctblqlew3hoqnm4dj.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
# Source node to ATen node mapping:
#   input_7 => add_tensor_17, convert_element_type_26
#   input_8 => convert_element_type_31, convert_element_type_32, mul_36, sigmoid_2
# Graph fragment:
#   %mm_default_17 : Tensor "bf16[s13, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_17]
#   %arg14_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %convert_element_type_26 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   %add_tensor_17 : Tensor "bf16[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_17, %convert_element_type_26), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[s13, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_17, torch.float32), kwargs = {})
#   %sigmoid_2 : Tensor "f32[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_31,), kwargs = {})
#   %mul_36 : Tensor "f32[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, %sigmoid_2), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_36, torch.bfloat16), kwargs = {})
#   return %convert_element_type_32
triton_poi_fused__to_copy_addmm_silu_14 = async_compile.triton('triton_poi_fused__to_copy_addmm_silu_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_silu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_silu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hx/chx52q3fjfcmxa5njnjztlzfuynodtwj46fds27r3nrg5nt6uax5.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_34
# Graph fragment:
#   %arg15_1 : Tensor "f32[3200, 256][256, 1]cuda:0" = PlaceHolder[target=arg15_1]
#   %convert_element_type_34 : Tensor "bf16[3200, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg15_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_34
triton_poi_fused__to_copy_15 = async_compile.triton('triton_poi_fused__to_copy_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6553600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/uz/cuzlcbacxwfku5tvqjqg2fojmqw5mi5g3lr2x43k5ehamxkrbxg3.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_33
# Graph fragment:
#   %arg16_1 : Tensor "f32[3200][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convert_element_type_33 : Tensor "bf16[3200][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg16_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_33
triton_poi_fused__to_copy_16 = async_compile.triton('triton_poi_fused__to_copy_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pb/cpbk7gstvneubols4zwjzhdhkugykizr2vmt4kqzg3ajnf5ke3rj.py
# Topologically Sorted Source Nodes: [x, x_norm, x_flat, v_t_x], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
# Source node to ATen node mapping:
#   v_t_x => convert_element_type_68
#   x => convert_element_type_65, convert_element_type_66, convolution_1
#   x_flat => view_5
#   x_norm => add_99, clone, convert_element_type_67, mul_75, rsqrt, sub_30, var_mean, view, view_1
# Graph fragment:
#   %buf8 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf8]
#   %buf7 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf7]
#   %getitem_1 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf10 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=buf10]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %convert_element_type_66, %convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_67 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [%arg7_1, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_30 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add_99 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_99,), kwargs = {})
#   %mul_75 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_75, [%arg7_1, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[s13, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [%arg0_1, 128, -1]), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[s13, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.bfloat16), kwargs = {})
#   return %convert_element_type_68
triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17 = async_compile.triton('triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 1024)
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 4096*(((x1 % 32)) // 32) + 131072*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 4096.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/c5/cc5tk7x5zrmdfvf44ydjkwt6ra3t7hz37x3s37pdciqt43lqubwi.py
# Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
# Source node to ATen node mapping:
#   out => add_153
#   out_1 => add_163
#   shift_1 => view_4
#   split => split_with_sizes
#   view_4 => view_6
#   x => convert_element_type_65, convert_element_type_66, convolution_1
#   x_1 => mul_110, sigmoid_5
#   x_2 => convert_element_type_73, convert_element_type_74, convert_element_type_75, convolution_2
#   x_flat => view_5
#   x_norm => add_99, clone, convert_element_type_67, mul_75, rsqrt, sub_30, var_mean, view, view_1
# Graph fragment:
#   %buf8 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf8]
#   %buf7 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf7]
#   %getitem_1 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf10 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=buf10]
#   %bmm_1 : Tensor "bf16[s13, 128, 1024][131072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %addmm_5 : Tensor "bf16[s13, 3200][3200, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %add_163 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=add_163]
#   %buf36 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=buf36]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %convert_element_type_66, %convert_element_type_65, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_67 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [%arg7_1, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %sub_30 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add_99 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_99,), kwargs = {})
#   %mul_75 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_75, [%arg7_1, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[s13, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [%arg0_1, 128, -1]), kwargs = {})
#   %add_153 : Tensor "f32[s13, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_153, [%arg0_1, 128, 32, 32]), kwargs = {})
#   %view_4 : Tensor "bf16[s13, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [%arg0_1, 128, 1, 1]), kwargs = {})
#   %add_163 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_110 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_5), kwargs = {})
#   %convert_element_type_75 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_110, torch.bfloat16), kwargs = {})
#   %convert_element_type_74 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg30_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg31_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_75, %convert_element_type_74, %convert_element_type_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_163,%buf36,%buf39
triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 1024
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 131072*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y3 // 4), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y3 // 4), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (3072 + y0 + 3200*y1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 4096.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (y0 + 128*x2 + 131072*y1), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/fq/cfqouqhavvgh54ys34yqxrz4evd4oky5ebwbj5z4ncqjnmr6ol43.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x_1 => mul_110, sigmoid_5
#   x_2 => convert_element_type_73, convert_element_type_74, convert_element_type_75, convolution_2
# Graph fragment:
#   %arg30_1 : Tensor "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=arg30_1]
#   %sigmoid_5 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_110 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_5), kwargs = {})
#   %convert_element_type_75 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_110, torch.bfloat16), kwargs = {})
#   %convert_element_type_74 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg30_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg31_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_75, %convert_element_type_74, %convert_element_type_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf37
triton_poi_fused__to_copy_convolution_silu_19 = async_compile.triton('triton_poi_fused__to_copy_convolution_silu_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_silu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_silu_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7c/c7cuzs7rideeebzkgalxa7yu7j7z2lh35pbq3bypw2t67fmssexk.py
# Topologically Sorted Source Nodes: [x_1, x_2, add_2, h_1], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_184
#   h_1 => convert_element_type_76, convert_element_type_77, mul_119, sigmoid_6
#   x_1 => mul_110, sigmoid_5
#   x_2 => convert_element_type_73, convert_element_type_74, convert_element_type_75, convolution_2
# Graph fragment:
#   %buf40 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf40]
#   %buf38 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf38]
#   %convolution : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution]
#   %sigmoid_5 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_110 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_5), kwargs = {})
#   %convert_element_type_75 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_110, torch.bfloat16), kwargs = {})
#   %convert_element_type_74 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg30_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg31_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_75, %convert_element_type_74, %convert_element_type_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_184 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_184, torch.float32), kwargs = {})
#   %sigmoid_6 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_76,), kwargs = {})
#   %mul_119 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_76, %sigmoid_6), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_119, torch.bfloat16), kwargs = {})
#   return %convert_element_type_77
triton_poi_fused__to_copy_add_convolution_silu_20 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_20(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ph/cphqsxpo4fltvb7taolmk7xd4tb2fks7kcvh2dw3lbwc36treamr.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_3 => convert_element_type_91, convert_element_type_92, convolution_5
# Graph fragment:
#   %arg36_1 : Tensor "f32[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=arg36_1]
#   %convert_element_type_92 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg36_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_91 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg37_1, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %convert_element_type_92, %convert_element_type_91, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf60
triton_poi_fused__to_copy_convolution_21 = async_compile.triton('triton_poi_fused__to_copy_convolution_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/tn/ctnakyurmmledx3qcfg6nrfzmnkmq2iuoxlvjpwuxkimcatepi4o.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_3 => convert_element_type_91, convert_element_type_92, convolution_5
# Graph fragment:
#   %buf62 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf62]
#   %buf61 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf61]
#   %convert_element_type_92 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg36_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_91 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg37_1, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %convert_element_type_92, %convert_element_type_91, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_5
triton_poi_fused__to_copy_convolution_22 = async_compile.triton('triton_poi_fused__to_copy_convolution_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rz/crztuq4ygk4nyreegzi4rfv2dj645fcpeervtyrz3xbdyg3q2f6i.py
# Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_2, expand
#   q_1 => view_16
#   q_2 => permute_15
#   view_11 => view_17
# Graph fragment:
#   %mm : Tensor "bf16[256*s13, 256][256, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_16 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [%arg7_1, 256, 256]), kwargs = {})
#   %view_17 : Tensor "bf16[s13, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [%arg0_1, 256, 4, 64]), kwargs = {})
#   %permute_15 : Tensor "bf16[s13, 4, 256, 64][65536, 64, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 3]), kwargs = {})
#   %expand : Tensor "bf16[s13, 4, 256, 64][65536, 64, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_15, [%arg7_1, 4, 256, 64]), kwargs = {})
#   %clone_2 : Tensor "bf16[s13, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused__unsafe_view_clone_expand_transpose_view_23 = async_compile.triton('triton_poi_fused__unsafe_view_clone_expand_transpose_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_expand_transpose_view_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_expand_transpose_view_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/im/cimfaprsp7eqzr5o22oulujowaz7ytv2lfsuzbnfd6344bzdk6if.py
# Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   k_1 => permute_16
#   kv => unsqueeze_2
#   matmul => clone_3, expand_1
#   transpose_6 => permute_18
#   view_12 => view_18
# Graph fragment:
#   %mm_1 : Tensor "bf16[s13, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "bf16[s13, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_18 : Tensor "bf16[s13, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_10, [%arg0_1, 1, 4, 64]), kwargs = {})
#   %permute_16 : Tensor "bf16[s13, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %permute_18 : Tensor "bf16[s13, 4, 64, 1][512, 64, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_16, [0, 1, 3, 2]), kwargs = {})
#   %expand_1 : Tensor "bf16[s13, 4, 64, 1][512, 64, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_18, [%arg7_1, 4, 64, 1]), kwargs = {})
#   %clone_3 : Tensor "bf16[s13, 4, 64, 1][256, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24 = async_compile.triton('triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/gd/cgdwm277zujfl53uggs33uaayy2qjjyrtxfifq6drz4yyo6uofw4.py
# Topologically Sorted Source Nodes: [matmul, attn_1, matmul_1], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div_1, exp_1, sum_1
#   matmul => view_22
#   matmul_1 => convert_element_type_102
# Graph fragment:
#   %bmm_4 : Tensor "bf16[4*s13, 256, 1][256, 1, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %view_22 : Tensor "bf16[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_4, [%arg7_1, 4, 256, 1]), kwargs = {})
#   %convert_element_type_default_10 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_22, torch.float32), kwargs = {})
#   %mul_tensor_18 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_10, 1), kwargs = {})
#   %amax_default_9 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_18, [-1], True), kwargs = {})
#   %sub_tensor_9 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_18, %amax_default_9), kwargs = {})
#   %mul_tensor_19 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_9, 0.125), kwargs = {})
#   %exp_1 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_19,), kwargs = {})
#   %sum_1 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_1), kwargs = {})
#   %convert_element_type_102 : Tensor "bf16[s13, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_102
triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25 = async_compile.triton('triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 - tmp3
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = (tmp7 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ns/cnscpxwomvjqz4yaegwqxnxq5ye4a6dhaaerez52b5jcqix5hnuu.py
# Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   kv => unsqueeze_2
#   matmul_1 => clone_4, expand_3
#   v_5 => permute_17
#   view_13 => view_19
# Graph fragment:
#   %mm_1 : Tensor "bf16[s13, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "bf16[s13, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_19 : Tensor "bf16[s13, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_11, [%arg0_1, 1, 4, 64]), kwargs = {})
#   %permute_17 : Tensor "bf16[s13, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %expand_3 : Tensor "bf16[s13, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_17, [%arg7_1, 4, 1, 64]), kwargs = {})
#   %clone_4 : Tensor "bf16[s13, 4, 1, 64][256, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26 = async_compile.triton('triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/56/c56tkofkrgehbgt6m5g47dozupczmntu3acu227tsetu7rohrlkj.py
# Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => view_25
#   out_4 => clone_5
#   transpose_7 => permute_19
# Graph fragment:
#   %bmm_5 : Tensor "bf16[4*s13, 256, 64][16384, 64, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %view_25 : Tensor "bf16[s13, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [%arg7_1, 4, 256, 64]), kwargs = {})
#   %permute_19 : Tensor "bf16[s13, 256, 4, 64][65536, 64, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_25, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "bf16[s13, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_clone_transpose_view_27 = async_compile.triton('triton_poi_fused_clone_transpose_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 4)
    x2 = ((xindex // 256) % 256)
    x3 = xindex // 65536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 16384*x1 + 65536*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/44/c44j75xbsjmmgkjwwzavwm4j2r6q3sj4durk7ugydhhhedlrhqte.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
# Source node to ATen node mapping:
#   input_16 => add_tensor_16, convert_element_type_105, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_505
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
# Graph fragment:
#   %convolution_5 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %mm_default_16 : Tensor "bf16[256*s13, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_16]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %add_tensor_16 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [%arg7_1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[s13, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [%arg0_1, 256, 16, 16]), kwargs = {})
#   %add_505 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg42_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_505, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf75
triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/6k/c6kg3hnxadu7ol2jxyxwf5zrw4pehfwxp7ifvzxcz25wwnkuh33d.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
# Source node to ATen node mapping:
#   input_16 => add_tensor_16, convert_element_type_105, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_505
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
# Graph fragment:
#   %arg42_1 : Tensor "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=arg42_1]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %add_tensor_16 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [%arg7_1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[s13, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [%arg0_1, 256, 16, 16]), kwargs = {})
#   %add_505 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg42_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_505, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf76
triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qz/cqzc7llb5h3lf7zkjacouz6oa27c3kbipya6vvx5e6jyxtnjomhq.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_16 => add_tensor_16, convert_element_type_105, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_505
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
#   x_norm_2 => clone_7, convert_element_type_112, var_mean_2, view_30
# Graph fragment:
#   %buf78 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf78]
#   %buf77 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf77]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %add_tensor_16 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [%arg7_1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[s13, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [%arg0_1, 256, 16, 16]), kwargs = {})
#   %add_505 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg42_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_505, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_112 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_7 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[s13, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [%arg7_1, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_13,%buf80
triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30 = async_compile.triton('triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 256*r0_3 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 8*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pr/cprawoglhrvrfa3gl57qlghkvohl2qqrb7jq6tb5la63au55l2uk.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_46
# Graph fragment:
#   %arg19_1 : Tensor "f32[6400, 256][256, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %convert_element_type_46 : Tensor "bf16[6400, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_46
triton_poi_fused__to_copy_31 = async_compile.triton('triton_poi_fused__to_copy_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13107200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/g4/cg4uzze7bdmge7rwxgpzqcjmjm2omrh4eiyls47vvl6whugsp7oe.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_45
# Graph fragment:
#   %arg20_1 : Tensor "f32[6400][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %convert_element_type_45 : Tensor "bf16[6400][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_45
triton_poi_fused__to_copy_32 = async_compile.triton('triton_poi_fused__to_copy_32', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/v7/cv7qzu3gb7bciuxs2yvkq3wun6w3oohx5f2l6lkrxyvq7euxmdc2.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, x_flat_2, v_t_x_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_16 => add_tensor_16, convert_element_type_105, view_28
#   out_5 => view_29
#   transpose_8 => permute_21
#   v_t_x_2 => convert_element_type_113
#   x_6 => add_505
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
#   x_flat_2 => view_35
#   x_norm_2 => add_526, clone_7, convert_element_type_112, mul_384, rsqrt_2, sub_129, var_mean_2, view_30, view_31
# Graph fragment:
#   %buf78 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf78]
#   %buf77 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf77]
#   %getitem_13 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf80 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=buf80]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %add_tensor_16 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [%arg7_1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[s13, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [%arg0_1, 256, 16, 16]), kwargs = {})
#   %add_505 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg42_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_505, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_112 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_7 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[s13, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [%arg7_1, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_129 : Tensor "f32[s13, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_526 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_526,), kwargs = {})
#   %mul_384 : Tensor "f32[s13, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_384, [%arg7_1, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [%arg0_1, 256, -1]), kwargs = {})
#   %convert_element_type_113 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.bfloat16), kwargs = {})
#   return %convert_element_type_113
triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 256)
    x2 = xindex // 65536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 4096*(((x1 % 16)) // 16) + 65536*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 2048.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ly/clyqucjke5rhyglkn5mfeups45jox2vpkmlz5vg2jkemgjrito4v.py
# Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   input_16 => add_tensor_16, convert_element_type_105, view_28
#   out_5 => view_29
#   out_6 => add_580
#   out_7 => add_590
#   shift_5 => view_34
#   split_2 => split_with_sizes_2
#   transpose_8 => permute_21
#   view_19 => view_36
#   x_6 => add_505
#   x_7 => convert_element_type_110, convert_element_type_111, convolution_6
#   x_8 => mul_419, sigmoid_9
#   x_9 => convert_element_type_118, convert_element_type_119, convert_element_type_120, convolution_7
#   x_flat_2 => view_35
#   x_norm_2 => add_526, clone_7, convert_element_type_112, mul_384, rsqrt_2, sub_129, var_mean_2, view_30, view_31
# Graph fragment:
#   %buf78 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf78]
#   %buf77 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf77]
#   %getitem_13 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf80 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=buf80]
#   %bmm_7 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %addmm_7 : Tensor "bf16[s13, 6400][6400, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %add_590 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_590]
#   %buf94 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=buf94]
#   %convert_element_type_105 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %add_tensor_16 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %convert_element_type_105), kwargs = {})
#   %view_28 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_16, [%arg7_1, 256, 256]), kwargs = {})
#   %permute_21 : Tensor "bf16[s13, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [%arg0_1, 256, 16, 16]), kwargs = {})
#   %add_505 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg42_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_110 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_505, %convert_element_type_111, %convert_element_type_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_112 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_7 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[s13, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [%arg7_1, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %sub_129 : Tensor "f32[s13, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_13), kwargs = {})
#   %add_526 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_526,), kwargs = {})
#   %mul_384 : Tensor "f32[s13, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_384, [%arg7_1, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [%arg0_1, 256, -1]), kwargs = {})
#   %add_580 : Tensor "f32[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_580, [%arg0_1, 256, 16, 16]), kwargs = {})
#   %view_34 : Tensor "bf16[s13, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_16, [%arg0_1, 256, 1, 1]), kwargs = {})
#   %add_590 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_590,), kwargs = {})
#   %mul_419 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_590, %sigmoid_9), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_419, torch.bfloat16), kwargs = {})
#   %convert_element_type_119 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg44_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_118 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg45_1, torch.bfloat16), kwargs = {})
#   %convolution_7 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %convert_element_type_119, %convert_element_type_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_590,%buf94,%buf97
triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 65536*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y3 // 8), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y3 // 8), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (6144 + y0 + 6400*y1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 2048.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (y0 + 256*x2 + 65536*y1), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/e3/ce36h7kf6mmosnpv65wej7sapt7aeohxkzftlwjv5acvkcocwiak.py
# Topologically Sorted Source Nodes: [x_8, x_9, add_9, h_4], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_9 => add_611
#   h_4 => convert_element_type_121, convert_element_type_122, mul_428, sigmoid_10
#   x_8 => mul_419, sigmoid_9
#   x_9 => convert_element_type_118, convert_element_type_119, convert_element_type_120, convolution_7
# Graph fragment:
#   %buf98 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf98]
#   %buf96 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf96]
#   %convolution_5 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %sigmoid_9 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_590,), kwargs = {})
#   %mul_419 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_590, %sigmoid_9), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_419, torch.bfloat16), kwargs = {})
#   %convert_element_type_119 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg44_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_118 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg45_1, torch.bfloat16), kwargs = {})
#   %convolution_7 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %convert_element_type_119, %convert_element_type_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_611 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %convert_element_type_121 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_611, torch.float32), kwargs = {})
#   %sigmoid_10 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_121,), kwargs = {})
#   %mul_428 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_121, %sigmoid_10), kwargs = {})
#   %convert_element_type_122 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_428, torch.bfloat16), kwargs = {})
#   return %convert_element_type_122
triton_poi_fused__to_copy_add_convolution_silu_35 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_35(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/j3/cj3dc4hnnfxk2ghrcdf6c633ha4icab7dq4pvzqp273fotzgwmox.py
# Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_6 => convert_element_type_153, convert_element_type_154, convolution_10
# Graph fragment:
#   %arg54_1 : Tensor "f32[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=arg54_1]
#   %convert_element_type_154 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg54_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_153 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg55_1, torch.bfloat16), kwargs = {})
#   %convolution_10 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_152, %convert_element_type_154, %convert_element_type_153, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf130
triton_poi_fused__to_copy_convolution_36 = async_compile.triton('triton_poi_fused__to_copy_convolution_36', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/po/cpoi6yzazzl6kx2fg2entmcsbkpqshtpahz7wbsvvbibvrks2ejm.py
# Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_6 => convert_element_type_153, convert_element_type_154, convolution_10
# Graph fragment:
#   %arg55_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg55_1]
#   %convert_element_type_154 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg54_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_153 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg55_1, torch.bfloat16), kwargs = {})
#   %convolution_10 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_152, %convert_element_type_154, %convert_element_type_153, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf131
triton_poi_fused__to_copy_convolution_37 = async_compile.triton('triton_poi_fused__to_copy_convolution_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/cb/ccbcj76wxun7d6etsz4edq3ujczn7qitl2dxegqhdl7yotrnjflx.py
# Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_6 => convert_element_type_153, convert_element_type_154, convolution_10
# Graph fragment:
#   %buf132 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf132]
#   %buf131 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf131]
#   %convert_element_type_154 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg54_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_153 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg55_1, torch.bfloat16), kwargs = {})
#   %convolution_10 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_152, %convert_element_type_154, %convert_element_type_153, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_10
triton_poi_fused__to_copy_convolution_38 = async_compile.triton('triton_poi_fused__to_copy_convolution_38', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_38(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dz/cdzt6qjo42rboeizu637xogjogfm4ee5wtimn4xsxvtnixkkfyd5.py
# Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   matmul_4 => clone_14, expand_8
#   q_7 => view_62
#   q_8 => permute_37
#   view_31 => view_63
# Graph fragment:
#   %mm_4 : Tensor "bf16[64*s13, 512][512, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %view_62 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [%arg7_1, 64, 512]), kwargs = {})
#   %view_63 : Tensor "bf16[s13, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_62, [%arg0_1, 64, 8, 64]), kwargs = {})
#   %permute_37 : Tensor "bf16[s13, 8, 64, 64][32768, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_63, [0, 2, 1, 3]), kwargs = {})
#   %expand_8 : Tensor "bf16[s13, 8, 64, 64][32768, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_37, [%arg7_1, 8, 64, 64]), kwargs = {})
#   %clone_14 : Tensor "bf16[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_14
triton_poi_fused__unsafe_view_clone_expand_transpose_view_39 = async_compile.triton('triton_poi_fused__unsafe_view_clone_expand_transpose_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_expand_transpose_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_expand_transpose_view_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/js/cjs5smioacfapi2rvuzzv2x4jjervclatkgl2gmfpoeshyijp32h.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   k_5 => permute_38
#   kv_2 => unsqueeze_4
#   matmul_4 => clone_15, expand_9
#   transpose_22 => permute_40
#   view_32 => view_64
# Graph fragment:
#   %mm_5 : Tensor "bf16[s13, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "bf16[s13, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_64 : Tensor "bf16[s13, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_24, [%arg0_1, 1, 8, 64]), kwargs = {})
#   %permute_38 : Tensor "bf16[s13, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_64, [0, 2, 1, 3]), kwargs = {})
#   %permute_40 : Tensor "bf16[s13, 8, 64, 1][1024, 64, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_38, [0, 1, 3, 2]), kwargs = {})
#   %expand_9 : Tensor "bf16[s13, 8, 64, 1][1024, 64, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_40, [%arg7_1, 8, 64, 1]), kwargs = {})
#   %clone_15 : Tensor "bf16[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_15
triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40 = async_compile.triton('triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/q7/cq7hpm46cjjj67chkevbh73llyujuh54vtakep3tzhxljlnc2bez.py
# Topologically Sorted Source Nodes: [matmul_4, attn_5, matmul_5], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_5 => div_3, exp_3, sum_3
#   matmul_4 => view_68
#   matmul_5 => convert_element_type_164
# Graph fragment:
#   %bmm_12 : Tensor "bf16[8*s13, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_12]
#   %view_68 : Tensor "bf16[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_12, [%arg7_1, 8, 64, 1]), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_68, torch.float32), kwargs = {})
#   %mul_tensor_14 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, 1), kwargs = {})
#   %amax_default_7 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_14, [-1], True), kwargs = {})
#   %sub_tensor_7 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_14, %amax_default_7), kwargs = {})
#   %mul_tensor_15 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_7, 0.125), kwargs = {})
#   %exp_3 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_15,), kwargs = {})
#   %sum_3 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [-1], True), kwargs = {})
#   %div_3 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_3, %sum_3), kwargs = {})
#   %convert_element_type_164 : Tensor "bf16[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_164
triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41 = async_compile.triton('triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 - tmp3
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = (tmp7 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vr/cvr442fuz6p6cji4psyuk2mquy2jpvth4vhemqqzuiygsbuyvp5b.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   kv_2 => unsqueeze_4
#   matmul_5 => clone_16, expand_11
#   v_13 => permute_39
#   view_33 => view_65
# Graph fragment:
#   %mm_5 : Tensor "bf16[s13, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "bf16[s13, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_65 : Tensor "bf16[s13, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_25, [%arg0_1, 1, 8, 64]), kwargs = {})
#   %permute_39 : Tensor "bf16[s13, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_65, [0, 2, 1, 3]), kwargs = {})
#   %expand_11 : Tensor "bf16[s13, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_39, [%arg7_1, 8, 1, 64]), kwargs = {})
#   %clone_16 : Tensor "bf16[s13, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_16
triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42 = async_compile.triton('triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + 1024*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/q6/cq6dtpwdxbi74dprfqdeafh752b27kxg52o3pukovzsr7w6z77ff.py
# Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_5 => view_71
#   out_12 => clone_17
#   transpose_23 => permute_41
# Graph fragment:
#   %bmm_13 : Tensor "bf16[8*s13, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %view_71 : Tensor "bf16[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_13, [%arg7_1, 8, 64, 64]), kwargs = {})
#   %permute_41 : Tensor "bf16[s13, 64, 8, 64][32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_71, [0, 2, 1, 3]), kwargs = {})
#   %clone_17 : Tensor "bf16[s13, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_17
triton_poi_fused_clone_transpose_view_43 = async_compile.triton('triton_poi_fused_clone_transpose_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 8)
    x2 = ((xindex // 512) % 64)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 4096*x1 + 32768*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/va/cvaeclkoprywksylpswjvzl2hiufaxvmwnmofisn3u3gwhls5icj.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
# Source node to ATen node mapping:
#   input_20 => add_tensor_13, convert_element_type_167, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_1132
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
# Graph fragment:
#   %convolution_10 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %mm_default_13 : Tensor "bf16[64*s13, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_13]
#   %arg59_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %add_tensor_13 : Tensor "bf16[64*s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [%arg7_1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %add_1132 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg60_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg61_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1132, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf145
triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dy/cdyfym52bedy5gxemqjblhgzkdxv7okqj2hpqk6ff2j662l6qsjw.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
# Source node to ATen node mapping:
#   input_20 => add_tensor_13, convert_element_type_167, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_1132
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
# Graph fragment:
#   %arg60_1 : Tensor "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=arg60_1]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %add_tensor_13 : Tensor "bf16[64*s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [%arg7_1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %add_1132 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg60_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg61_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1132, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf146
triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_45 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_45', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hc/chcuvs6qya4f372ykr4mcxd6t3j5eyqvzjzqco7neka6wrh6n75h.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_20 => add_tensor_13, convert_element_type_167, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_1132
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
#   x_norm_4 => clone_19, convert_element_type_174, var_mean_4, view_76
# Graph fragment:
#   %buf148 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf148]
#   %buf147 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf147]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %add_tensor_13 : Tensor "bf16[64*s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [%arg7_1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %add_1132 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg60_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg61_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1132, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_174 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_19 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[s13, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_19, [%arg7_1, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_27,%buf150
triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46 = async_compile.triton('triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 32768*x1), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([XBLOCK, 1], 1024, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp13, xmask)
    tl.store(out_ptr1 + (x4), tmp19, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/5i/c5iqqkbrlttxnxgmxlmvdlywcbbajgba3ep7qsdozb6yjspiyfu7.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_58
# Graph fragment:
#   %arg23_1 : Tensor "f32[12800, 256][256, 1]cuda:0" = PlaceHolder[target=arg23_1]
#   %convert_element_type_58 : Tensor "bf16[12800, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg23_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_58
triton_poi_fused__to_copy_47 = async_compile.triton('triton_poi_fused__to_copy_47', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26214400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3276800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/4f/c4f65vl3d5ugqr2wdzb7mio2ztypmwxinm5djr74hmet6iffpsip.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_57
# Graph fragment:
#   %arg24_1 : Tensor "f32[12800][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %convert_element_type_57 : Tensor "bf16[12800][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg24_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_57
triton_poi_fused__to_copy_48 = async_compile.triton('triton_poi_fused__to_copy_48', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2u/c2urxmuatqe7niimzr3k2qgre66irlghlamazv53wplpjtrdo3vp.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, x_flat_4, v_t_x_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_20 => add_tensor_13, convert_element_type_167, view_74
#   out_13 => view_75
#   transpose_24 => permute_43
#   v_t_x_4 => convert_element_type_175
#   x_14 => add_1132
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
#   x_flat_4 => view_81
#   x_norm_4 => add_1153, clone_19, convert_element_type_174, mul_874, rsqrt_4, sub_274, var_mean_4, view_76, view_77
# Graph fragment:
#   %buf148 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf148]
#   %buf147 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf147]
#   %getitem_27 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf150 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=buf150]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %add_tensor_13 : Tensor "bf16[64*s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [%arg7_1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %add_1132 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg60_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg61_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1132, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_174 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_19 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[s13, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_19, [%arg7_1, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_274 : Tensor "f32[s13, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_1153 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1153,), kwargs = {})
#   %mul_874 : Tensor "f32[s13, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_274, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_874, [%arg7_1, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[s13, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [%arg0_1, 512, -1]), kwargs = {})
#   %convert_element_type_175 : Tensor "bf16[s13, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_81, torch.bfloat16), kwargs = {})
#   return %convert_element_type_175
triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 64)
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 4096*(((x1 % 8)) // 8) + 32768*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 1024.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7v/c7voxzabd3duu4uxv2u2usxqj3del43p7mnjkhv7no3sb2lrtlww.py
# Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
# Source node to ATen node mapping:
#   input_20 => add_tensor_13, convert_element_type_167, view_74
#   out_13 => view_75
#   out_14 => add_1207
#   out_15 => add_1217
#   shift_9 => view_80
#   split_4 => split_with_sizes_4
#   transpose_24 => permute_43
#   view_39 => view_82
#   x_14 => add_1132
#   x_15 => convert_element_type_172, convert_element_type_173, convolution_11
#   x_16 => mul_909, sigmoid_13
#   x_17 => convert_element_type_180, convert_element_type_181, convert_element_type_182, convolution_12
#   x_flat_4 => view_81
#   x_norm_4 => add_1153, clone_19, convert_element_type_174, mul_874, rsqrt_4, sub_274, var_mean_4, view_76, view_77
# Graph fragment:
#   %buf148 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf148]
#   %buf147 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf147]
#   %getitem_27 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf150 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=buf150]
#   %bmm_15 : Tensor "bf16[s13, 512, 64][32768, 64, 1]cuda:0" = PlaceHolder[target=bmm_15]
#   %addmm_9 : Tensor "bf16[s13, 12800][12800, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %add_1217 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=add_1217]
#   %buf164 : Tensor "bf16[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=buf164]
#   %convert_element_type_167 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg59_1, torch.bfloat16), kwargs = {})
#   %add_tensor_13 : Tensor "bf16[64*s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %convert_element_type_167), kwargs = {})
#   %view_74 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [%arg7_1, 64, 512]), kwargs = {})
#   %permute_43 : Tensor "bf16[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %add_1132 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg60_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg61_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_1132, %convert_element_type_173, %convert_element_type_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_174 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_19 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[s13, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_19, [%arg7_1, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %sub_274 : Tensor "f32[s13, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_27), kwargs = {})
#   %add_1153 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1153,), kwargs = {})
#   %mul_874 : Tensor "f32[s13, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_274, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_874, [%arg7_1, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[s13, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [%arg0_1, 512, -1]), kwargs = {})
#   %add_1207 : Tensor "f32[s13, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1207, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %view_80 : Tensor "bf16[s13, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_30, [%arg0_1, 512, 1, 1]), kwargs = {})
#   %add_1217 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1217,), kwargs = {})
#   %mul_909 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1217, %sigmoid_13), kwargs = {})
#   %convert_element_type_182 : Tensor "bf16[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_909, torch.bfloat16), kwargs = {})
#   %convert_element_type_181 : Tensor "bf16[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg62_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_180 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg63_1, torch.bfloat16), kwargs = {})
#   %convolution_12 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_182, %convert_element_type_181, %convert_element_type_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_1217,%buf164,%buf167
triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 32768*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y3 // 16), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y3 // 16), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (12288 + y0 + 12800*y1), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 1024.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (y0 + 512*x2 + 32768*y1), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pp/cpppvgt2yblbospybrei5e2ygbml23r3jghcwnk6ngdfy5lldcn3.py
# Topologically Sorted Source Nodes: [x_16, x_17, add_17, h_7], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_17 => add_1238
#   h_7 => convert_element_type_183, convert_element_type_184, mul_918, sigmoid_14
#   x_16 => mul_909, sigmoid_13
#   x_17 => convert_element_type_180, convert_element_type_181, convert_element_type_182, convolution_12
# Graph fragment:
#   %buf168 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf168]
#   %buf166 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf166]
#   %convolution_10 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %sigmoid_13 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1217,), kwargs = {})
#   %mul_909 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1217, %sigmoid_13), kwargs = {})
#   %convert_element_type_182 : Tensor "bf16[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_909, torch.bfloat16), kwargs = {})
#   %convert_element_type_181 : Tensor "bf16[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg62_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_180 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg63_1, torch.bfloat16), kwargs = {})
#   %convolution_12 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_182, %convert_element_type_181, %convert_element_type_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_1238 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %convert_element_type_183 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1238, torch.float32), kwargs = {})
#   %sigmoid_14 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_183,), kwargs = {})
#   %mul_918 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_183, %sigmoid_14), kwargs = {})
#   %convert_element_type_184 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_918, torch.bfloat16), kwargs = {})
#   return %convert_element_type_184
triton_poi_fused__to_copy_add_convolution_silu_51 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_51(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/q7/cq7rptvdb5rs2jy5cl3dh3hko62umoiad5gaeymbihui5ugihfgt.py
# Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_21 => add_1549
#   h_8 => convert_element_type_213, convert_element_type_214, mul_1162, sigmoid_16
#   qkv => convert_element_type_218
#   view_50 => view_106
#   x_20 => mul_1153, sigmoid_15
#   x_21 => convert_element_type_210, convert_element_type_211, convert_element_type_212, convolution_14
#   x_flat_6 => permute_56
#   x_norm_6 => add_1572, add_1573, convert_element_type_215, mul_1173, mul_1174, rsqrt_6, sub_372, var_mean_6
# Graph fragment:
#   %buf196 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf196]
#   %buf194 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf194]
#   %convert_element_type_184 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convert_element_type_184]
#   %getitem_39 : Tensor "f32[s13, 64, 1][64, 1, 64*s13]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf198 : Tensor "f32[s13, 64, 1][64, 1, 64*s13]cuda:0" = PlaceHolder[target=buf198]
#   %arg72_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg72_1]
#   %arg73_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %sigmoid_15 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1528,), kwargs = {})
#   %mul_1153 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1528, %sigmoid_15), kwargs = {})
#   %convert_element_type_212 : Tensor "bf16[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1153, torch.bfloat16), kwargs = {})
#   %convert_element_type_211 : Tensor "bf16[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg70_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_210 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg71_1, torch.bfloat16), kwargs = {})
#   %convolution_14 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_212, %convert_element_type_211, %convert_element_type_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_1549 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convert_element_type_184), kwargs = {})
#   %convert_element_type_213 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1549, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_1162 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1162, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [%arg0_1, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_215 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_56, torch.float32), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_215, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_372 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_215, %getitem_39), kwargs = {})
#   %add_1572 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1572,), kwargs = {})
#   %mul_1173 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_372, %rsqrt_6), kwargs = {})
#   %mul_1174 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1173, %arg72_1), kwargs = {})
#   %add_1573 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1174, %arg73_1), kwargs = {})
#   %convert_element_type_218 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1573, torch.bfloat16), kwargs = {})
#   return %getitem_39,%buf198,%convert_element_type_218
triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_52 = async_compile.triton('triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tmp9 - tmp19
    tmp27 = 512.0
    tmp28 = (tmp25 / tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 512*x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lk/clktrarfycaiosew2e3xdrkyplzoy7yxnmpw7fzu5wuo6p3kd5ww.py
# Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   qkv => convert_element_type_217
# Graph fragment:
#   %arg74_1 : Tensor "f32[1536, 512][512, 1]cuda:0" = PlaceHolder[target=arg74_1]
#   %convert_element_type_217 : Tensor "bf16[1536, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg74_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_217
triton_poi_fused__to_copy_53 = async_compile.triton('triton_poi_fused__to_copy_53', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/zt/cztwre7lczkhm27rzduq75xd7xzd3u2i5w7g4xs67fww36sxdx3d.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   matmul_8 => clone_26, expand_16
#   q_12 => select
#   qkv => add_tensor_10, convert_element_type_216, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
# Graph fragment:
#   %mm_default_10 : Tensor "bf16[64*s13, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_10]
#   %arg75_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %convert_element_type_216 : Tensor "bf16[1536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg75_1, torch.bfloat16), kwargs = {})
#   %add_tensor_10 : Tensor "bf16[64*s13, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %convert_element_type_216), kwargs = {})
#   %view_108 : Tensor "bf16[s13, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [%arg7_1, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "bf16[s13, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [%arg0_1, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "bf16[3, s13, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select : Tensor "bf16[s13, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 0), kwargs = {})
#   %expand_16 : Tensor "bf16[s13, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%select, [%arg7_1, 8, 64, 64]), kwargs = {})
#   %clone_26 : Tensor "bf16[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_16,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_26
triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_54 = async_compile.triton('triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_54(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 1536*x1 + 98304*x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x4), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pm/cpmucng7tqcl7sudwxbooydqpttwtfbb2dpd24l367t6ylvmwyc2.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   k_8 => select_1
#   matmul_8 => clone_27, expand_17
#   qkv => add_tensor_10, convert_element_type_216, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   transpose_35 => permute_59
# Graph fragment:
#   %mm_default_10 : Tensor "bf16[64*s13, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_10]
#   %arg75_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %convert_element_type_216 : Tensor "bf16[1536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg75_1, torch.bfloat16), kwargs = {})
#   %add_tensor_10 : Tensor "bf16[64*s13, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %convert_element_type_216), kwargs = {})
#   %view_108 : Tensor "bf16[s13, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [%arg7_1, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "bf16[s13, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [%arg0_1, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "bf16[3, s13, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_1 : Tensor "bf16[s13, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 1), kwargs = {})
#   %permute_59 : Tensor "bf16[s13, 8, 64, 64][98304, 64, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [0, 1, 3, 2]), kwargs = {})
#   %expand_17 : Tensor "bf16[s13, 8, 64, 64][98304, 64, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_59, [%arg7_1, 8, 64, 64]), kwargs = {})
#   %clone_27 : Tensor "bf16[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_17,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_27
triton_poi_fused__to_copy_addmm_clone_expand_permute_select_transpose_view_55 = async_compile.triton('triton_poi_fused__to_copy_addmm_clone_expand_permute_select_transpose_view_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_clone_expand_permute_select_transpose_view_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_clone_expand_permute_select_transpose_view_55(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + 1536*x2 + 98304*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (512 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x2 + 64*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/d7/cd7iugtwvo2p6loal5zuhsjty3z4kzsz42qqxtxjhdlujey752ar.py
# Topologically Sorted Source Nodes: [matmul_8, attn_9, out_20], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_9 => div_5, exp_5, sum_5
#   matmul_8 => view_112
#   out_20 => convert_element_type_225
# Graph fragment:
#   %bmm_20 : Tensor "bf16[8*s13, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %amax_default_5 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 512*s13]cuda:0" = PlaceHolder[target=amax_default_5]
#   %sum_5 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 512*s13]cuda:0" = PlaceHolder[target=sum_5]
#   %view_112 : Tensor "bf16[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_20, [%arg7_1, 8, 64, 64]), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f32[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_112, torch.float32), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_6, 1), kwargs = {})
#   %amax_default_5 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_10, [-1], True), kwargs = {})
#   %sub_tensor_5 : Tensor "f32[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_10, %amax_default_5), kwargs = {})
#   %mul_tensor_11 : Tensor "f32[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_5, 0.125), kwargs = {})
#   %exp_5 : Tensor "f32[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_11,), kwargs = {})
#   %sum_5 : Tensor "f32[s13, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1], True), kwargs = {})
#   %div_5 : Tensor "f32[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_5), kwargs = {})
#   %convert_element_type_225 : Tensor "bf16[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_5, torch.bfloat16), kwargs = {})
#   return %amax_default_5,%sum_5,%convert_element_type_225
triton_per_fused__softmax__to_copy_amax_mul_sub_view_56 = async_compile.triton('triton_per_fused__softmax__to_copy_amax_mul_sub_view_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_amax_mul_sub_view_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax__to_copy_amax_mul_sub_view_56(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tmp3 - tmp7
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = (tmp11 / tmp15)
    tmp17 = tmp16.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/yk/cyk3nrb2egdlpe2sdvobprcsb4cpx554nvi4r6kedn4xa6v44t2g.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   out_20 => clone_28, expand_19
#   qkv => add_tensor_10, convert_element_type_216, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   v_20 => select_2
# Graph fragment:
#   %mm_default_10 : Tensor "bf16[64*s13, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_10]
#   %arg75_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %convert_element_type_216 : Tensor "bf16[1536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg75_1, torch.bfloat16), kwargs = {})
#   %add_tensor_10 : Tensor "bf16[64*s13, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %convert_element_type_216), kwargs = {})
#   %view_108 : Tensor "bf16[s13, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [%arg7_1, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "bf16[s13, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [%arg0_1, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "bf16[3, s13, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_2 : Tensor "bf16[s13, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 2), kwargs = {})
#   %expand_19 : Tensor "bf16[s13, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%select_2, [%arg7_1, 8, 64, 64]), kwargs = {})
#   %clone_28 : Tensor "bf16[s13, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_28
triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_57 = async_compile.triton('triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_57(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + 64*x2 + 1536*x1 + 98304*x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x4), tmp3, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/jy/cjydmfb7wwjngn3ljo2vcptwpyfbbbcb6gfzu4g4ogpf3sdzxgq6.py
# Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.addmm, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_21 => add_1549
#   h_8 => convert_element_type_213, convert_element_type_214, mul_1162, sigmoid_16
#   h_9 => convert_element_type_234, convert_element_type_235, convert_element_type_236, convolution_15
#   out_22 => add_tensor_9, convert_element_type_228, view_118
#   out_23 => add_1744
#   out_24 => add_1753, add_1754, convert_element_type_233, mul_1358, mul_1359, rsqrt_7, sub_414, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   view_50 => view_106
#   x_20 => mul_1153, sigmoid_15
#   x_21 => convert_element_type_210, convert_element_type_211, convert_element_type_212, convolution_14
#   x_flat_6 => permute_56
# Graph fragment:
#   %buf196 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf196]
#   %buf194 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf194]
#   %convert_element_type_184 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convert_element_type_184]
#   %mm_default_9 : Tensor "bf16[64*s13, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default_9]
#   %arg77_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %convert_element_type_233 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=convert_element_type_233]
#   %getitem_41 : Tensor "f32[s13, 64, 1][64, 1, 64*s13]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf216 : Tensor "f32[s13, 64, 1][64, 1, 64*s13]cuda:0" = PlaceHolder[target=buf216]
#   %arg78_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg78_1]
#   %arg79_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %sigmoid_15 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1528,), kwargs = {})
#   %mul_1153 : Tensor "f32[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1528, %sigmoid_15), kwargs = {})
#   %convert_element_type_212 : Tensor "bf16[s13, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1153, torch.bfloat16), kwargs = {})
#   %convert_element_type_211 : Tensor "bf16[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg70_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_210 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg71_1, torch.bfloat16), kwargs = {})
#   %convolution_14 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_212, %convert_element_type_211, %convert_element_type_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_1549 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convert_element_type_184), kwargs = {})
#   %convert_element_type_213 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1549, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_1162 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1162, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [%arg0_1, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_228 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg77_1, torch.bfloat16), kwargs = {})
#   %add_tensor_9 : Tensor "bf16[64*s13, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_9, %convert_element_type_228), kwargs = {})
#   %view_118 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_9, [%arg7_1, 64, 512]), kwargs = {})
#   %add_1744 : Tensor "bf16[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %convert_element_type_233 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1744, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_233, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_414 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_41), kwargs = {})
#   %add_1753 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1753,), kwargs = {})
#   %mul_1358 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_414, %rsqrt_7), kwargs = {})
#   %mul_1359 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1358, %arg78_1), kwargs = {})
#   %add_1754 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1359, %arg79_1), kwargs = {})
#   %permute_62 : Tensor "f32[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_1754, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %convert_element_type_236 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_119, torch.bfloat16), kwargs = {})
#   %convert_element_type_235 : Tensor "bf16[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg80_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_234 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg81_1, torch.bfloat16), kwargs = {})
#   %convolution_15 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_236, %convert_element_type_235, %convert_element_type_234, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %convert_element_type_233,%getitem_41,%buf216,%buf220
triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_58 = async_compile.triton('triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp8 + tmp12
    tmp14 = tmp13.to(tl.float32)
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
    tmp42 = tmp41.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 512*x0), tmp42, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lo/clo7tgobp3jcikkd7eslvoizhsprmea5sapmkjiujnq33jtwp5zy.py
# Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_9 => convert_element_type_234, convert_element_type_235, convert_element_type_236, convolution_15
#   out_24 => add_1753, add_1754, mul_1358, mul_1359, rsqrt_7, sub_414, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
# Graph fragment:
#   %arg80_1 : Tensor "f32[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0" = PlaceHolder[target=arg80_1]
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_233, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_414 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_41), kwargs = {})
#   %add_1753 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1753,), kwargs = {})
#   %mul_1358 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_414, %rsqrt_7), kwargs = {})
#   %mul_1359 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1358, %arg78_1), kwargs = {})
#   %add_1754 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1359, %arg79_1), kwargs = {})
#   %permute_62 : Tensor "f32[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_1754, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %convert_element_type_236 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_119, torch.bfloat16), kwargs = {})
#   %convert_element_type_235 : Tensor "bf16[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg80_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_234 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg81_1, torch.bfloat16), kwargs = {})
#   %convolution_15 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_236, %convert_element_type_235, %convert_element_type_234, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf221
triton_poi_fused__to_copy_convolution_native_layer_norm_transpose_view_59 = async_compile.triton('triton_poi_fused__to_copy_convolution_native_layer_norm_transpose_view_59', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_native_layer_norm_transpose_view_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_native_layer_norm_transpose_view_59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dg/cdg5v4pfx5yqhvv3er3q7na5wm3ttpmv4gb65u2egc52766u4c7n.py
# Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   gate => unsqueeze_7
#   h_10 => add_1812
#   h_16_gated => mul_1387
#   h_9 => convert_element_type_234, convert_element_type_235, convert_element_type_236, convolution_15
#   input_26 => add_tensor_7, convert_element_type_244
#   input_27 => sigmoid_18
#   out_24 => add_1753, add_1754, mul_1358, mul_1359, rsqrt_7, sub_414, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   unsqueeze_4 => unsqueeze_6
# Graph fragment:
#   %buf223 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf223]
#   %buf222 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf222]
#   %convert_element_type_152 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convert_element_type_152]
#   %mm_default_7 : Tensor "bf16[s13, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_7]
#   %arg85_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg85_1]
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_233, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_414 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_41), kwargs = {})
#   %add_1753 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[s13, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1753,), kwargs = {})
#   %mul_1358 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_414, %rsqrt_7), kwargs = {})
#   %mul_1359 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1358, %arg78_1), kwargs = {})
#   %add_1754 : Tensor "f32[s13, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1359, %arg79_1), kwargs = {})
#   %permute_62 : Tensor "f32[s13, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_1754, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [%arg0_1, 512, 8, 8]), kwargs = {})
#   %convert_element_type_236 : Tensor "bf16[s13, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_119, torch.bfloat16), kwargs = {})
#   %convert_element_type_235 : Tensor "bf16[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg80_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_234 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg81_1, torch.bfloat16), kwargs = {})
#   %convolution_15 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_236, %convert_element_type_235, %convert_element_type_234, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %convert_element_type_244 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg85_1, torch.bfloat16), kwargs = {})
#   %add_tensor_7 : Tensor "bf16[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_7, %convert_element_type_244), kwargs = {})
#   %sigmoid_18 : Tensor "bf16[s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor_7,), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[s13, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_18, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[s13, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %mul_1387 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_152, %unsqueeze_7), kwargs = {})
#   %add_1812 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %mul_1387), kwargs = {})
#   return %add_1812
triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_60 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 256
    y0 = (yindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2 + 256*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp3 * tmp8
    tmp10 = tmp2 + tmp9
    tl.store(out_ptr0 + (y0 + 256*x2 + 65536*y1), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2a/c2aitba73pr4ujzexd656k5zijo62ovvkcxypb5pqsakxldb3nmt.py
# Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.t, aten.mm]
# Source node to ATen node mapping:
#   q_13 => permute_65
#   q_14 => convert_element_type_249, mm_8, permute_66, view_121
#   view_52 => view_120
# Graph fragment:
#   %add_1812 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_1812]
#   %view_120 : Tensor "bf16[s13, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1812, [%arg0_1, 256, 256]), kwargs = {})
#   %permute_65 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_120, [0, 2, 1]), kwargs = {})
#   %view_121 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_65, [%mul_193, 256]), kwargs = {})
#   %convert_element_type_249 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg86_1, torch.bfloat16), kwargs = {})
#   %permute_66 : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_249, [1, 0]), kwargs = {})
#   %mm_8 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_121, %permute_66), kwargs = {})
#   return %buf230
triton_poi_fused__to_copy_mm_t_transpose_view_61 = async_compile.triton('triton_poi_fused__to_copy_mm_t_transpose_view_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mm_t_transpose_view_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mm_t_transpose_view_61(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256*x1 + 65536*(x0 // 256) + ((x0 % 256))), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vu/cvunaoz2r5uhdp2cvswcyoh2j3qp53r4ehxdkmfirgqnduh2xa2z.py
# Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
# Source node to ATen node mapping:
#   input_28 => add_tensor_6, convert_element_type_261, view_134
#   out_27 => view_135
#   transpose_44 => permute_74
#   x_22 => add_2012
#   x_23 => convert_element_type_266, convert_element_type_267, convolution_16
# Graph fragment:
#   %add_1812 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_1812]
#   %mm_default_6 : Tensor "bf16[256*s13, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_6]
#   %arg89_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg89_1]
#   %convert_element_type_261 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg89_1, torch.bfloat16), kwargs = {})
#   %add_tensor_6 : Tensor "bf16[256*s13, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %convert_element_type_261), kwargs = {})
#   %view_134 : Tensor "bf16[s13, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_6, [%arg7_1, 256, 256]), kwargs = {})
#   %permute_74 : Tensor "bf16[s13, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_134, [0, 2, 1]), kwargs = {})
#   %view_135 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_74, [%arg0_1, 256, 16, 16]), kwargs = {})
#   %add_2012 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1812, %view_135), kwargs = {})
#   %convert_element_type_267 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg90_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_266 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg91_1, torch.bfloat16), kwargs = {})
#   %convolution_16 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_2012, %convert_element_type_267, %convert_element_type_266, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf242
triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_62 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_62(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 65536*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/zg/czgfjhwwietq2pioytdfxvjc5avao5yfly7a5placcahmcspzhnk.py
# Topologically Sorted Source Nodes: [x_24, x_25, add_27, h_11], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_27 => add_2118
#   h_11 => convert_element_type_277, convert_element_type_278, mul_1633, sigmoid_20
#   x_24 => mul_1624, sigmoid_19
#   x_25 => convert_element_type_274, convert_element_type_275, convert_element_type_276, convolution_17
# Graph fragment:
#   %buf259 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf259]
#   %buf257 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf257]
#   %add_1812 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_1812]
#   %sigmoid_19 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2097,), kwargs = {})
#   %mul_1624 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2097, %sigmoid_19), kwargs = {})
#   %convert_element_type_276 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1624, torch.bfloat16), kwargs = {})
#   %convert_element_type_275 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg92_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_274 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg93_1, torch.bfloat16), kwargs = {})
#   %convolution_17 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_276, %convert_element_type_275, %convert_element_type_274, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_2118 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_1812), kwargs = {})
#   %convert_element_type_277 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2118, torch.float32), kwargs = {})
#   %sigmoid_20 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_277,), kwargs = {})
#   %mul_1633 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_277, %sigmoid_20), kwargs = {})
#   %convert_element_type_278 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1633, torch.bfloat16), kwargs = {})
#   return %convert_element_type_278
triton_poi_fused__to_copy_add_convolution_silu_63 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_63(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0 + 256*x2 + 65536*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 256*y3), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vl/cvlxos5x55lemyjurtyinvd4ydev7ek6a5ydt2n3gxeahses6tai.py
# Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_43 => add_3362
#   h_15 => convert_element_type_397, convert_element_type_398, mul_2609, sigmoid_28
#   h_16 => convert_element_type_399, convert_element_type_400, convolution_26
#   x_40 => mul_2600, sigmoid_27
#   x_41 => convert_element_type_394, convert_element_type_395, convert_element_type_396, convolution_25
# Graph fragment:
#   %arg126_1 : Tensor "f32[256, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=arg126_1]
#   %sigmoid_27 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3341,), kwargs = {})
#   %mul_2600 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3341, %sigmoid_27), kwargs = {})
#   %convert_element_type_396 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2600, torch.bfloat16), kwargs = {})
#   %convert_element_type_395 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg124_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_394 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg125_1, torch.bfloat16), kwargs = {})
#   %convolution_25 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_396, %convert_element_type_395, %convert_element_type_394, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3362 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convert_element_type_368), kwargs = {})
#   %convert_element_type_397 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3362, torch.float32), kwargs = {})
#   %sigmoid_28 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_397,), kwargs = {})
#   %mul_2609 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_397, %sigmoid_28), kwargs = {})
#   %convert_element_type_398 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2609, torch.bfloat16), kwargs = {})
#   %convert_element_type_400 : Tensor "bf16[256, 128, 4, 4][2048, 1, 512, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg126_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_399 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg127_1, torch.bfloat16), kwargs = {})
#   %convolution_26 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_398, %convert_element_type_400, %convert_element_type_399, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf379
triton_poi_fused__to_copy_add_convolution_silu_64 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_64', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rq/crq3y4ml7ckvbapdfbd42isq2hd6iavd5t2qu6axnlkneiun2wly.py
# Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_40 => convert_element_type_409
# Graph fragment:
#   %arg130_1 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=arg130_1]
#   %convert_element_type_409 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg130_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_409
triton_poi_fused__to_copy_65 = async_compile.triton('triton_poi_fused__to_copy_65', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_65(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/e6/ce6mwde4la3xtasssudxjc6nsqf25srtbnafydj7u7dct2ketjhx.py
# Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   add_43 => add_3362
#   gate_1 => unsqueeze_14
#   h_15 => convert_element_type_397, convert_element_type_398, mul_2609, sigmoid_28
#   h_16 => convert_element_type_399, convert_element_type_400, convolution_26
#   h_17 => add_3404
#   h_32_gated => mul_2627
#   input_40 => add_tensor, convert_element_type_408
#   input_41 => sigmoid_30
#   unsqueeze_11 => unsqueeze_13
#   x_40 => mul_2600, sigmoid_27
#   x_41 => convert_element_type_394, convert_element_type_395, convert_element_type_396, convolution_25
# Graph fragment:
#   %buf381 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf381]
#   %buf380 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf380]
#   %convert_element_type_90 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_90]
#   %mm_default : Tensor "bf16[s13, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg131_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg131_1]
#   %sigmoid_27 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3341,), kwargs = {})
#   %mul_2600 : Tensor "f32[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3341, %sigmoid_27), kwargs = {})
#   %convert_element_type_396 : Tensor "bf16[s13, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2600, torch.bfloat16), kwargs = {})
#   %convert_element_type_395 : Tensor "bf16[256, 256, 1, 1][256, 1, 256, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg124_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_394 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg125_1, torch.bfloat16), kwargs = {})
#   %convolution_25 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_396, %convert_element_type_395, %convert_element_type_394, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3362 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convert_element_type_368), kwargs = {})
#   %convert_element_type_397 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3362, torch.float32), kwargs = {})
#   %sigmoid_28 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_397,), kwargs = {})
#   %mul_2609 : Tensor "f32[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_397, %sigmoid_28), kwargs = {})
#   %convert_element_type_398 : Tensor "bf16[s13, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2609, torch.bfloat16), kwargs = {})
#   %convert_element_type_400 : Tensor "bf16[256, 128, 4, 4][2048, 1, 512, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg126_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_399 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg127_1, torch.bfloat16), kwargs = {})
#   %convolution_26 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_398, %convert_element_type_400, %convert_element_type_399, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %convert_element_type_408 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg131_1, torch.bfloat16), kwargs = {})
#   %add_tensor : Tensor "bf16[s13, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %convert_element_type_408), kwargs = {})
#   %sigmoid_30 : Tensor "bf16[s13, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor,), kwargs = {})
#   %unsqueeze_13 : Tensor "bf16[s13, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_30, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[s13, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_2627 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_90, %unsqueeze_14), kwargs = {})
#   %add_3404 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %mul_2627), kwargs = {})
#   return %add_3404
triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_66 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_66', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_66(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 131072
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 + 128*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp3 * tmp8
    tmp10 = tmp2 + tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wd/cwdat3eu6sk2bfqpojpmv23vjew3t7oogbq3ajnit4k7fw5cs6hg.py
# Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_59 => add_3954
#   h_22 => convert_element_type_476, mul_2944, sigmoid_40
#   input_42 => clone_65, var_mean_18, view_270
#   x_55 => mul_2935, sigmoid_39
#   x_56 => convert_element_type_473, convert_element_type_474, convert_element_type_475, convolution_36
# Graph fragment:
#   %buf466 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf466]
#   %buf464 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf464]
#   %convert_element_type_464 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_464]
#   %sigmoid_39 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3933,), kwargs = {})
#   %mul_2935 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3933, %sigmoid_39), kwargs = {})
#   %convert_element_type_475 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2935, torch.bfloat16), kwargs = {})
#   %convert_element_type_474 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg150_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_473 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg151_1, torch.bfloat16), kwargs = {})
#   %convolution_36 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_475, %convert_element_type_474, %convert_element_type_473, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3954 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %convert_element_type_464), kwargs = {})
#   %convert_element_type_476 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3954, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_2944 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_65 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_2944,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_65, [%arg7_1, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_103,%buf468
triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_67 = async_compile.triton('triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_67(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.sigmoid(tmp5)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
    tl.store(out_ptr1 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/4u/c4ulv7ginunclap7ttjc7c7dakendson6kuifl5by6vpco4rikmv.py
# Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42, input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_59 => add_3954
#   h_22 => convert_element_type_476, mul_2944, sigmoid_40
#   input_42 => add_3975, add_3976, clone_65, mul_2961, mul_2962, rsqrt_18, sub_931, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, var_mean_18, view_270, view_271
#   input_43 => mul_2969, sigmoid_41
#   input_44 => convert_element_type_479, convert_element_type_480, convert_element_type_481, convolution_37
#   x_55 => mul_2935, sigmoid_39
#   x_56 => convert_element_type_473, convert_element_type_474, convert_element_type_475, convolution_36
# Graph fragment:
#   %buf466 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf466]
#   %buf464 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf464]
#   %convert_element_type_464 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_464]
#   %getitem_103 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=getitem_103]
#   %buf468 : Tensor "f32[s13, 32, 1, 1][32, 1, 32*s13, 32*s13]cuda:0" = PlaceHolder[target=buf468]
#   %arg152_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg152_1]
#   %arg153_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg153_1]
#   %add_3976 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_3976]
#   %sigmoid_39 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3933,), kwargs = {})
#   %mul_2935 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3933, %sigmoid_39), kwargs = {})
#   %convert_element_type_475 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2935, torch.bfloat16), kwargs = {})
#   %convert_element_type_474 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg150_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_473 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg151_1, torch.bfloat16), kwargs = {})
#   %convolution_36 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_475, %convert_element_type_474, %convert_element_type_473, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3954 : Tensor "bf16[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %convert_element_type_464), kwargs = {})
#   %convert_element_type_476 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3954, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_2944 : Tensor "f32[s13, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_65 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_2944,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_65, [%arg7_1, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_931 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %getitem_103), kwargs = {})
#   %add_3975 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_102, 1e-05), kwargs = {})
#   %rsqrt_18 : Tensor "f32[s13, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3975,), kwargs = {})
#   %mul_2961 : Tensor "f32[s13, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_931, %rsqrt_18), kwargs = {})
#   %view_271 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_2961, [%arg7_1, 128, 32, 32]), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg152_1, 0), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, 2), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 3), kwargs = {})
#   %mul_2962 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_271, %unsqueeze_17), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg153_1, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 3), kwargs = {})
#   %add_3976 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2962, %unsqueeze_20), kwargs = {})
#   %sigmoid_41 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3976,), kwargs = {})
#   %mul_2969 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3976, %sigmoid_41), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2969, torch.bfloat16), kwargs = {})
#   %convert_element_type_480 : Tensor "bf16[4, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg154_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg155_1, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[s13, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_3976,%buf471
triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_68 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 1024
    y0 = (yindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*y1 + (x2 // 4)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*y1 + (x2 // 4)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 - tmp8
    tmp11 = 4096.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sigmoid(tmp20)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 1024*x2 + 131072*y1), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lc/clcjcljljexm3lzxj2cf7wne5hckb3yc6fpjg4qqw33uimfhjuwl.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_2969, sigmoid_41
#   input_44 => convert_element_type_479, convert_element_type_480, convert_element_type_481, convolution_37
# Graph fragment:
#   %arg155_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg155_1]
#   %sigmoid_41 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3976,), kwargs = {})
#   %mul_2969 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3976, %sigmoid_41), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2969, torch.bfloat16), kwargs = {})
#   %convert_element_type_480 : Tensor "bf16[4, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg154_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg155_1, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[s13, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf473
triton_poi_fused__to_copy_convolution_silu_69 = async_compile.triton('triton_poi_fused__to_copy_convolution_silu_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_silu_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_silu_69(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/au/cau6545nlv7nugjomxhvqiifs5rgg3vduf2ov5lmmh7cvdnbmx4o.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_2969, sigmoid_41
#   input_44 => convert_element_type_479, convert_element_type_480, convert_element_type_481, convolution_37
# Graph fragment:
#   %buf471 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=buf471]
#   %sigmoid_41 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3976,), kwargs = {})
#   %mul_2969 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3976, %sigmoid_41), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2969, torch.bfloat16), kwargs = {})
#   %convert_element_type_480 : Tensor "bf16[4, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg154_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg155_1, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[s13, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf474
triton_poi_fused__to_copy_convolution_silu_70 = async_compile.triton('triton_poi_fused__to_copy_convolution_silu_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_silu_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_silu_70(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 1024
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 131072*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/xr/cxr7mey2mm24rmp6ocpazue477t4xtj5xdi5n7ajdxz2ukfk4vcl.py
# Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_43 => mul_2969, sigmoid_41
#   input_44 => convert_element_type_479, convert_element_type_480, convert_element_type_481, convolution_37
# Graph fragment:
#   %buf475 : Tensor "bf16[s13, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf475]
#   %buf473 : Tensor "bf16[4][1]cuda:0" = PlaceHolder[target=buf473]
#   %sigmoid_41 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3976,), kwargs = {})
#   %mul_2969 : Tensor "f32[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3976, %sigmoid_41), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[s13, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2969, torch.bfloat16), kwargs = {})
#   %convert_element_type_480 : Tensor "bf16[4, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg154_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg155_1, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[s13, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_37
triton_poi_fused__to_copy_convolution_silu_71 = async_compile.triton('triton_poi_fused__to_copy_convolution_silu_71', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_silu_71', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_silu_71(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1 = args
        args.clear()
        s13 = arg0_1
        s73 = arg7_1
        assert_size_stride(arg1_1, (s13, ), (1, ))
        assert_size_stride(arg2_1, (1024, 256), (256, 1))
        assert_size_stride(arg3_1, (1024, ), (1, ))
        assert_size_stride(arg4_1, (256, 1024), (1024, 1))
        assert_size_stride(arg5_1, (256, ), (1, ))
        assert_size_stride(arg6_1, (4, 256), (256, 1))
        assert_size_stride(arg8_1, (s13, ), (1, ))
        assert_size_stride(arg9_1, (512, 512), (512, 1))
        assert_size_stride(arg10_1, (512, ), (1, ))
        assert_size_stride(arg11_1, (256, 512), (512, 1))
        assert_size_stride(arg12_1, (256, ), (1, ))
        assert_size_stride(arg13_1, (256, 256), (256, 1))
        assert_size_stride(arg14_1, (256, ), (1, ))
        assert_size_stride(arg15_1, (3200, 256), (256, 1))
        assert_size_stride(arg16_1, (3200, ), (1, ))
        assert_size_stride(arg17_1, (256, 256), (256, 1))
        assert_size_stride(arg18_1, (256, ), (1, ))
        assert_size_stride(arg19_1, (6400, 256), (256, 1))
        assert_size_stride(arg20_1, (6400, ), (1, ))
        assert_size_stride(arg21_1, (256, 256), (256, 1))
        assert_size_stride(arg22_1, (256, ), (1, ))
        assert_size_stride(arg23_1, (12800, 256), (256, 1))
        assert_size_stride(arg24_1, (12800, ), (1, ))
        assert_size_stride(arg25_1, (128, 4, 3, 3), (36, 1, 12, 4))
        assert_size_stride(arg26_1, (128, ), (1, ))
        assert_size_stride(arg27_1, (s13, 4, 32, 32), (4096, 1024, 32, 1))
        assert_size_stride(arg28_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg29_1, (128, ), (1, ))
        assert_size_stride(arg30_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg31_1, (128, ), (1, ))
        assert_size_stride(arg32_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg33_1, (128, ), (1, ))
        assert_size_stride(arg34_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg35_1, (128, ), (1, ))
        assert_size_stride(arg36_1, (256, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg37_1, (256, ), (1, ))
        assert_size_stride(arg38_1, (256, 256), (256, 1))
        assert_size_stride(arg39_1, (512, 256), (256, 1))
        assert_size_stride(arg40_1, (256, 256), (256, 1))
        assert_size_stride(arg41_1, (256, ), (1, ))
        assert_size_stride(arg42_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg43_1, (256, ), (1, ))
        assert_size_stride(arg44_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg45_1, (256, ), (1, ))
        assert_size_stride(arg46_1, (256, 256), (256, 1))
        assert_size_stride(arg47_1, (512, 256), (256, 1))
        assert_size_stride(arg48_1, (256, 256), (256, 1))
        assert_size_stride(arg49_1, (256, ), (1, ))
        assert_size_stride(arg50_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg51_1, (256, ), (1, ))
        assert_size_stride(arg52_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg53_1, (256, ), (1, ))
        assert_size_stride(arg54_1, (512, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg55_1, (512, ), (1, ))
        assert_size_stride(arg56_1, (512, 512), (512, 1))
        assert_size_stride(arg57_1, (1024, 256), (256, 1))
        assert_size_stride(arg58_1, (512, 512), (512, 1))
        assert_size_stride(arg59_1, (512, ), (1, ))
        assert_size_stride(arg60_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(arg61_1, (512, ), (1, ))
        assert_size_stride(arg62_1, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(arg63_1, (512, ), (1, ))
        assert_size_stride(arg64_1, (512, 512), (512, 1))
        assert_size_stride(arg65_1, (1024, 256), (256, 1))
        assert_size_stride(arg66_1, (512, 512), (512, 1))
        assert_size_stride(arg67_1, (512, ), (1, ))
        assert_size_stride(arg68_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(arg69_1, (512, ), (1, ))
        assert_size_stride(arg70_1, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(arg71_1, (512, ), (1, ))
        assert_size_stride(arg72_1, (512, ), (1, ))
        assert_size_stride(arg73_1, (512, ), (1, ))
        assert_size_stride(arg74_1, (1536, 512), (512, 1))
        assert_size_stride(arg75_1, (1536, ), (1, ))
        assert_size_stride(arg76_1, (512, 512), (512, 1))
        assert_size_stride(arg77_1, (512, ), (1, ))
        assert_size_stride(arg78_1, (512, ), (1, ))
        assert_size_stride(arg79_1, (512, ), (1, ))
        assert_size_stride(arg80_1, (512, 256, 4, 4), (4096, 1, 1024, 256))
        assert_size_stride(arg81_1, (256, ), (1, ))
        assert_size_stride(arg82_1, (256, 256), (256, 1))
        assert_size_stride(arg83_1, (256, ), (1, ))
        assert_size_stride(arg84_1, (256, 256), (256, 1))
        assert_size_stride(arg85_1, (256, ), (1, ))
        assert_size_stride(arg86_1, (256, 256), (256, 1))
        assert_size_stride(arg87_1, (512, 256), (256, 1))
        assert_size_stride(arg88_1, (256, 256), (256, 1))
        assert_size_stride(arg89_1, (256, ), (1, ))
        assert_size_stride(arg90_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg91_1, (256, ), (1, ))
        assert_size_stride(arg92_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg93_1, (256, ), (1, ))
        assert_size_stride(arg94_1, (256, 256), (256, 1))
        assert_size_stride(arg95_1, (512, 256), (256, 1))
        assert_size_stride(arg96_1, (256, 256), (256, 1))
        assert_size_stride(arg97_1, (256, ), (1, ))
        assert_size_stride(arg98_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg99_1, (256, ), (1, ))
        assert_size_stride(arg100_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg101_1, (256, ), (1, ))
        assert_size_stride(arg102_1, (256, 256), (256, 1))
        assert_size_stride(arg103_1, (512, 256), (256, 1))
        assert_size_stride(arg104_1, (256, 256), (256, 1))
        assert_size_stride(arg105_1, (256, ), (1, ))
        assert_size_stride(arg106_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg107_1, (256, ), (1, ))
        assert_size_stride(arg108_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg109_1, (256, ), (1, ))
        assert_size_stride(arg110_1, (256, 256), (256, 1))
        assert_size_stride(arg111_1, (512, 256), (256, 1))
        assert_size_stride(arg112_1, (256, 256), (256, 1))
        assert_size_stride(arg113_1, (256, ), (1, ))
        assert_size_stride(arg114_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg115_1, (256, ), (1, ))
        assert_size_stride(arg116_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg117_1, (256, ), (1, ))
        assert_size_stride(arg118_1, (256, 256), (256, 1))
        assert_size_stride(arg119_1, (512, 256), (256, 1))
        assert_size_stride(arg120_1, (256, 256), (256, 1))
        assert_size_stride(arg121_1, (256, ), (1, ))
        assert_size_stride(arg122_1, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(arg123_1, (256, ), (1, ))
        assert_size_stride(arg124_1, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(arg125_1, (256, ), (1, ))
        assert_size_stride(arg126_1, (256, 128, 4, 4), (2048, 1, 512, 128))
        assert_size_stride(arg127_1, (128, ), (1, ))
        assert_size_stride(arg128_1, (256, 256), (256, 1))
        assert_size_stride(arg129_1, (256, ), (1, ))
        assert_size_stride(arg130_1, (128, 256), (256, 1))
        assert_size_stride(arg131_1, (128, ), (1, ))
        assert_size_stride(arg132_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg133_1, (128, ), (1, ))
        assert_size_stride(arg134_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg135_1, (128, ), (1, ))
        assert_size_stride(arg136_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg137_1, (128, ), (1, ))
        assert_size_stride(arg138_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg139_1, (128, ), (1, ))
        assert_size_stride(arg140_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg141_1, (128, ), (1, ))
        assert_size_stride(arg142_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg143_1, (128, ), (1, ))
        assert_size_stride(arg144_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg145_1, (128, ), (1, ))
        assert_size_stride(arg146_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg147_1, (128, ), (1, ))
        assert_size_stride(arg148_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg149_1, (128, ), (1, ))
        assert_size_stride(arg150_1, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(arg151_1, (128, ), (1, ))
        assert_size_stride(arg152_1, (128, ), (1, ))
        assert_size_stride(arg153_1, (128, ), (1, ))
        assert_size_stride(arg154_1, (4, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg155_1, (4, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((s13, 4, 32, 32), (4096, 1, 128, 4), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            triton_poi_fused__to_copy_convolution_0_ynumel = 4*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_0.run(arg27_1, buf3, triton_poi_fused__to_copy_convolution_0_ynumel, 1024, stream=stream0)
            del arg27_1
            buf1 = empty_strided_cuda((128, 4, 3, 3), (36, 1, 12, 4), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_1.run(arg25_1, buf1, 4608, stream=stream0)
            del arg25_1
            buf2 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg26_1, buf2, 128, stream=stream0)
            del arg26_1
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            buf4 = extern_kernels.convolution(buf3, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf4, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf1
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            triton_poi_fused__to_copy_convolution_3_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_3.run(buf5, buf2, triton_poi_fused__to_copy_convolution_3_xnumel, stream=stream0)
            buf6 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_4.run(arg28_1, buf6, 147456, stream=stream0)
            del arg28_1
            buf7 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg29_1, buf7, 128, stream=stream0)
            del arg29_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            buf8 = extern_kernels.convolution(buf5, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf8, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf9 = empty_strided_cuda((s13, 32, 1, 1), (32, 1, 32*s13, 32*s13), torch.float32)
            buf10 = empty_strided_cuda((s13, 32, 1, 1), (32, 1, 32*s13, 32*s13), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_norm], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5.run(buf8, buf7, buf9, buf10, triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel, 4096, stream=stream0)
            buf12 = empty_strided_cuda((s13, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten._to_copy]
            triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_6_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_6.run(arg1_1, buf12, triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_6_xnumel, stream=stream0)
            del arg1_1
            buf13 = empty_strided_cuda((1024, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg2_1, buf13, 262144, stream=stream0)
            del arg2_1
            buf14 = empty_strided_cuda((s13, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, arange, mul, truediv, freqs, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.unsqueeze, aten.arange, aten.mul, aten.div, aten.exp, aten.cos, aten.sin, aten.cat, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf12, reinterpret_tensor(buf13, (256, 1024), (1, 256), 0), out=buf14)
            buf15 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            triton_poi_fused__to_copy_addmm_silu_8_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_8.run(buf15, arg3_1, triton_poi_fused__to_copy_addmm_silu_8_xnumel, stream=stream0)
            del arg3_1
            buf16 = reinterpret_tensor(buf13, (256, 1024), (1024, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg4_1, buf16, 262144, stream=stream0)
            del arg4_1
            buf17 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg5_1, buf17, 256, stream=stream0)
            del arg5_1
            buf18 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [input_3, input_1, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf17, buf15, reinterpret_tensor(buf16, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf18)
            buf19 = empty_strided_cuda((s13, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy]
            triton_poi_fused__to_copy_cat_embedding_10_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_embedding_10.run(buf18, arg8_1, arg6_1, buf19, triton_poi_fused__to_copy_cat_embedding_10_xnumel, stream=stream0)
            del arg6_1
            del arg8_1
            buf20 = reinterpret_tensor(buf16, (512, 512), (512, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg9_1, buf20, 262144, stream=stream0)
            del arg9_1
            buf21 = empty_strided_cuda((s13, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf19, reinterpret_tensor(buf20, (512, 512), (1, 512), 0), out=buf21)
            buf22 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            triton_poi_fused__to_copy_addmm_silu_11_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_11.run(buf22, arg10_1, triton_poi_fused__to_copy_addmm_silu_11_xnumel, stream=stream0)
            del arg10_1
            buf23 = empty_strided_cuda((256, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg11_1, buf23, 131072, stream=stream0)
            del arg11_1
            buf24 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg12_1, buf24, 256, stream=stream0)
            del arg12_1
            buf25 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_6, input_4, input_5], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf24, buf22, reinterpret_tensor(buf23, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf25)
            buf26 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg13_1, buf26, 65536, stream=stream0)
            del arg13_1
            buf27 = empty_strided_cuda((s13, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf26, (256, 256), (1, 256), 0), out=buf27)
            buf28 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            triton_poi_fused__to_copy_addmm_silu_14_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_14.run(buf28, arg14_1, triton_poi_fused__to_copy_addmm_silu_14_xnumel, stream=stream0)
            del arg14_1
            buf29 = empty_strided_cuda((3200, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(arg15_1, buf29, 819200, stream=stream0)
            del arg15_1
            buf30 = empty_strided_cuda((3200, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_16.run(arg16_1, buf30, 3200, stream=stream0)
            del arg16_1
            buf31 = empty_strided_cuda((s13, 3200), (3200, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9, input_7, input_8], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf30, buf28, reinterpret_tensor(buf29, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf31)
            del buf29
            del buf30
            buf32 = empty_strided_cuda((s13, 128, 1024), (131072, 1, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, x_norm, x_flat, v_t_x], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17.run(buf8, buf7, buf9, buf10, buf32, triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel, stream=stream0)
            buf33 = empty_strided_cuda((s13, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, x_norm, split, v_1, transpose, x_flat, v_t_x], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 12, 128), (3200, 1, 12), 1536), buf32, out=buf33)
            buf34 = reinterpret_tensor(buf32, (s13, 128, 1024), (131072, 1024, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 128, 12), (3200, 12, 1), 0), buf33, out=buf34)
            buf39 = empty_strided_cuda((s13, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, x_norm, split, x_flat, out, view_4, shift_1, out_1, x_1, x_2], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18.run(buf8, buf7, buf9, buf10, buf34, buf31, buf39, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel, 1024, stream=stream0)
            del buf34
            del buf8
            buf37 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_19.run(arg30_1, buf37, 16384, stream=stream0)
            del arg30_1
            buf38 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg31_1, buf38, 128, stream=stream0)
            del arg31_1
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf40 = extern_kernels.convolution(buf39, buf37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf40, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [x_1, x_2, add_2, h_1], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_20_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_20.run(buf41, buf38, buf5, triton_poi_fused__to_copy_add_convolution_silu_20_xnumel, stream=stream0)
            buf42 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_4.run(arg32_1, buf42, 147456, stream=stream0)
            del arg32_1
            buf43 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg33_1, buf43, 128, stream=stream0)
            del arg33_1
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.convolution]
            buf44 = extern_kernels.convolution(buf41, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf44, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf42
            buf45 = buf9; del buf9  # reuse
            buf46 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5.run(buf44, buf43, buf45, buf46, triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel, 4096, stream=stream0)
            buf48 = reinterpret_tensor(buf23, (512, 256), (256, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg39_1, buf48, 131072, stream=stream0)
            del arg39_1
            buf49 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf48, (256, 512), (1, 256), 0), out=buf49)
            buf50 = reinterpret_tensor(buf5, (s13, 128, 1024), (131072, 1, 128), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1, x_flat_1, v_t_x_1], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17.run(buf44, buf43, buf45, buf46, buf50, triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel, stream=stream0)
            buf51 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, v_3, transpose_1, x_flat_1, v_t_x_1], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 12, 128), (3200, 1, 12), 1536), buf50, out=buf51)
            buf52 = reinterpret_tensor(buf50, (s13, 128, 1024), (131072, 1024, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [split_1, u_3, mixed_1], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 128, 12), (3200, 12, 1), 0), buf51, out=buf52)
            del buf51
            buf57 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_norm_1, split_1, x_flat_1, out_2, view_9, shift_3, out_3, x_4, x_5], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18.run(buf44, buf43, buf45, buf46, buf52, buf31, buf57, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel, 1024, stream=stream0)
            del buf44
            del buf52
            buf55 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_19.run(arg34_1, buf55, 16384, stream=stream0)
            del arg34_1
            buf56 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg35_1, buf56, 128, stream=stream0)
            del arg35_1
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf58 = extern_kernels.convolution(buf57, buf55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf58, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf55
            del buf57
            buf59 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5, add_5, h_2], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_20_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_20.run(buf59, buf56, buf41, triton_poi_fused__to_copy_add_convolution_silu_20_xnumel, stream=stream0)
            del buf41
            del buf56
            buf60 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_21.run(arg36_1, buf60, 294912, stream=stream0)
            del arg36_1
            buf61 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg37_1, buf61, 256, stream=stream0)
            del arg37_1
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            buf62 = extern_kernels.convolution(buf59, buf60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf62, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf60
            buf63 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            triton_poi_fused__to_copy_convolution_22_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_22.run(buf63, buf61, triton_poi_fused__to_copy_convolution_22_xnumel, stream=stream0)
            buf64 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [q_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg38_1, buf64, 65536, stream=stream0)
            del arg38_1
            buf65 = empty_strided_cuda((256*s13, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_3, view_10, q, q_1], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf63, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf64, (256, 256), (1, 256), 0), out=buf65)
            buf66 = empty_strided_cuda((s13, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23.run(buf65, buf66, triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel, stream=stream0)
            buf67 = reinterpret_tensor(buf28, (s13, 4, 64, 1), (256, 64, 1, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24.run(buf49, buf67, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel, stream=stream0)
            buf68 = reinterpret_tensor(buf15, (4*s13, 256, 1), (256, 1, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [kv, chunk, q_1, view_11, q_2, matmul, view_12, k_1, transpose_6], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf66, (4*s13, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf67, (4*s13, 64, 1), (64, 1, 0), 0), out=buf68)
            buf69 = reinterpret_tensor(buf68, (s13, 4, 256, 1), (1024, 256, 1, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [matmul, attn_1, matmul_1], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25.run(buf69, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel, stream=stream0)
            buf70 = reinterpret_tensor(buf67, (s13, 4, 1, 64), (256, 64, 64, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26.run(buf49, buf70, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel, stream=stream0)
            buf71 = reinterpret_tensor(buf66, (4*s13, 256, 64), (16384, 64, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [kv, chunk, matmul, attn_1, matmul_1, view_13, v_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf69, (4*s13, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf70, (4*s13, 1, 64), (64, 0, 1), 0), out=buf71)
            buf72 = reinterpret_tensor(buf65, (s13, 256, 4, 64), (65536, 256, 64, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_27_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_27.run(buf71, buf72, triton_poi_fused_clone_transpose_view_27_xnumel, stream=stream0)
            buf73 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg40_1, buf73, 65536, stream=stream0)
            del arg40_1
            buf74 = reinterpret_tensor(buf71, (256*s13, 256), (256, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4, input_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf72, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf73, (256, 256), (1, 256), 0), out=buf74)
            buf75 = reinterpret_tensor(buf74, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28.run(buf75, buf63, arg41_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel, stream=stream0)
            del arg41_1
            buf76 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29.run(arg42_1, buf76, 589824, stream=stream0)
            del arg42_1
            buf77 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg43_1, buf77, 256, stream=stream0)
            del arg43_1
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf78 = extern_kernels.convolution(buf75, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf78, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf79 = buf46; del buf46  # reuse
            buf80 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30.run(buf78, buf77, buf79, buf80, triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel, 2048, stream=stream0)
            buf82 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg17_1, buf82, 65536, stream=stream0)
            del arg17_1
            buf83 = reinterpret_tensor(buf70, (s13, 256), (256, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf82, (256, 256), (1, 256), 0), out=buf83)
            buf84 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            triton_poi_fused__to_copy_addmm_silu_14_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_14.run(buf84, arg18_1, triton_poi_fused__to_copy_addmm_silu_14_xnumel, stream=stream0)
            del arg18_1
            buf85 = empty_strided_cuda((6400, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_31.run(arg19_1, buf85, 1638400, stream=stream0)
            del arg19_1
            buf86 = empty_strided_cuda((6400, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_32.run(arg20_1, buf86, 6400, stream=stream0)
            del arg20_1
            buf87 = empty_strided_cuda((s13, 6400), (6400, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12, input_10, input_11], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf86, buf84, reinterpret_tensor(buf85, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf87)
            del buf85
            del buf86
            buf88 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg47_1, buf88, 131072, stream=stream0)
            del arg47_1
            buf89 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf88, (256, 512), (1, 256), 0), out=buf89)
            del buf88
            buf90 = reinterpret_tensor(buf75, (s13, 256, 256), (65536, 1, 256), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, x_flat_2, v_t_x_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33.run(buf78, buf77, buf79, buf80, buf90, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel, stream=stream0)
            buf91 = empty_strided_cuda((s13, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, v_7, transpose_9, x_flat_2, v_t_x_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 12, 256), (6400, 1, 12), 3072), buf90, out=buf91)
            buf92 = reinterpret_tensor(buf90, (s13, 256, 256), (65536, 256, 1), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_2], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 256, 12), (6400, 12, 1), 0), buf91, out=buf92)
            buf97 = reinterpret_tensor(buf72, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [input_16, transpose_8, out_5, x_6, x_7, x_norm_2, split_2, x_flat_2, out_6, view_19, shift_5, out_7, x_8, x_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34.run(buf78, buf77, buf79, buf80, buf92, buf87, buf97, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel, 256, stream=stream0)
            del buf78
            del buf92
            buf95 = reinterpret_tensor(buf82, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg44_1, buf95, 65536, stream=stream0)
            del arg44_1
            buf96 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg45_1, buf96, 256, stream=stream0)
            del arg45_1
            # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf98 = extern_kernels.convolution(buf97, buf95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf98, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf99 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [x_8, x_9, add_9, h_4], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_35_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_35.run(buf99, buf96, buf63, triton_poi_fused__to_copy_add_convolution_silu_35_xnumel, stream=stream0)
            buf100 = reinterpret_tensor(buf95, (256, 256), (256, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [q_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg46_1, buf100, 65536, stream=stream0)
            del arg46_1
            buf101 = reinterpret_tensor(buf63, (256*s13, 256), (256, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [x_8, x_9, add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf99, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf100, (256, 256), (1, 256), 0), out=buf101)
            buf102 = reinterpret_tensor(buf97, (s13, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [q_4, view_21, q_5, matmul_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23.run(buf101, buf102, triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel, stream=stream0)
            buf103 = reinterpret_tensor(buf84, (s13, 4, 64, 1), (256, 64, 1, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_22, k_3, transpose_14, matmul_2], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24.run(buf89, buf103, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel, stream=stream0)
            buf104 = reinterpret_tensor(buf69, (4*s13, 256, 1), (256, 1, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, q_4, view_21, q_5, matmul_2, view_22, k_3, transpose_14], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf102, (4*s13, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf103, (4*s13, 64, 1), (64, 1, 0), 0), out=buf104)
            buf105 = reinterpret_tensor(buf104, (s13, 4, 256, 1), (1024, 256, 1, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [matmul_2, attn_3, matmul_3], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25.run(buf105, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel, stream=stream0)
            buf106 = reinterpret_tensor(buf103, (s13, 4, 1, 64), (256, 64, 64, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_23, v_9, matmul_3], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26.run(buf89, buf106, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel, stream=stream0)
            buf107 = reinterpret_tensor(buf102, (4*s13, 256, 64), (16384, 64, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, matmul_2, attn_3, matmul_3, view_23, v_9], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf105, (4*s13, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf106, (4*s13, 1, 64), (64, 0, 1), 0), out=buf107)
            buf108 = reinterpret_tensor(buf101, (s13, 256, 4, 64), (65536, 256, 64, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_27_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_27.run(buf107, buf108, triton_poi_fused_clone_transpose_view_27_xnumel, stream=stream0)
            buf109 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg48_1, buf109, 65536, stream=stream0)
            del arg48_1
            buf110 = reinterpret_tensor(buf107, (256*s13, 256), (256, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8, input_18], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf108, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf109, (256, 256), (1, 256), 0), out=buf110)
            buf111 = reinterpret_tensor(buf110, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28.run(buf111, buf99, arg49_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel, stream=stream0)
            del arg49_1
            buf112 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29.run(arg50_1, buf112, 589824, stream=stream0)
            del arg50_1
            buf113 = buf96; del buf96  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg51_1, buf113, 256, stream=stream0)
            del arg51_1
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf114 = extern_kernels.convolution(buf111, buf112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf114, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf112
            buf115 = buf80; del buf80  # reuse
            buf116 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30.run(buf114, buf113, buf115, buf116, triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel, 2048, stream=stream0)
            buf118 = reinterpret_tensor(buf20, (1024, 256), (256, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg57_1, buf118, 262144, stream=stream0)
            del arg57_1
            buf119 = reinterpret_tensor(buf105, (s13, 1024), (1024, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf118, (256, 1024), (1, 256), 0), out=buf119)
            buf120 = reinterpret_tensor(buf111, (s13, 256, 256), (65536, 1, 256), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3, x_flat_3, v_t_x_3], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33.run(buf114, buf113, buf115, buf116, buf120, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel, stream=stream0)
            buf121 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, v_11, transpose_17, x_flat_3, v_t_x_3], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 12, 256), (6400, 1, 12), 3072), buf120, out=buf121)
            buf122 = reinterpret_tensor(buf120, (s13, 256, 256), (65536, 256, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [split_3, u_7, mixed_3], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 256, 12), (6400, 12, 1), 0), buf121, out=buf122)
            del buf121
            buf127 = reinterpret_tensor(buf108, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [input_18, transpose_16, out_9, x_10, x_11, x_norm_3, split_3, x_flat_3, out_10, view_29, shift_7, out_11, x_12, x_13], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34.run(buf114, buf113, buf115, buf116, buf122, buf87, buf127, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel, 256, stream=stream0)
            del buf114
            del buf122
            buf125 = reinterpret_tensor(buf109, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg52_1, buf125, 65536, stream=stream0)
            del arg52_1
            buf126 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg53_1, buf126, 256, stream=stream0)
            del arg53_1
            # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf128 = extern_kernels.convolution(buf127, buf125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf128, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf127
            buf129 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13, add_13, h_5], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_35_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_35.run(buf129, buf126, buf99, triton_poi_fused__to_copy_add_convolution_silu_35_xnumel, stream=stream0)
            del buf126
            del buf99
            buf130 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_36.run(arg54_1, buf130, 1179648, stream=stream0)
            del arg54_1
            buf131 = empty_strided_cuda((512, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_37.run(arg55_1, buf131, 512, stream=stream0)
            del arg55_1
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            buf132 = extern_kernels.convolution(buf129, buf130, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf132, (s13, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf130
            buf133 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            triton_poi_fused__to_copy_convolution_38_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_38.run(buf133, buf131, triton_poi_fused__to_copy_convolution_38_xnumel, stream=stream0)
            buf134 = reinterpret_tensor(buf118, (512, 512), (512, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [q_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg56_1, buf134, 262144, stream=stream0)
            del arg56_1
            buf135 = empty_strided_cuda((64*s13, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_6, view_30, q_6, q_7], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf133, (64*s13, 512), (512, 1), 0), reinterpret_tensor(buf134, (512, 512), (1, 512), 0), out=buf135)
            buf136 = empty_strided_cuda((s13, 8, 64, 64), (32768, 4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_39_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_39.run(buf135, buf136, triton_poi_fused__unsafe_view_clone_expand_transpose_view_39_xnumel, stream=stream0)
            buf137 = reinterpret_tensor(buf89, (s13, 8, 64, 1), (512, 64, 1, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40.run(buf119, buf137, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40_xnumel, stream=stream0)
            buf138 = reinterpret_tensor(buf19, (8*s13, 64, 1), (64, 1, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, q_7, view_31, q_8, matmul_4, view_32, k_5, transpose_22], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf136, (8*s13, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf137, (8*s13, 64, 1), (64, 1, 0), 0), out=buf138)
            buf139 = reinterpret_tensor(buf138, (s13, 8, 64, 1), (512, 64, 1, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [matmul_4, attn_5, matmul_5], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41.run(buf139, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41_xnumel, stream=stream0)
            buf140 = reinterpret_tensor(buf137, (s13, 8, 1, 64), (512, 64, 64, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42.run(buf119, buf140, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42_xnumel, stream=stream0)
            del buf119
            buf141 = reinterpret_tensor(buf136, (8*s13, 64, 64), (4096, 64, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, matmul_4, attn_5, matmul_5, view_33, v_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf139, (8*s13, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf140, (8*s13, 1, 64), (64, 0, 1), 0), out=buf141)
            del buf139
            del buf140
            buf142 = reinterpret_tensor(buf135, (s13, 64, 8, 64), (32768, 512, 64, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_43_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_43.run(buf141, buf142, triton_poi_fused_clone_transpose_view_43_xnumel, stream=stream0)
            buf143 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg58_1, buf143, 262144, stream=stream0)
            del arg58_1
            buf144 = reinterpret_tensor(buf141, (64*s13, 512), (512, 1), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12, input_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf142, (64*s13, 512), (512, 1), 0), reinterpret_tensor(buf143, (512, 512), (1, 512), 0), out=buf144)
            del buf142
            del buf143
            buf145 = reinterpret_tensor(buf144, (s13, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44.run(buf145, buf133, arg59_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44_xnumel, stream=stream0)
            del arg59_1
            buf146 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_45.run(arg60_1, buf146, 2359296, stream=stream0)
            del arg60_1
            buf147 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_37.run(arg61_1, buf147, 512, stream=stream0)
            del arg61_1
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf148 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf148, (s13, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf145
            del buf146
            buf149 = buf116; del buf116  # reuse
            buf150 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46.run(buf148, buf147, buf149, buf150, triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46_xnumel, 1024, stream=stream0)
            buf152 = reinterpret_tensor(buf125, (256, 256), (256, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg21_1, buf152, 65536, stream=stream0)
            del arg21_1
            buf153 = reinterpret_tensor(buf106, (s13, 256), (256, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf152, (256, 256), (1, 256), 0), out=buf153)
            del buf152
            buf154 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            triton_poi_fused__to_copy_addmm_silu_14_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_14.run(buf154, arg22_1, triton_poi_fused__to_copy_addmm_silu_14_xnumel, stream=stream0)
            del arg22_1
            buf155 = empty_strided_cuda((12800, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_47.run(arg23_1, buf155, 3276800, stream=stream0)
            del arg23_1
            buf156 = empty_strided_cuda((12800, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_48.run(arg24_1, buf156, 12800, stream=stream0)
            del arg24_1
            buf157 = empty_strided_cuda((s13, 12800), (12800, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15, input_13, input_14], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.addmm(buf156, buf154, reinterpret_tensor(buf155, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf157)
            del buf155
            del buf156
            buf158 = empty_strided_cuda((1024, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg65_1, buf158, 262144, stream=stream0)
            del arg65_1
            buf159 = empty_strided_cuda((s13, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf158, (256, 1024), (1, 256), 0), out=buf159)
            buf160 = empty_strided_cuda((s13, 512, 64), (32768, 1, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, x_flat_4, v_t_x_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49.run(buf148, buf147, buf149, buf150, buf160, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49_xnumel, stream=stream0)
            buf161 = empty_strided_cuda((s13, 12, 64), (768, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, v_15, transpose_25, x_flat_4, v_t_x_4], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf157, (s13, 12, 512), (12800, 1, 12), 6144), buf160, out=buf161)
            buf162 = reinterpret_tensor(buf160, (s13, 512, 64), (32768, 64, 1), 0); del buf160  # reuse
            # Topologically Sorted Source Nodes: [split_4, u_9, mixed_4], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf157, (s13, 512, 12), (12800, 12, 1), 0), buf161, out=buf162)
            buf167 = empty_strided_cuda((s13, 512, 8, 8), (32768, 1, 4096, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, transpose_24, out_13, x_14, x_15, x_norm_4, split_4, x_flat_4, out_14, view_39, shift_9, out_15, x_16, x_17], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50_ynumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50.run(buf148, buf147, buf149, buf150, buf162, buf157, buf167, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50_ynumel, 64, stream=stream0)
            del buf148
            del buf162
            buf165 = reinterpret_tensor(buf158, (512, 512, 1, 1), (512, 1, 512, 512), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg62_1, buf165, 262144, stream=stream0)
            del arg62_1
            buf166 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_37.run(arg63_1, buf166, 512, stream=stream0)
            del arg63_1
            # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf168 = extern_kernels.convolution(buf167, buf165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf168, (s13, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf169 = buf168; del buf168  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, add_17, h_7], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_51_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_51.run(buf169, buf166, buf133, triton_poi_fused__to_copy_add_convolution_silu_51_xnumel, stream=stream0)
            buf170 = reinterpret_tensor(buf165, (512, 512), (512, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [q_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg64_1, buf170, 262144, stream=stream0)
            del arg64_1
            buf171 = reinterpret_tensor(buf133, (64*s13, 512), (512, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [x_16, x_17, add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf169, (64*s13, 512), (512, 1), 0), reinterpret_tensor(buf170, (512, 512), (1, 512), 0), out=buf171)
            buf172 = reinterpret_tensor(buf167, (s13, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [q_10, view_41, q_11, matmul_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_39_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_39.run(buf171, buf172, triton_poi_fused__unsafe_view_clone_expand_transpose_view_39_xnumel, stream=stream0)
            buf173 = empty_strided_cuda((s13, 8, 64, 1), (512, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_42, k_7, transpose_30, matmul_6], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40.run(buf159, buf173, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_40_xnumel, stream=stream0)
            buf174 = empty_strided_cuda((8*s13, 64, 1), (64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, q_10, view_41, q_11, matmul_6, view_42, k_7, transpose_30], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf172, (8*s13, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf173, (8*s13, 64, 1), (64, 1, 0), 0), out=buf174)
            buf175 = reinterpret_tensor(buf174, (s13, 8, 64, 1), (512, 64, 1, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [matmul_6, attn_7, matmul_7], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41.run(buf175, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_41_xnumel, stream=stream0)
            buf176 = reinterpret_tensor(buf173, (s13, 8, 1, 64), (512, 64, 64, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_43, v_17, matmul_7], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42.run(buf159, buf176, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_42_xnumel, stream=stream0)
            buf177 = reinterpret_tensor(buf172, (8*s13, 64, 64), (4096, 64, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, matmul_6, attn_7, matmul_7, view_43, v_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf175, (8*s13, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf176, (8*s13, 1, 64), (64, 0, 1), 0), out=buf177)
            del buf175
            buf178 = reinterpret_tensor(buf171, (s13, 64, 8, 64), (32768, 512, 64, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_43_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_43.run(buf177, buf178, triton_poi_fused_clone_transpose_view_43_xnumel, stream=stream0)
            buf179 = buf170; del buf170  # reuse
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg66_1, buf179, 262144, stream=stream0)
            del arg66_1
            buf180 = reinterpret_tensor(buf177, (64*s13, 512), (512, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16, input_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf178, (64*s13, 512), (512, 1), 0), reinterpret_tensor(buf179, (512, 512), (1, 512), 0), out=buf180)
            buf181 = reinterpret_tensor(buf180, (s13, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf180  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44.run(buf181, buf169, arg67_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_44_xnumel, stream=stream0)
            del arg67_1
            buf182 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_45.run(arg68_1, buf182, 2359296, stream=stream0)
            del arg68_1
            buf183 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_37.run(arg69_1, buf183, 512, stream=stream0)
            del arg69_1
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf184 = extern_kernels.convolution(buf181, buf182, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf184, (s13, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf182
            buf185 = buf150; del buf150  # reuse
            buf186 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46.run(buf184, buf183, buf185, buf186, triton_per_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_46_xnumel, 1024, stream=stream0)
            buf188 = reinterpret_tensor(buf181, (s13, 512, 64), (32768, 1, 512), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5, x_flat_5, v_t_x_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49.run(buf184, buf183, buf185, buf186, buf188, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_49_xnumel, stream=stream0)
            buf189 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, v_19, transpose_33, x_flat_5, v_t_x_5], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf157, (s13, 12, 512), (12800, 1, 12), 6144), buf188, out=buf189)
            buf190 = reinterpret_tensor(buf188, (s13, 512, 64), (32768, 64, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [split_5, u_11, mixed_5], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf157, (s13, 512, 12), (12800, 12, 1), 0), buf189, out=buf190)
            del buf189
            buf195 = reinterpret_tensor(buf178, (s13, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [input_22, transpose_32, out_17, x_18, x_19, x_norm_5, split_5, x_flat_5, out_18, view_49, shift_11, out_19, x_20, x_21], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50_ynumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50.run(buf184, buf183, buf185, buf186, buf190, buf157, buf195, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_50_ynumel, 64, stream=stream0)
            del buf157
            buf193 = reinterpret_tensor(buf179, (512, 512, 1, 1), (512, 1, 512, 512), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg70_1, buf193, 262144, stream=stream0)
            del arg70_1
            buf194 = buf183; del buf183  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_37.run(arg71_1, buf194, 512, stream=stream0)
            del arg71_1
            # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf196 = extern_kernels.convolution(buf195, buf193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf196, (s13, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf200 = reinterpret_tensor(buf195, (s13, 64, 512), (32768, 512, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.native_layer_norm]
            triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_52_xnumel = 64*s13
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_52.run(buf196, buf194, buf169, arg72_1, arg73_1, buf200, triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_52_xnumel, 512, stream=stream0)
            del arg72_1
            del arg73_1
            buf201 = empty_strided_cuda((1536, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_53.run(arg74_1, buf201, 786432, stream=stream0)
            del arg74_1
            buf202 = empty_strided_cuda((64*s13, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf200, (64*s13, 512), (512, 1), 0), reinterpret_tensor(buf201, (512, 1536), (1, 512), 0), out=buf202)
            del buf201
            buf203 = reinterpret_tensor(buf200, (s13, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.expand, aten.clone]
            triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_54_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_54.run(buf202, arg75_1, buf203, triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_54_xnumel, stream=stream0)
            buf204 = reinterpret_tensor(buf190, (s13, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__to_copy_addmm_clone_expand_permute_select_transpose_view_55_ynumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_clone_expand_permute_select_transpose_view_55.run(buf202, arg75_1, buf204, triton_poi_fused__to_copy_addmm_clone_expand_permute_select_transpose_view_55_ynumel, 64, stream=stream0)
            buf205 = reinterpret_tensor(buf184, (8*s13, 64, 64), (4096, 64, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8, k_8, transpose_35], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.expand, aten.clone, aten._unsafe_view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf203, (8*s13, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf204, (8*s13, 64, 64), (4096, 64, 1), 0), out=buf205)
            buf208 = reinterpret_tensor(buf205, (s13, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf205  # reuse
            # Topologically Sorted Source Nodes: [matmul_8, attn_9, out_20], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_per_fused__softmax__to_copy_amax_mul_sub_view_56_xnumel = 512*s13
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_amax_mul_sub_view_56.run(buf208, triton_per_fused__softmax__to_copy_amax_mul_sub_view_56_xnumel, 64, stream=stream0)
            buf209 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.expand, aten.clone]
            triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_57_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_57.run(buf202, arg75_1, buf209, triton_poi_fused__to_copy_addmm_clone_expand_permute_select_view_57_xnumel, stream=stream0)
            del arg75_1
            del buf202
            buf210 = reinterpret_tensor(buf203, (8*s13, 64, 64), (4096, 64, 1), 0); del buf203  # reuse
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, matmul_8, attn_9, out_20, v_20], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.mul, aten.sub, aten._softmax, aten.expand, aten.select, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf208, (8*s13, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf209, (8*s13, 64, 64), (4096, 64, 1), 0), out=buf210)
            del buf208
            buf211 = reinterpret_tensor(buf209, (s13, 64, 8, 64), (32768, 512, 64, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_43_xnumel = 32768*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_43.run(buf210, buf211, triton_poi_fused_clone_transpose_view_43_xnumel, stream=stream0)
            buf212 = reinterpret_tensor(buf193, (512, 512), (512, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg76_1, buf212, 262144, stream=stream0)
            del arg76_1
            buf213 = reinterpret_tensor(buf210, (64*s13, 512), (512, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21, out_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf211, (64*s13, 512), (512, 1), 0), reinterpret_tensor(buf212, (512, 512), (1, 512), 0), out=buf213)
            del buf212
            buf220 = reinterpret_tensor(buf211, (s13, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, add_21, h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.addmm, aten.native_layer_norm]
            triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_58_xnumel = 64*s13
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_58.run(buf196, buf194, buf169, buf213, arg77_1, arg78_1, arg79_1, buf220, triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_58_xnumel, 512, stream=stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del buf169
            del buf194
            del buf196
            del buf213
            buf218 = empty_strided_cuda((512, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg87_1, buf218, 131072, stream=stream0)
            del arg87_1
            buf219 = reinterpret_tensor(buf176, (s13, 512), (512, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf218, (256, 512), (1, 256), 0), out=buf219)
            buf221 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_native_layer_norm_transpose_view_59.run(arg80_1, buf221, 2097152, stream=stream0)
            del arg80_1
            buf222 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg81_1, buf222, 256, stream=stream0)
            del arg81_1
            # Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution]
            buf223 = extern_kernels.convolution(buf220, buf221, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf223, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf220
            del buf221
            buf224 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg82_1, buf224, 65536, stream=stream0)
            del arg82_1
            buf225 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf224, (256, 256), (1, 256), 0), out=buf225)
            buf226 = buf225; del buf225  # reuse
            # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            triton_poi_fused__to_copy_addmm_silu_14_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_14.run(buf226, arg83_1, triton_poi_fused__to_copy_addmm_silu_14_xnumel, stream=stream0)
            del arg83_1
            buf227 = buf224; del buf224  # reuse
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg84_1, buf227, 65536, stream=stream0)
            del arg84_1
            buf228 = empty_strided_cuda((s13, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.mm(buf226, reinterpret_tensor(buf227, (256, 256), (1, 256), 0), out=buf228)
            del buf226
            buf229 = empty_strided_cuda((s13, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_24, transpose_37, out_25, h_9, input_26, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten.native_layer_norm, aten.transpose, aten.view, aten._to_copy, aten.convolution, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
            triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_60_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_60.run(buf223, buf222, buf129, buf228, arg85_1, buf229, triton_poi_fused__to_copy_add_addmm_convolution_mul_native_layer_norm_sigmoid_transpose_unsqueeze_view_60_ynumel, 256, stream=stream0)
            del arg85_1
            ps0 = 256*s13
            buf230 = reinterpret_tensor(buf223, (256*s13, 256), (1, 256*s13), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.t, aten.mm]
            triton_poi_fused__to_copy_mm_t_transpose_view_61_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_mm_t_transpose_view_61.run(buf229, buf230, ps0, triton_poi_fused__to_copy_mm_t_transpose_view_61_xnumel, stream=stream0)
            buf231 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [q_14], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg86_1, buf231, 65536, stream=stream0)
            del arg86_1
            buf232 = reinterpret_tensor(buf129, (256*s13, 256), (256, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [view_52, q_13, q_14], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf230, reinterpret_tensor(buf231, (256, 256), (1, 256), 0), out=buf232)
            buf233 = reinterpret_tensor(buf230, (s13, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [q_14, view_53, q_15, matmul_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23.run(buf232, buf233, triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel, stream=stream0)
            buf234 = reinterpret_tensor(buf228, (s13, 4, 64, 1), (256, 64, 1, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_54, k_10, transpose_42, matmul_10], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24.run(buf219, buf234, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel, stream=stream0)
            buf235 = reinterpret_tensor(buf159, (4*s13, 256, 1), (256, 1, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, q_14, view_53, q_15, matmul_10, view_54, k_10, transpose_42], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf233, (4*s13, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf234, (4*s13, 64, 1), (64, 1, 0), 0), out=buf235)
            buf236 = reinterpret_tensor(buf235, (s13, 4, 256, 1), (1024, 256, 1, 1), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [matmul_10, attn_11, matmul_11], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25.run(buf236, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel, stream=stream0)
            buf237 = reinterpret_tensor(buf234, (s13, 4, 1, 64), (256, 64, 64, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_55, v_22, matmul_11], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26.run(buf219, buf237, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel, stream=stream0)
            buf238 = reinterpret_tensor(buf233, (4*s13, 256, 64), (16384, 64, 1), 0); del buf233  # reuse
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, matmul_10, attn_11, matmul_11, view_55, v_22], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf236, (4*s13, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf237, (4*s13, 1, 64), (64, 0, 1), 0), out=buf238)
            buf239 = reinterpret_tensor(buf232, (s13, 256, 4, 64), (65536, 256, 64, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_27_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_27.run(buf238, buf239, triton_poi_fused_clone_transpose_view_27_xnumel, stream=stream0)
            buf240 = buf231; del buf231  # reuse
            # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg88_1, buf240, 65536, stream=stream0)
            del arg88_1
            buf241 = reinterpret_tensor(buf238, (256*s13, 256), (256, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26, input_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf239, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf240, (256, 256), (1, 256), 0), out=buf241)
            buf242 = reinterpret_tensor(buf241, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_62_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_62.run(buf242, buf229, arg89_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_62_ynumel, 256, stream=stream0)
            del arg89_1
            buf243 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29.run(arg90_1, buf243, 589824, stream=stream0)
            del arg90_1
            buf244 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg91_1, buf244, 256, stream=stream0)
            del arg91_1
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf245 = extern_kernels.convolution(buf242, buf243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf245, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf246 = buf186; del buf186  # reuse
            buf247 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30.run(buf245, buf244, buf246, buf247, triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel, 2048, stream=stream0)
            buf249 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg95_1, buf249, 131072, stream=stream0)
            del arg95_1
            buf250 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf249, (256, 512), (1, 256), 0), out=buf250)
            buf251 = reinterpret_tensor(buf242, (s13, 256, 256), (65536, 1, 256), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, x_flat_7, v_t_x_6], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33.run(buf245, buf244, buf246, buf247, buf251, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel, stream=stream0)
            buf252 = empty_strided_cuda((s13, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, v_24, transpose_45, x_flat_7, v_t_x_6], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 12, 256), (6400, 1, 12), 3072), buf251, out=buf252)
            buf253 = reinterpret_tensor(buf251, (s13, 256, 256), (65536, 256, 1), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [split_6, u_13, mixed_6], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 256, 12), (6400, 12, 1), 0), buf252, out=buf253)
            buf258 = reinterpret_tensor(buf239, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf239  # reuse
            # Topologically Sorted Source Nodes: [input_28, transpose_44, out_27, x_22, x_23, x_norm_7, split_6, x_flat_7, out_28, view_61, shift_13, out_29, x_24, x_25], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34.run(buf245, buf244, buf246, buf247, buf253, buf87, buf258, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel, 256, stream=stream0)
            del buf245
            del buf253
            buf256 = reinterpret_tensor(buf240, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [x_24, x_25], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg92_1, buf256, 65536, stream=stream0)
            del arg92_1
            buf257 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [x_24, x_25], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg93_1, buf257, 256, stream=stream0)
            del arg93_1
            # Topologically Sorted Source Nodes: [x_24, x_25], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf259 = extern_kernels.convolution(buf258, buf256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf259, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf260 = buf259; del buf259  # reuse
            # Topologically Sorted Source Nodes: [x_24, x_25, add_27, h_11], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_63_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_63.run(buf260, buf257, buf229, triton_poi_fused__to_copy_add_convolution_silu_63_ynumel, 256, stream=stream0)
            buf261 = reinterpret_tensor(buf256, (256, 256), (256, 1), 0); del buf256  # reuse
            # Topologically Sorted Source Nodes: [q_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg94_1, buf261, 65536, stream=stream0)
            del arg94_1
            buf262 = reinterpret_tensor(buf229, (256*s13, 256), (256, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [x_24, x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf260, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf261, (256, 256), (1, 256), 0), out=buf262)
            buf263 = reinterpret_tensor(buf258, (s13, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [q_17, view_63, q_18, matmul_12], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23.run(buf262, buf263, triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel, stream=stream0)
            buf264 = reinterpret_tensor(buf237, (s13, 4, 64, 1), (256, 64, 1, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_64, k_12, transpose_50, matmul_12], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24.run(buf250, buf264, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel, stream=stream0)
            buf265 = reinterpret_tensor(buf236, (4*s13, 256, 1), (256, 1, 1), 0); del buf236  # reuse
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, q_17, view_63, q_18, matmul_12, view_64, k_12, transpose_50], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf263, (4*s13, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf264, (4*s13, 64, 1), (64, 1, 0), 0), out=buf265)
            buf266 = reinterpret_tensor(buf265, (s13, 4, 256, 1), (1024, 256, 1, 1), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [matmul_12, attn_13, matmul_13], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25.run(buf266, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel, stream=stream0)
            buf267 = reinterpret_tensor(buf264, (s13, 4, 1, 64), (256, 64, 64, 1), 0); del buf264  # reuse
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_65, v_26, matmul_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26.run(buf250, buf267, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel, stream=stream0)
            buf268 = reinterpret_tensor(buf263, (4*s13, 256, 64), (16384, 64, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, matmul_12, attn_13, matmul_13, view_65, v_26], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf266, (4*s13, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf267, (4*s13, 1, 64), (64, 0, 1), 0), out=buf268)
            buf269 = reinterpret_tensor(buf262, (s13, 256, 4, 64), (65536, 256, 64, 1), 0); del buf262  # reuse
            # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_27_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_27.run(buf268, buf269, triton_poi_fused_clone_transpose_view_27_xnumel, stream=stream0)
            buf270 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg96_1, buf270, 65536, stream=stream0)
            del arg96_1
            buf271 = reinterpret_tensor(buf268, (256*s13, 256), (256, 1), 0); del buf268  # reuse
            # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30, input_30], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf269, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf270, (256, 256), (1, 256), 0), out=buf271)
            buf272 = reinterpret_tensor(buf271, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28.run(buf272, buf260, arg97_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel, stream=stream0)
            del arg97_1
            buf273 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29.run(arg98_1, buf273, 589824, stream=stream0)
            del arg98_1
            buf274 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg99_1, buf274, 256, stream=stream0)
            del arg99_1
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf275 = extern_kernels.convolution(buf272, buf273, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf275, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf276 = buf247; del buf247  # reuse
            buf277 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30.run(buf275, buf274, buf276, buf277, triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel, 2048, stream=stream0)
            buf279 = buf249; del buf249  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg103_1, buf279, 131072, stream=stream0)
            del arg103_1
            buf280 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf279, (256, 512), (1, 256), 0), out=buf280)
            buf281 = reinterpret_tensor(buf272, (s13, 256, 256), (65536, 1, 256), 0); del buf272  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8, x_flat_8, v_t_x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33.run(buf275, buf274, buf276, buf277, buf281, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel, stream=stream0)
            buf282 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, v_28, transpose_53, x_flat_8, v_t_x_7], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 12, 256), (6400, 1, 12), 3072), buf281, out=buf282)
            buf283 = reinterpret_tensor(buf281, (s13, 256, 256), (65536, 256, 1), 0); del buf281  # reuse
            # Topologically Sorted Source Nodes: [split_7, u_15, mixed_7], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 256, 12), (6400, 12, 1), 0), buf282, out=buf283)
            buf288 = reinterpret_tensor(buf269, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf269  # reuse
            # Topologically Sorted Source Nodes: [input_30, transpose_52, out_31, x_26, x_27, x_norm_8, split_7, x_flat_8, out_32, view_71, shift_15, out_33, x_28, x_29], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34.run(buf275, buf274, buf276, buf277, buf283, buf87, buf288, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel, 256, stream=stream0)
            del buf275
            del buf283
            buf286 = reinterpret_tensor(buf270, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf270  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg100_1, buf286, 65536, stream=stream0)
            del arg100_1
            buf287 = buf274; del buf274  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg101_1, buf287, 256, stream=stream0)
            del arg101_1
            # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf289 = extern_kernels.convolution(buf288, buf286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf289, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf290 = buf289; del buf289  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, add_31, h_12], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_35_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_35.run(buf290, buf287, buf260, triton_poi_fused__to_copy_add_convolution_silu_35_xnumel, stream=stream0)
            buf291 = reinterpret_tensor(buf286, (256, 256), (256, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [q_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg102_1, buf291, 65536, stream=stream0)
            del arg102_1
            buf292 = reinterpret_tensor(buf260, (256*s13, 256), (256, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf290, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf291, (256, 256), (1, 256), 0), out=buf292)
            buf293 = reinterpret_tensor(buf288, (s13, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [q_20, view_73, q_21, matmul_14], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23.run(buf292, buf293, triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel, stream=stream0)
            buf294 = reinterpret_tensor(buf267, (s13, 4, 64, 1), (256, 64, 1, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_74, k_14, transpose_58, matmul_14], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24.run(buf280, buf294, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel, stream=stream0)
            buf295 = reinterpret_tensor(buf266, (4*s13, 256, 1), (256, 1, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, q_20, view_73, q_21, matmul_14, view_74, k_14, transpose_58], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf293, (4*s13, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf294, (4*s13, 64, 1), (64, 1, 0), 0), out=buf295)
            buf296 = reinterpret_tensor(buf295, (s13, 4, 256, 1), (1024, 256, 1, 1), 0); del buf295  # reuse
            # Topologically Sorted Source Nodes: [matmul_14, attn_15, matmul_15], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25.run(buf296, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel, stream=stream0)
            buf297 = reinterpret_tensor(buf294, (s13, 4, 1, 64), (256, 64, 64, 1), 0); del buf294  # reuse
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_75, v_30, matmul_15], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26.run(buf280, buf297, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel, stream=stream0)
            buf298 = reinterpret_tensor(buf293, (4*s13, 256, 64), (16384, 64, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, matmul_14, attn_15, matmul_15, view_75, v_30], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf296, (4*s13, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf297, (4*s13, 1, 64), (64, 0, 1), 0), out=buf298)
            buf299 = reinterpret_tensor(buf292, (s13, 256, 4, 64), (65536, 256, 64, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_27_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_27.run(buf298, buf299, triton_poi_fused_clone_transpose_view_27_xnumel, stream=stream0)
            buf300 = buf291; del buf291  # reuse
            # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg104_1, buf300, 65536, stream=stream0)
            del arg104_1
            buf301 = reinterpret_tensor(buf298, (256*s13, 256), (256, 1), 0); del buf298  # reuse
            # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34, input_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf299, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf300, (256, 256), (1, 256), 0), out=buf301)
            buf302 = reinterpret_tensor(buf301, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28.run(buf302, buf290, arg105_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel, stream=stream0)
            del arg105_1
            buf303 = buf273; del buf273  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29.run(arg106_1, buf303, 589824, stream=stream0)
            del arg106_1
            buf304 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg107_1, buf304, 256, stream=stream0)
            del arg107_1
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf305 = extern_kernels.convolution(buf302, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf305, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf306 = buf277; del buf277  # reuse
            buf307 = buf276; del buf276  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30.run(buf305, buf304, buf306, buf307, triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel, 2048, stream=stream0)
            buf309 = buf279; del buf279  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg111_1, buf309, 131072, stream=stream0)
            del arg111_1
            buf310 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf309, (256, 512), (1, 256), 0), out=buf310)
            buf311 = reinterpret_tensor(buf302, (s13, 256, 256), (65536, 1, 256), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9, x_flat_9, v_t_x_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33.run(buf305, buf304, buf306, buf307, buf311, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel, stream=stream0)
            buf312 = buf282; del buf282  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, v_32, transpose_61, x_flat_9, v_t_x_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 12, 256), (6400, 1, 12), 3072), buf311, out=buf312)
            buf313 = reinterpret_tensor(buf311, (s13, 256, 256), (65536, 256, 1), 0); del buf311  # reuse
            # Topologically Sorted Source Nodes: [split_8, u_17, mixed_8], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 256, 12), (6400, 12, 1), 0), buf312, out=buf313)
            buf318 = reinterpret_tensor(buf299, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf299  # reuse
            # Topologically Sorted Source Nodes: [input_32, transpose_60, out_35, x_30, x_31, x_norm_9, split_8, x_flat_9, out_36, view_81, shift_17, out_37, x_32, x_33], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34.run(buf305, buf304, buf306, buf307, buf313, buf87, buf318, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel, 256, stream=stream0)
            del buf305
            del buf313
            buf316 = reinterpret_tensor(buf300, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf300  # reuse
            # Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg108_1, buf316, 65536, stream=stream0)
            del arg108_1
            buf317 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg109_1, buf317, 256, stream=stream0)
            del arg109_1
            # Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf319 = extern_kernels.convolution(buf318, buf316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf319, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf320 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [x_32, x_33, add_35, h_13], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_35_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_35.run(buf320, buf317, buf290, triton_poi_fused__to_copy_add_convolution_silu_35_xnumel, stream=stream0)
            buf321 = reinterpret_tensor(buf316, (256, 256), (256, 1), 0); del buf316  # reuse
            # Topologically Sorted Source Nodes: [q_23], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg110_1, buf321, 65536, stream=stream0)
            del arg110_1
            buf322 = reinterpret_tensor(buf290, (256*s13, 256), (256, 1), 0); del buf290  # reuse
            # Topologically Sorted Source Nodes: [x_32, x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf320, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf321, (256, 256), (1, 256), 0), out=buf322)
            buf323 = reinterpret_tensor(buf318, (s13, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf318  # reuse
            # Topologically Sorted Source Nodes: [q_23, view_83, q_24, matmul_16], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23.run(buf322, buf323, triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel, stream=stream0)
            buf324 = reinterpret_tensor(buf297, (s13, 4, 64, 1), (256, 64, 1, 1), 0); del buf297  # reuse
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_84, k_16, transpose_66, matmul_16], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24.run(buf310, buf324, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel, stream=stream0)
            buf325 = reinterpret_tensor(buf296, (4*s13, 256, 1), (256, 1, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, q_23, view_83, q_24, matmul_16, view_84, k_16, transpose_66], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf323, (4*s13, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf324, (4*s13, 64, 1), (64, 1, 0), 0), out=buf325)
            buf326 = reinterpret_tensor(buf325, (s13, 4, 256, 1), (1024, 256, 1, 1), 0); del buf325  # reuse
            # Topologically Sorted Source Nodes: [matmul_16, attn_17, matmul_17], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25.run(buf326, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel, stream=stream0)
            buf327 = reinterpret_tensor(buf324, (s13, 4, 1, 64), (256, 64, 64, 1), 0); del buf324  # reuse
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_85, v_34, matmul_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26.run(buf310, buf327, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel, stream=stream0)
            buf328 = reinterpret_tensor(buf323, (4*s13, 256, 64), (16384, 64, 1), 0); del buf323  # reuse
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, matmul_16, attn_17, matmul_17, view_85, v_34], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf326, (4*s13, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf327, (4*s13, 1, 64), (64, 0, 1), 0), out=buf328)
            buf329 = reinterpret_tensor(buf322, (s13, 256, 4, 64), (65536, 256, 64, 1), 0); del buf322  # reuse
            # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_27_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_27.run(buf328, buf329, triton_poi_fused_clone_transpose_view_27_xnumel, stream=stream0)
            buf330 = buf321; del buf321  # reuse
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg112_1, buf330, 65536, stream=stream0)
            del arg112_1
            buf331 = reinterpret_tensor(buf328, (256*s13, 256), (256, 1), 0); del buf328  # reuse
            # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38, input_34], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf329, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf330, (256, 256), (1, 256), 0), out=buf331)
            buf332 = reinterpret_tensor(buf331, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf331  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28.run(buf332, buf320, arg113_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel, stream=stream0)
            del arg113_1
            buf333 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29.run(arg114_1, buf333, 589824, stream=stream0)
            del arg114_1
            buf334 = buf317; del buf317  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg115_1, buf334, 256, stream=stream0)
            del arg115_1
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf335 = extern_kernels.convolution(buf332, buf333, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf335, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf336 = buf307; del buf307  # reuse
            buf337 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30.run(buf335, buf334, buf336, buf337, triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel, 2048, stream=stream0)
            buf339 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(arg119_1, buf339, 131072, stream=stream0)
            del arg119_1
            buf340 = buf310; del buf310  # reuse
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf339, (256, 512), (1, 256), 0), out=buf340)
            del buf339
            buf341 = reinterpret_tensor(buf332, (s13, 256, 256), (65536, 1, 256), 0); del buf332  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10, x_flat_10, v_t_x_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33.run(buf335, buf334, buf336, buf337, buf341, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel, stream=stream0)
            buf342 = buf312; del buf312  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, v_36, transpose_69, x_flat_10, v_t_x_9], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 12, 256), (6400, 1, 12), 3072), buf341, out=buf342)
            buf343 = reinterpret_tensor(buf341, (s13, 256, 256), (65536, 256, 1), 0); del buf341  # reuse
            # Topologically Sorted Source Nodes: [split_9, u_19, mixed_9], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 256, 12), (6400, 12, 1), 0), buf342, out=buf343)
            buf348 = reinterpret_tensor(buf329, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf329  # reuse
            # Topologically Sorted Source Nodes: [input_34, transpose_68, out_39, x_34, x_35, x_norm_10, split_9, x_flat_10, out_40, view_91, shift_19, out_41, x_36, x_37], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34.run(buf335, buf334, buf336, buf337, buf343, buf87, buf348, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel, 256, stream=stream0)
            del buf335
            del buf343
            buf346 = reinterpret_tensor(buf330, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg116_1, buf346, 65536, stream=stream0)
            del arg116_1
            buf347 = buf334; del buf334  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg117_1, buf347, 256, stream=stream0)
            del arg117_1
            # Topologically Sorted Source Nodes: [x_36, x_37], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf349 = extern_kernels.convolution(buf348, buf346, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf349, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf350 = buf349; del buf349  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, add_39, h_14], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_35_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_35.run(buf350, buf347, buf320, triton_poi_fused__to_copy_add_convolution_silu_35_xnumel, stream=stream0)
            buf351 = reinterpret_tensor(buf346, (256, 256), (256, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [q_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg118_1, buf351, 65536, stream=stream0)
            del arg118_1
            buf352 = reinterpret_tensor(buf320, (256*s13, 256), (256, 1), 0); del buf320  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.view, aten.transpose, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf350, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf351, (256, 256), (1, 256), 0), out=buf352)
            buf353 = reinterpret_tensor(buf348, (s13, 4, 256, 64), (65536, 16384, 64, 1), 0); del buf348  # reuse
            # Topologically Sorted Source Nodes: [q_26, view_93, q_27, matmul_18], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_view_23.run(buf352, buf353, triton_poi_fused__unsafe_view_clone_expand_transpose_view_23_xnumel, stream=stream0)
            buf354 = reinterpret_tensor(buf327, (s13, 4, 64, 1), (256, 64, 1, 1), 0); del buf327  # reuse
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_94, k_18, transpose_74, matmul_18], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24.run(buf340, buf354, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_24_xnumel, stream=stream0)
            buf355 = reinterpret_tensor(buf326, (4*s13, 256, 1), (256, 1, 1), 0); del buf326  # reuse
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, q_26, view_93, q_27, matmul_18, view_94, k_18, transpose_74], Original ATen: [aten.unsqueeze, aten.split, aten._unsafe_view, aten.view, aten.transpose, aten.expand, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf353, (4*s13, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf354, (4*s13, 64, 1), (64, 1, 0), 0), out=buf355)
            buf356 = reinterpret_tensor(buf355, (s13, 4, 256, 1), (1024, 256, 1, 1), 0); del buf355  # reuse
            # Topologically Sorted Source Nodes: [matmul_18, attn_19, matmul_19], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25.run(buf356, triton_poi_fused__softmax__to_copy_amax_mul_sub_view_25_xnumel, stream=stream0)
            buf357 = reinterpret_tensor(buf354, (s13, 4, 1, 64), (256, 64, 64, 1), 0); del buf354  # reuse
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_95, v_38, matmul_19], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.expand, aten.clone]
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26.run(buf340, buf357, triton_poi_fused_clone_expand_split_transpose_unsqueeze_view_26_xnumel, stream=stream0)
            del buf340
            buf358 = reinterpret_tensor(buf353, (4*s13, 256, 64), (16384, 64, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, matmul_18, attn_19, matmul_19, view_95, v_38], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.expand, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf356, (4*s13, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf357, (4*s13, 1, 64), (64, 0, 1), 0), out=buf358)
            del buf356
            buf359 = reinterpret_tensor(buf352, (s13, 256, 4, 64), (65536, 256, 64, 1), 0); del buf352  # reuse
            # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42], Original ATen: [aten.view, aten.transpose, aten.clone]
            triton_poi_fused_clone_transpose_view_27_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_27.run(buf358, buf359, triton_poi_fused_clone_transpose_view_27_xnumel, stream=stream0)
            buf360 = buf351; del buf351  # reuse
            # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg120_1, buf360, 65536, stream=stream0)
            del arg120_1
            buf361 = reinterpret_tensor(buf358, (256*s13, 256), (256, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42, input_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf359, (256*s13, 256), (256, 1), 0), reinterpret_tensor(buf360, (256, 256), (1, 256), 0), out=buf361)
            buf362 = reinterpret_tensor(buf361, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf361  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28.run(buf362, buf350, arg121_1, triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_28_xnumel, stream=stream0)
            del arg121_1
            buf363 = buf333; del buf333  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_transpose_view_29.run(arg122_1, buf363, 589824, stream=stream0)
            del arg122_1
            buf364 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg123_1, buf364, 256, stream=stream0)
            del arg123_1
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution]
            buf365 = extern_kernels.convolution(buf362, buf363, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf365, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf363
            buf366 = buf337; del buf337  # reuse
            buf367 = buf336; del buf336  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30.run(buf365, buf364, buf366, buf367, triton_red_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_30_xnumel, 2048, stream=stream0)
            buf369 = reinterpret_tensor(buf362, (s13, 256, 256), (65536, 1, 256), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11, x_flat_11, v_t_x_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33.run(buf365, buf364, buf366, buf367, buf369, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_transpose_view_33_xnumel, stream=stream0)
            buf370 = buf342; del buf342  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, v_40, transpose_77, x_flat_11, v_t_x_10], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 12, 256), (6400, 1, 12), 3072), buf369, out=buf370)
            buf371 = reinterpret_tensor(buf369, (s13, 256, 256), (65536, 256, 1), 0); del buf369  # reuse
            # Topologically Sorted Source Nodes: [split_10, u_21, mixed_10], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (s13, 256, 12), (6400, 12, 1), 0), buf370, out=buf371)
            del buf370
            buf376 = reinterpret_tensor(buf359, (s13, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf359  # reuse
            # Topologically Sorted Source Nodes: [input_36, transpose_76, out_43, x_38, x_39, x_norm_11, split_10, x_flat_11, out_44, view_101, shift_21, out_45, x_40, x_41], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.silu]
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34.run(buf365, buf364, buf366, buf367, buf371, buf87, buf376, triton_poi_fused__to_copy_add_addmm_clone_convolution_native_group_norm_silu_split_with_sizes_transpose_view_34_ynumel, 256, stream=stream0)
            del buf365
            del buf371
            del buf87
            buf374 = reinterpret_tensor(buf360, (256, 256, 1, 1), (256, 1, 256, 256), 0); del buf360  # reuse
            # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg124_1, buf374, 65536, stream=stream0)
            del arg124_1
            buf375 = buf364; del buf364  # reuse
            # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(arg125_1, buf375, 256, stream=stream0)
            del arg125_1
            # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf377 = extern_kernels.convolution(buf376, buf374, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf377, (s13, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            del buf376
            buf378 = buf377; del buf377  # reuse
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_35_xnumel = 65536*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_35.run(buf378, buf375, buf350, triton_poi_fused__to_copy_add_convolution_silu_35_xnumel, stream=stream0)
            del buf350
            del buf375
            buf379 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_64.run(arg126_1, buf379, 524288, stream=stream0)
            del arg126_1
            buf380 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg127_1, buf380, 128, stream=stream0)
            del arg127_1
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            buf381 = extern_kernels.convolution(buf378, buf379, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf381, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf378
            del buf379
            buf382 = reinterpret_tensor(buf374, (256, 256), (256, 1), 0); del buf374  # reuse
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg128_1, buf382, 65536, stream=stream0)
            del arg128_1
            buf383 = reinterpret_tensor(buf357, (s13, 256), (256, 1), 0); del buf357  # reuse
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.mm(buf25, reinterpret_tensor(buf382, (256, 256), (1, 256), 0), out=buf383)
            del buf25
            del buf382
            buf384 = buf383; del buf383  # reuse
            # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten._to_copy, aten.addmm, aten.silu]
            triton_poi_fused__to_copy_addmm_silu_14_xnumel = 256*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_silu_14.run(buf384, arg129_1, triton_poi_fused__to_copy_addmm_silu_14_xnumel, stream=stream0)
            del arg129_1
            buf385 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_65.run(arg130_1, buf385, 32768, stream=stream0)
            del arg130_1
            buf386 = empty_strided_cuda((s13, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_38, input_39, input_40], Original ATen: [aten._to_copy, aten.addmm, aten.silu, aten.t]
            extern_kernels.mm(buf384, reinterpret_tensor(buf385, (256, 128), (1, 256), 0), out=buf386)
            del buf384
            del buf385
            buf387 = buf381; del buf381  # reuse
            # Topologically Sorted Source Nodes: [x_40, x_41, add_43, h_15, h_16, input_40, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.addmm, aten.sigmoid, aten.unsqueeze, aten.mul]
            triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_66_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_66.run(buf387, buf380, buf59, buf386, arg131_1, triton_poi_fused__to_copy_add_addmm_convolution_mul_sigmoid_silu_unsqueeze_66_xnumel, stream=stream0)
            del arg131_1
            del buf386
            buf388 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_4.run(arg132_1, buf388, 147456, stream=stream0)
            del arg132_1
            buf389 = buf380; del buf380  # reuse
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg133_1, buf389, 128, stream=stream0)
            del arg133_1
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.convolution]
            buf390 = extern_kernels.convolution(buf387, buf388, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf390, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf391 = buf367; del buf367  # reuse
            buf392 = buf366; del buf366  # reuse
            # Topologically Sorted Source Nodes: [x_42, x_norm_12], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5.run(buf390, buf389, buf391, buf392, triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel, 4096, stream=stream0)
            buf394 = reinterpret_tensor(buf59, (s13, 128, 1024), (131072, 1, 128), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [x_42, x_norm_12, x_flat_12, v_t_x_11], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17.run(buf390, buf389, buf391, buf392, buf394, triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel, stream=stream0)
            buf395 = empty_strided_cuda((s13, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, v_42, transpose_78, x_flat_12, v_t_x_11], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 12, 128), (3200, 1, 12), 1536), buf394, out=buf395)
            buf396 = reinterpret_tensor(buf394, (s13, 128, 1024), (131072, 1024, 1), 0); del buf394  # reuse
            # Topologically Sorted Source Nodes: [split_11, u_23, mixed_11], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 128, 12), (3200, 12, 1), 0), buf395, out=buf396)
            buf401 = empty_strided_cuda((s13, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42, x_norm_12, split_11, x_flat_12, out_46, view_106, shift_23, out_47, x_43, x_44], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18.run(buf390, buf389, buf391, buf392, buf396, buf31, buf401, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel, 1024, stream=stream0)
            del buf390
            del buf396
            buf399 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_43, x_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_19.run(arg134_1, buf399, 16384, stream=stream0)
            del arg134_1
            buf400 = buf389; del buf389  # reuse
            # Topologically Sorted Source Nodes: [x_43, x_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg135_1, buf400, 128, stream=stream0)
            del arg135_1
            # Topologically Sorted Source Nodes: [x_43, x_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf402 = extern_kernels.convolution(buf401, buf399, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf402, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf403 = buf402; del buf402  # reuse
            # Topologically Sorted Source Nodes: [x_43, x_44, add_47, h_18], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_20_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_20.run(buf403, buf400, buf387, triton_poi_fused__to_copy_add_convolution_silu_20_xnumel, stream=stream0)
            buf404 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_4.run(arg136_1, buf404, 147456, stream=stream0)
            del arg136_1
            buf405 = buf400; del buf400  # reuse
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg137_1, buf405, 128, stream=stream0)
            del arg137_1
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy, aten.convolution]
            buf406 = extern_kernels.convolution(buf403, buf404, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf406, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf407 = buf392; del buf392  # reuse
            buf408 = buf391; del buf391  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5.run(buf406, buf405, buf407, buf408, triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel, 4096, stream=stream0)
            buf410 = reinterpret_tensor(buf387, (s13, 128, 1024), (131072, 1, 128), 0); del buf387  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13, x_flat_13, v_t_x_12], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17.run(buf406, buf405, buf407, buf408, buf410, triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel, stream=stream0)
            buf411 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, v_44, transpose_79, x_flat_13, v_t_x_12], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 12, 128), (3200, 1, 12), 1536), buf410, out=buf411)
            buf412 = reinterpret_tensor(buf410, (s13, 128, 1024), (131072, 1024, 1), 0); del buf410  # reuse
            # Topologically Sorted Source Nodes: [split_12, u_25, mixed_12], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 128, 12), (3200, 12, 1), 0), buf411, out=buf412)
            buf417 = buf401; del buf401  # reuse
            # Topologically Sorted Source Nodes: [x_45, x_norm_13, split_12, x_flat_13, out_48, view_111, shift_25, out_49, x_46, x_47], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18.run(buf406, buf405, buf407, buf408, buf412, buf31, buf417, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel, 1024, stream=stream0)
            del buf406
            del buf412
            buf415 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_19.run(arg138_1, buf415, 16384, stream=stream0)
            del arg138_1
            buf416 = buf405; del buf405  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg139_1, buf416, 128, stream=stream0)
            del arg139_1
            # Topologically Sorted Source Nodes: [x_46, x_47], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf418 = extern_kernels.convolution(buf417, buf415, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf418, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf419 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47, add_50, h_19], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_20_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_20.run(buf419, buf416, buf403, triton_poi_fused__to_copy_add_convolution_silu_20_xnumel, stream=stream0)
            buf420 = buf404; del buf404  # reuse
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_4.run(arg140_1, buf420, 147456, stream=stream0)
            del arg140_1
            buf421 = buf416; del buf416  # reuse
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg141_1, buf421, 128, stream=stream0)
            del arg141_1
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.convolution]
            buf422 = extern_kernels.convolution(buf419, buf420, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf422, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf423 = buf408; del buf408  # reuse
            buf424 = buf407; del buf407  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5.run(buf422, buf421, buf423, buf424, triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel, 4096, stream=stream0)
            buf426 = reinterpret_tensor(buf403, (s13, 128, 1024), (131072, 1, 128), 0); del buf403  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14, x_flat_14, v_t_x_13], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17.run(buf422, buf421, buf423, buf424, buf426, triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel, stream=stream0)
            buf427 = buf411; del buf411  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, v_46, transpose_80, x_flat_14, v_t_x_13], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 12, 128), (3200, 1, 12), 1536), buf426, out=buf427)
            buf428 = reinterpret_tensor(buf426, (s13, 128, 1024), (131072, 1024, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [split_13, u_27, mixed_13], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 128, 12), (3200, 12, 1), 0), buf427, out=buf428)
            buf433 = buf417; del buf417  # reuse
            # Topologically Sorted Source Nodes: [x_48, x_norm_14, split_13, x_flat_14, out_50, view_116, shift_27, out_51, x_49, x_50], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18.run(buf422, buf421, buf423, buf424, buf428, buf31, buf433, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel, 1024, stream=stream0)
            del buf422
            del buf428
            buf431 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_19.run(arg142_1, buf431, 16384, stream=stream0)
            del arg142_1
            buf432 = buf421; del buf421  # reuse
            # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg143_1, buf432, 128, stream=stream0)
            del arg143_1
            # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf434 = extern_kernels.convolution(buf433, buf431, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf434, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf435 = buf434; del buf434  # reuse
            # Topologically Sorted Source Nodes: [x_49, x_50, add_53, h_20], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_20_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_20.run(buf435, buf432, buf419, triton_poi_fused__to_copy_add_convolution_silu_20_xnumel, stream=stream0)
            buf436 = buf420; del buf420  # reuse
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_4.run(arg144_1, buf436, 147456, stream=stream0)
            del arg144_1
            buf437 = buf432; del buf432  # reuse
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg145_1, buf437, 128, stream=stream0)
            del arg145_1
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy, aten.convolution]
            buf438 = extern_kernels.convolution(buf435, buf436, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf438, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf439 = buf424; del buf424  # reuse
            buf440 = buf423; del buf423  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5.run(buf438, buf437, buf439, buf440, triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel, 4096, stream=stream0)
            buf442 = reinterpret_tensor(buf419, (s13, 128, 1024), (131072, 1, 128), 0); del buf419  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15, x_flat_15, v_t_x_14], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17.run(buf438, buf437, buf439, buf440, buf442, triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel, stream=stream0)
            buf443 = buf427; del buf427  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, v_48, transpose_81, x_flat_15, v_t_x_14], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 12, 128), (3200, 1, 12), 1536), buf442, out=buf443)
            buf444 = reinterpret_tensor(buf442, (s13, 128, 1024), (131072, 1024, 1), 0); del buf442  # reuse
            # Topologically Sorted Source Nodes: [split_14, u_29, mixed_14], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 128, 12), (3200, 12, 1), 0), buf443, out=buf444)
            buf449 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [x_51, x_norm_15, split_14, x_flat_15, out_52, view_121, shift_29, out_53, x_52, x_53], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18.run(buf438, buf437, buf439, buf440, buf444, buf31, buf449, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel, 1024, stream=stream0)
            del buf438
            del buf444
            buf447 = buf431; del buf431  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_19.run(arg146_1, buf447, 16384, stream=stream0)
            del arg146_1
            buf448 = buf437; del buf437  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg147_1, buf448, 128, stream=stream0)
            del arg147_1
            # Topologically Sorted Source Nodes: [x_52, x_53], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf450 = extern_kernels.convolution(buf449, buf447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf450, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf451 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53, add_56, h_21], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add]
            triton_poi_fused__to_copy_add_convolution_silu_20_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_20.run(buf451, buf448, buf435, triton_poi_fused__to_copy_add_convolution_silu_20_xnumel, stream=stream0)
            buf452 = buf436; del buf436  # reuse
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_4.run(arg148_1, buf452, 147456, stream=stream0)
            del arg148_1
            buf453 = buf448; del buf448  # reuse
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg149_1, buf453, 128, stream=stream0)
            del arg149_1
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.convolution]
            buf454 = extern_kernels.convolution(buf451, buf452, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf454, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf452
            buf455 = buf440; del buf440  # reuse
            buf456 = buf439; del buf439  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_convolution_native_group_norm_5.run(buf454, buf453, buf455, buf456, triton_red_fused__to_copy_clone_convolution_native_group_norm_5_xnumel, 4096, stream=stream0)
            buf458 = reinterpret_tensor(buf435, (s13, 128, 1024), (131072, 1, 128), 0); del buf435  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16, x_flat_16, v_t_x_15], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.view]
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel = 131072*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17.run(buf454, buf453, buf455, buf456, buf458, triton_poi_fused__to_copy_clone_convolution_native_group_norm_view_17_xnumel, stream=stream0)
            buf459 = buf443; del buf443  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, v_50, transpose_82, x_flat_16, v_t_x_15], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 12, 128), (3200, 1, 12), 1536), buf458, out=buf459)
            buf460 = reinterpret_tensor(buf458, (s13, 128, 1024), (131072, 1024, 1), 0); del buf458  # reuse
            # Topologically Sorted Source Nodes: [split_15, u_31, mixed_15], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf31, (s13, 128, 12), (3200, 12, 1), 0), buf459, out=buf460)
            del buf459
            buf465 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_norm_16, split_15, x_flat_16, out_54, view_126, shift_31, out_55, x_55, x_56], Original ATen: [aten._to_copy, aten.convolution, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18.run(buf454, buf453, buf455, buf456, buf460, buf31, buf465, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_split_with_sizes_view_18_ynumel, 1024, stream=stream0)
            del buf31
            del buf454
            del buf460
            buf463 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [x_55, x_56], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_19.run(arg150_1, buf463, 16384, stream=stream0)
            del arg150_1
            buf464 = buf453; del buf453  # reuse
            # Topologically Sorted Source Nodes: [x_55, x_56], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_2.run(arg151_1, buf464, 128, stream=stream0)
            del arg151_1
            # Topologically Sorted Source Nodes: [x_55, x_56], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf466 = extern_kernels.convolution(buf465, buf463, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf466, (s13, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf463
            buf467 = buf456; del buf456  # reuse
            buf468 = buf455; del buf455  # reuse
            # Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
            triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_67_xnumel = 32*s13
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_67.run(buf466, buf464, buf451, buf467, buf468, triton_red_fused__to_copy_add_clone_convolution_native_group_norm_silu_67_xnumel, 4096, stream=stream0)
            buf471 = reinterpret_tensor(buf465, (s13, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf465  # reuse
            # Topologically Sorted Source Nodes: [x_55, x_56, add_59, h_22, input_42, input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution, aten.add, aten.clone, aten.native_group_norm]
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_68_ynumel = 1024*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_68.run(buf466, buf464, buf451, buf467, buf468, arg152_1, arg153_1, buf471, triton_poi_fused__to_copy_add_clone_convolution_native_group_norm_silu_68_ynumel, 128, stream=stream0)
            del arg152_1
            del arg153_1
            del buf451
            del buf464
            del buf467
            del buf468
            buf472 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_1.run(arg154_1, buf472, 4608, stream=stream0)
            del arg154_1
            buf473 = empty_strided_cuda((4, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_69.run(arg155_1, buf473, 4, stream=stream0)
            del arg155_1
            buf474 = buf466; del buf466  # reuse
            # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            triton_poi_fused__to_copy_convolution_silu_70_ynumel = 128*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_70.run(buf471, buf474, triton_poi_fused__to_copy_convolution_silu_70_ynumel, 1024, stream=stream0)
            del buf471
            # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            buf475 = extern_kernels.convolution(buf474, buf472, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf475, (s13, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
            del buf472
            del buf474
            buf476 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.silu, aten._to_copy, aten.convolution]
            triton_poi_fused__to_copy_convolution_silu_71_xnumel = 4096*s13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_silu_71.run(buf476, buf473, triton_poi_fused__to_copy_convolution_silu_71_xnumel, stream=stream0)
            del buf473
        return (buf476, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = 4
    arg8_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg9_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3200, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((6400, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((6400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((12800, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((12800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, 4, 3, 3), (36, 1, 12, 4), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((4, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((4, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
