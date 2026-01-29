# AOT ID: ['0_forward']
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vj/cvjqfuizgqedplsb2qeiqfmw2bkzhdudz6t7b62agyygh54e3fso.py
# Topologically Sorted Source Nodes: [arange, mul, truediv, freqs, getitem, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.arange, aten.mul, aten.div, aten.exp, aten.unsqueeze, aten.cos, aten.sin, aten.cat, aten._to_copy]
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
#   %primals_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_1]
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, -9.210340371976184), kwargs = {})
#   %div : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 128), kwargs = {})
#   %exp : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
#   %unsqueeze : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp, 0), kwargs = {})
#   %mul_1 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %cos : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %sin : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %cat : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_0 = async_compile.triton('triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
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
    tmp20 = tl.load(in_ptr0 + (x1), tmp17, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x2), tmp33, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/4u/c4u7buso2rtjhubpmvzdnc5jtu7juvnkhv5velbksz2f7l4krrks.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_1
# Graph fragment:
#   %primals_2 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %convert_element_type_1 : Tensor "bf16[1024, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/n3/cn36lg7bj3k3dli5xximcedtnq473f63ezjgfifedrvf2x53ib5o.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type
# Graph fragment:
#   %primals_3 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=primals_3]
#   %convert_element_type : Tensor "bf16[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/m6/cm6wh2utu2lsmutgllfizc6vpefk45gyyxnmet2f72psmo6fveap.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_6, convert_element_type_7, mul_2, sigmoid
# Graph fragment:
#   %addmm : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=addmm]
#   %convert_element_type_6 : Tensor "f32[128, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul_2 : Tensor "f32[128, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_7
triton_poi_fused_silu_3 = async_compile.triton('triton_poi_fused_silu_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hr/chrqcosavxqso5c6uyipotfsgwt3b6klk7ruhamrohh637fvammn.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_8
# Graph fragment:
#   %primals_5 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_5]
#   %convert_element_type_8 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
#   return %convert_element_type_8
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qb/cqbe2g7rvui7uhx5fetfwpnnzh37uc7qfavk5wq5tvbuuroeyqdj.py
# Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   cat_1 => cat_1
#   input_4 => convert_element_type_15
#   s_emb => embedding
# Graph fragment:
#   %addmm_1 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %primals_7 : Tensor "i64[128][1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_6 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %embedding : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_6, %primals_7), kwargs = {})
#   %cat_1 : Tensor "f32[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%addmm_1, %embedding], -1), kwargs = {})
#   %convert_element_type_15 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_15
triton_poi_fused__to_copy_cat_embedding_5 = async_compile.triton('triton_poi_fused__to_copy_cat_embedding_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_embedding_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_embedding_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full([XBLOCK], 4, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tl.broadcast_to(tmp16, [XBLOCK])) & (tl.broadcast_to(tmp16, [XBLOCK]) < 4)) | ~(tmp9), "index out of bounds: 0 <= tl.broadcast_to(tmp16, [XBLOCK]) < 4")
    tmp18 = tl.load(in_ptr2 + (256*tmp16 + ((-256) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp8, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lf/clftxbqspklvfixhzuqur3ut4oeyjj2l3fzgkbtxss74q3tb3mdp.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_13
# Graph fragment:
#   %primals_9 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_9]
#   %convert_element_type_13 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_13
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rh/crh3db36ipes2beohagsxivt5uhmgybpcmtmwjsuurp37jdgithj.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_19, convert_element_type_20, mul_3, sigmoid_1
# Graph fragment:
#   %addmm_2 : Tensor "bf16[128, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %convert_element_type_19 : Tensor "f32[128, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm_2, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_19,), kwargs = {})
#   %mul_3 : Tensor "f32[128, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %sigmoid_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[128, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_20
triton_poi_fused_silu_7 = async_compile.triton('triton_poi_fused_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ye/cye3tc6fi7wb6qqdnuxsre6mlt5uuyckowtfuk5siedf4zwnkblw.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_6 => convert_element_type_22
# Graph fragment:
#   %primals_10 : Tensor "f32[256, 512][512, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %convert_element_type_22 : Tensor "bf16[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.bfloat16), kwargs = {})
#   return %convert_element_type_22
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vt/cvt5iwrl5wivvqzxsqpruimqfqlibazse7or3gsxky6ok6ktghdl.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_7 => convert_element_type_27
# Graph fragment:
#   %primals_12 : Tensor "f32[256, 256][256, 1]cuda:0" = PlaceHolder[target=primals_12]
#   %convert_element_type_27 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_12, torch.bfloat16), kwargs = {})
#   return %convert_element_type_27
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/an/can2wktesv6p5awa63qopa4zrtxh2gzwwqys3ypbsifanpq6dyka.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   input_8 => convert_element_type_31, convert_element_type_32, mul_4, sigmoid_2
# Graph fragment:
#   %addmm_4 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %convert_element_type_31 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm_4, torch.float32), kwargs = {})
#   %sigmoid_2 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_31,), kwargs = {})
#   %mul_4 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, %sigmoid_2), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   return %convert_element_type_32
triton_poi_fused_silu_10 = async_compile.triton('triton_poi_fused_silu_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/xc/cxcnebpg2eupxouncsddhyc4gfrcqaa3vyar5ymu7fprb7a34thp.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_34
# Graph fragment:
#   %primals_14 : Tensor "f32[3200, 256][256, 1]cuda:0" = PlaceHolder[target=primals_14]
#   %convert_element_type_34 : Tensor "bf16[3200, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.bfloat16), kwargs = {})
#   return %convert_element_type_34
triton_poi_fused__to_copy_11 = async_compile.triton('triton_poi_fused__to_copy_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6553600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/s4/cs4ummahgxxjguafi7vutuwz2vuepgongxtzyiannqxqn44mu4k7.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_33
# Graph fragment:
#   %primals_15 : Tensor "f32[3200][1]cuda:0" = PlaceHolder[target=primals_15]
#   %convert_element_type_33 : Tensor "bf16[3200][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_15, torch.bfloat16), kwargs = {})
#   return %convert_element_type_33
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wc/cwcbetnhfylhj22k257i2irmsqkjwkakp33lkzusbeoi3gpsup2j.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_46
# Graph fragment:
#   %primals_18 : Tensor "f32[6400, 256][256, 1]cuda:0" = PlaceHolder[target=primals_18]
#   %convert_element_type_46 : Tensor "bf16[6400, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_18, torch.bfloat16), kwargs = {})
#   return %convert_element_type_46
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13107200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/7c/c7cxui6lr7rr3xlpt3hrsep3pveqqhauqnmun7nte3lhach5sxcc.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_45
# Graph fragment:
#   %primals_19 : Tensor "f32[6400][1]cuda:0" = PlaceHolder[target=primals_19]
#   %convert_element_type_45 : Tensor "bf16[6400][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_19, torch.bfloat16), kwargs = {})
#   return %convert_element_type_45
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/br/cbry6cbxcanzd3qqmoxcqkw6darkumbzn2j6tck2suzathyv2mhv.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_58
# Graph fragment:
#   %primals_22 : Tensor "f32[12800, 256][256, 1]cuda:0" = PlaceHolder[target=primals_22]
#   %convert_element_type_58 : Tensor "bf16[12800, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_22, torch.bfloat16), kwargs = {})
#   return %convert_element_type_58
triton_poi_fused__to_copy_15 = async_compile.triton('triton_poi_fused__to_copy_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26214400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3276800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/rf/crfqko53wehp6asobbtz2zp5v4txh7mkmjck2tofbdcpaew3fta4.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_57
# Graph fragment:
#   %primals_23 : Tensor "f32[12800][1]cuda:0" = PlaceHolder[target=primals_23]
#   %convert_element_type_57 : Tensor "bf16[12800][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_23, torch.bfloat16), kwargs = {})
#   return %convert_element_type_57
triton_poi_fused__to_copy_16 = async_compile.triton('triton_poi_fused__to_copy_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ue/cue4a4m3yyio4c265otqlkanvlk5fzu5xpgbudwhseu5qvnh3xll.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h => convert_element_type_63
# Graph fragment:
#   %primals_24 : Tensor "f32[128, 4, 3, 3][36, 1, 12, 4]cuda:0" = PlaceHolder[target=primals_24]
#   %convert_element_type_63 : Tensor "bf16[128, 4, 3, 3][36, 1, 12, 4]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_24, torch.bfloat16), kwargs = {})
#   return %convert_element_type_63
triton_poi_fused__to_copy_17 = async_compile.triton('triton_poi_fused__to_copy_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ky/ckyb5j3qrnjs463njb6estz27wpy2yartqbqejxvbjedpb6bospa.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h => convert_element_type_64
# Graph fragment:
#   %primals_26 : Tensor "f32[128, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=primals_26]
#   %convert_element_type_64 : Tensor "bf16[128, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_26, torch.bfloat16), kwargs = {})
#   return %convert_element_type_64
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/tt/cttzla4jhncdtkh36d2lbanwcxt6m23nqcacq2vrtttjynpqsbg2.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h => convert_element_type_62, convolution
# Graph fragment:
#   %primals_25 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_25]
#   %convert_element_type_62 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_25, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_64, %convert_element_type_63, %convert_element_type_62, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf39
triton_poi_fused__to_copy_convolution_19 = async_compile.triton('triton_poi_fused__to_copy_convolution_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hd/chdrxst6rr56gzz7tnzk3h77sj7zm6jgoulnpmcuzkntylqnegnn.py
# Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h => convert_element_type_62, convolution
# Graph fragment:
#   %buf40 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf40]
#   %buf39 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf39]
#   %convert_element_type_62 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_25, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_64, %convert_element_type_63, %convert_element_type_62, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused__to_copy_convolution_20 = async_compile.triton('triton_poi_fused__to_copy_convolution_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/yf/cyfazbxrtm25aougl6txn5cmiy3kekjd3gjgk7bzdmuasnpa6ejf.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type_66
# Graph fragment:
#   %primals_27 : Tensor "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=primals_27]
#   %convert_element_type_66 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_27, torch.bfloat16), kwargs = {})
#   return %convert_element_type_66
triton_poi_fused__to_copy_21 = async_compile.triton('triton_poi_fused__to_copy_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/lh/clhagcm7lw2hi35avsk64wsvjvoytb6kgprhper4bnyffiitpbdw.py
# Topologically Sorted Source Nodes: [x_norm], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_norm => add, clone, convert_element_type_67, rsqrt, var_mean, view
# Graph fragment:
#   %convolution_1 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_1]
#   %buf47 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf47]
#   %convert_element_type_67 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [128, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   return %getitem_1,%buf47,%rsqrt
triton_red_fused__to_copy_clone_native_group_norm_22 = async_compile.triton('triton_red_fused__to_copy_clone_native_group_norm_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_native_group_norm_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_clone_native_group_norm_22(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
    tmp9 = 4096.0
    tmp10 = (tmp7 / tmp9)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.store(out_ptr2 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/t5/ct5jkondwkknt2vpgx5x2ndyctbxvjhtmkrz7mwvy4i7pyjsatl3.py
# Topologically Sorted Source Nodes: [x_norm, x_flat, v_t_x], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   v_t_x => convert_element_type_68
#   x_flat => view_5
#   x_norm => add, clone, convert_element_type_67, mul_7, rsqrt, sub, var_mean, view, view_1
# Graph fragment:
#   %convolution_1 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_1]
#   %getitem_1 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf47 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf47]
#   %convert_element_type_68 : Tensor "bf16[128, 128, 1024][131072, 1, 128]cuda:0" = PlaceHolder[target=convert_element_type_68]
#   %convert_element_type_67 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [128, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %mul_7 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [128, 128, 32, 32]), kwargs = {})
#   %view_5 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, 128, -1]), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.bfloat16), kwargs = {})
#   %permute_496 : Tensor "bf16[128, 1024, 128][131072, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_68, [0, 2, 1]), kwargs = {})
#   return %convert_element_type_68,%permute_496
triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23 = async_compile.triton('triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 128*y0 + 4096*(((y0 % 32)) // 32) + 131072*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*y1 + (x2 // 4)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*y1 + (x2 // 4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 4096.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 128*y3), tmp11, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 1024*x2 + 131072*y1), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/en/cenou4uj6mlqvhs574y42rkb43epemrqhbxtoewbvzu5czaveqak.py
# Topologically Sorted Source Nodes: [x_norm, split, shift_1, x_flat, out, view_4, out_1, x_1, x_2], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   out => add_1
#   out_1 => add_2
#   shift_1 => view_4
#   split => split_with_sizes
#   view_4 => view_6
#   x_1 => mul_8, sigmoid_5
#   x_2 => convert_element_type_73, convert_element_type_75, convolution_2
#   x_flat => view_5
#   x_norm => add, clone, convert_element_type_67, mul_7, rsqrt, sub, var_mean, view, view_1
# Graph fragment:
#   %convolution_1 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_1]
#   %getitem_1 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf47 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf47]
#   %bmm_1 : Tensor "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %addmm_5 : Tensor "bf16[128, 3200][3200, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %add_2 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=add_2]
#   %convert_element_type_75 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=convert_element_type_75]
#   %convert_element_type_67 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %clone : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_67,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [128, 32, 4, 1024]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %mul_7 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [128, 128, 32, 32]), kwargs = {})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_5, [1536, 1536, 128], 1), kwargs = {})
#   %view_4 : Tensor "bf16[128, 128, 1, 1][3200, 1, 1, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [128, 128, 1, 1]), kwargs = {})
#   %view_5 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [128, 128, -1]), kwargs = {})
#   %add_1 : Tensor "f32[128, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %bmm_1), kwargs = {})
#   %view_6 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [128, 128, 32, 32]), kwargs = {})
#   %add_2 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_4), kwargs = {})
#   %sigmoid_5 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_8 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sigmoid_5), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_30, torch.bfloat16), kwargs = {})
#   %convert_element_type_75 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_75, %convert_element_type_74, %convert_element_type_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %full_default : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([128, 128, 32, 32], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_104 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_5), kwargs = {})
#   %mul_434 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %sub_104), kwargs = {})
#   %add_214 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_434, 1), kwargs = {})
#   %mul_435 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_5, %add_214), kwargs = {})
#   return %add_2,%convert_element_type_75,%mul_435,%buf57
triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 234881024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1024)
    x1 = ((xindex // 1024) % 128)
    x2 = xindex // 131072
    x4 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 128*x0 + 131072*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (3072 + x1 + 3200*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 4096.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp17
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22 + tmp20
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp24, None)
    tl.store(out_ptr3 + (x1 + 128*x0 + 131072*x2), tmp19, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/xb/cxbxabwcqi5fm7f7avuzutkrv7swsj2ohk2nlzq2ncplomnvrkyd.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_2 => convert_element_type_74
# Graph fragment:
#   %primals_29 : Tensor "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=primals_29]
#   %convert_element_type_74 : Tensor "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_29, torch.bfloat16), kwargs = {})
#   return %convert_element_type_74
triton_poi_fused__to_copy_25 = async_compile.triton('triton_poi_fused__to_copy_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/nw/cnw2bzl7jv3kkq25dhvjsiln7v7ty4jzk6pp6axj6gufusv45grk.py
# Topologically Sorted Source Nodes: [x_2, add_2, h_1], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_2 => add_3
#   h_1 => convert_element_type_76, convert_element_type_77, mul_9, sigmoid_6
#   x_2 => convert_element_type_73, convolution_2
# Graph fragment:
#   %buf58 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf58]
#   %buf56 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf56]
#   %convolution_2 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_2]
#   %convolution : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution]
#   %convert_element_type_73 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_30, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_75, %convert_element_type_74, %convert_element_type_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %sigmoid_6 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_76,), kwargs = {})
#   %mul_9 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_76, %sigmoid_6), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   return %convolution_2,%convert_element_type_77
triton_poi_fused__to_copy_add_convolution_silu_26 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201326848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_26(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/j6/cj67tej2gww7um24zm45smij3o6n5dhxcmbakg5pghuhftxzp2bi.py
# Topologically Sorted Source Nodes: [x_5, add_5, h_2], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_5 => add_7
#   h_2 => convert_element_type_89, convert_element_type_90, mul_12, sigmoid_8
#   x_5 => convert_element_type_86, convolution_4
# Graph fragment:
#   %buf77 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf77]
#   %buf75 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf75]
#   %convert_element_type_77 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_77]
#   %convert_element_type_86 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_34, torch.bfloat16), kwargs = {})
#   %convolution_4 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_88, %convert_element_type_87, %convert_element_type_86, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_7 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_4, %convert_element_type_77), kwargs = {})
#   %convert_element_type_89 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %sigmoid_8 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_89,), kwargs = {})
#   %mul_12 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %sigmoid_8), kwargs = {})
#   %convert_element_type_90 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   return %convert_element_type_90
triton_poi_fused__to_copy_add_convolution_silu_27 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134217984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/3d/c3da2pri2z25kajtypa6yrqxgqikmx4ahjukuw6rjd4ybhtwfodi.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_3 => convert_element_type_92
# Graph fragment:
#   %primals_35 : Tensor "f32[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=primals_35]
#   %convert_element_type_92 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_35, torch.bfloat16), kwargs = {})
#   return %convert_element_type_92
triton_poi_fused__to_copy_28 = async_compile.triton('triton_poi_fused__to_copy_28', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/w3/cw3hflsgjyiijf6v3wssxwsvnjlzvpny4mssndswmsflrsraxqeb.py
# Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_3 => convert_element_type_91, convolution_5
# Graph fragment:
#   %buf81 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf81]
#   %buf80 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf80]
#   %convert_element_type_91 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_36, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %convert_element_type_92, %convert_element_type_91, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_5
triton_poi_fused__to_copy_convolution_29 = async_compile.triton('triton_poi_fused__to_copy_convolution_29', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50332160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_29(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/dx/cdx22aqw7rxn4kqdtuzohoser24b65zdcg75kvbg2vc3vn2tmgr5.py
# Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_2
#   q_1 => view_16
#   q_2 => permute_15
#   view_11 => view_17
# Graph fragment:
#   %mm : Tensor "bf16[32768, 256][256, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_16 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 256, 256]), kwargs = {})
#   %view_17 : Tensor "bf16[128, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [128, 256, 4, 64]), kwargs = {})
#   %permute_15 : Tensor "bf16[128, 4, 256, 64][65536, 64, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "bf16[128, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused__unsafe_view_clone_transpose_view_30 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_30', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wh/cwhkcdsaiceiouzg62a6udadr6kzrwkgevju22vxpiq7g7o4no5w.py
# Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   k_1 => permute_16
#   kv => unsqueeze_2
#   matmul => clone_3
#   transpose_6 => permute_18
#   view_12 => view_18
# Graph fragment:
#   %mm_1 : Tensor "bf16[128, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "bf16[128, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_18 : Tensor "bf16[128, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_10, [128, 1, 4, 64]), kwargs = {})
#   %permute_16 : Tensor "bf16[128, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %permute_18 : Tensor "bf16[128, 4, 64, 1][512, 64, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_16, [0, 1, 3, 2]), kwargs = {})
#   %clone_3 : Tensor "bf16[128, 4, 64, 1][256, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ck/cck2iw7jidu2mgtpwl3paxyaook3iig2ye7ogt4prelffp4lij7u.py
# Topologically Sorted Source Nodes: [matmul, attn_1, matmul_1], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div_1, exp_1, sum_1
#   matmul => view_22
#   matmul_1 => convert_element_type_102
# Graph fragment:
#   %bmm_4 : Tensor "bf16[512, 256, 1][256, 1, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %view_22 : Tensor "bf16[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_4, [128, 4, 256, 1]), kwargs = {})
#   %convert_element_type_default_35 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_22, torch.float32), kwargs = {})
#   %mul_tensor_18 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_35, 1), kwargs = {})
#   %amax_default_9 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_18, [-1], True), kwargs = {})
#   %sub_tensor_9 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_18, %amax_default_9), kwargs = {})
#   %mul_tensor_19 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_9, 0.125), kwargs = {})
#   %exp_1 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_19,), kwargs = {})
#   %sum_1 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div_1 : Tensor "f32[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_1), kwargs = {})
#   %convert_element_type_102 : Tensor "bf16[128, 4, 256, 1][1024, 256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   return %expand_2
triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32 = async_compile.triton('triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 - tmp3
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = (tmp7 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/bd/cbdcw4pia3r44t7bmwlz7xdlomyimo3gsmlydefotzfijb7nviyh.py
# Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk => split
#   kv => unsqueeze_2
#   matmul_1 => clone_4
#   v_5 => permute_17
#   view_13 => view_19
# Graph fragment:
#   %mm_1 : Tensor "bf16[128, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %unsqueeze_2 : Tensor "bf16[128, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_2, 256, -1), kwargs = {})
#   %view_19 : Tensor "bf16[128, 1, 4, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_11, [128, 1, 4, 64]), kwargs = {})
#   %permute_17 : Tensor "bf16[128, 4, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %clone_4 : Tensor "bf16[128, 4, 1, 64][256, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 512*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/si/csi7ls767f4fy23kbdrxdplxsvid6lv7uyseje2udndzj5xa7ncl.py
# Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => view_25
#   out_4 => clone_5
#   transpose_7 => permute_19
# Graph fragment:
#   %bmm_5 : Tensor "bf16[512, 256, 64][16384, 64, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %view_25 : Tensor "bf16[128, 4, 256, 64][65536, 16384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [128, 4, 256, 64]), kwargs = {})
#   %permute_19 : Tensor "bf16[128, 256, 4, 64][65536, 64, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_25, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "bf16[128, 256, 4, 64][65536, 256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_clone_transpose_view_34 = async_compile.triton('triton_poi_fused_clone_transpose_view_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ig/cige5ug4wlg4bk3fgnnytzrtnvr4ay6q6wzy2iahoocuucrrlxjk.py
# Topologically Sorted Source Nodes: [transpose_8, out_5, x_6], Original ATen: [aten.transpose, aten.view, aten.add]
# Source node to ATen node mapping:
#   out_5 => view_29
#   transpose_8 => permute_21
#   x_6 => add_8
# Graph fragment:
#   %convolution_5 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %getitem_12 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=getitem_12]
#   %permute_21 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_12, [0, 2, 1]), kwargs = {})
#   %view_29 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [128, 256, 16, 16]), kwargs = {})
#   %add_8 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %view_29), kwargs = {})
#   return %add_8
triton_poi_fused_add_transpose_view_35 = async_compile.triton('triton_poi_fused_add_transpose_view_35', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_transpose_view_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67108864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_transpose_view_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2i/c2ijgiocdgdmqsijvza3rkcyrdb2nlmeoi37pn2xbv7dow7j4anv.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_7 => convert_element_type_111
# Graph fragment:
#   %primals_41 : Tensor "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=primals_41]
#   %convert_element_type_111 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_41, torch.bfloat16), kwargs = {})
#   return %convert_element_type_111
triton_poi_fused__to_copy_36 = async_compile.triton('triton_poi_fused__to_copy_36', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/fx/cfx63wyl7u26rvy63dihetdqsf4phcvs5spxzrtjmfhtzswlrvqu.py
# Topologically Sorted Source Nodes: [x_norm_2], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_norm_2 => add_9, clone_6, convert_element_type_112, rsqrt_2, var_mean_2, view_30
# Graph fragment:
#   %convolution_6 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_6]
#   %buf106 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf106]
#   %convert_element_type_112 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_6 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [128, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   return %getitem_15,%buf106,%rsqrt_2
triton_red_fused__to_copy_clone_native_group_norm_37 = async_compile.triton('triton_red_fused__to_copy_clone_native_group_norm_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 2048},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_native_group_norm_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_clone_native_group_norm_37(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 256*r0_3 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
    tmp9 = 2048.0
    tmp10 = (tmp7 / tmp9)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.store(out_ptr2 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/hr/chrreqk7hr3hbjv6c4ka2sxgthhyt5lropj2ahdappp64xfy7sql.py
# Topologically Sorted Source Nodes: [x_norm_2, x_flat_2, v_t_x_2], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   v_t_x_2 => convert_element_type_113
#   x_flat_2 => view_35
#   x_norm_2 => add_9, clone_6, convert_element_type_112, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
# Graph fragment:
#   %convolution_6 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_6]
#   %getitem_15 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf106 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf106]
#   %convert_element_type_113 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0" = PlaceHolder[target=convert_element_type_113]
#   %convert_element_type_112 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_6 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [128, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_3 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_15), kwargs = {})
#   %mul_14 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [128, 256, 16, 16]), kwargs = {})
#   %view_35 : Tensor "f32[128, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [128, 256, -1]), kwargs = {})
#   %convert_element_type_113 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.bfloat16), kwargs = {})
#   %permute_457 : Tensor "bf16[128, 256, 256][65536, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_113, [0, 2, 1]), kwargs = {})
#   return %convert_element_type_113,%permute_457
triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38 = async_compile.triton('triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 33554432, 'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y0 + 4096*(((y0 % 16)) // 16) + 65536*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*y1 + (x2 // 8)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*y1 + (x2 // 8)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 2048.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 256*y3), tmp11, xmask)
    tl.store(out_ptr1 + (y0 + 256*x2 + 65536*y1), tmp11, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/in/cinheea5z3y2tpmacjiosfqd5gqo2mvu45vd6uxfv7j5vp27nrnh.py
# Topologically Sorted Source Nodes: [x_norm_2, split_2, shift_5, x_flat_2, out_6, view_19, out_7, x_8, x_9], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   out_6 => add_10
#   out_7 => add_11
#   shift_5 => view_34
#   split_2 => split_with_sizes_2
#   view_19 => view_36
#   x_8 => mul_15, sigmoid_9
#   x_9 => convert_element_type_118, convert_element_type_120, convolution_7
#   x_flat_2 => view_35
#   x_norm_2 => add_9, clone_6, convert_element_type_112, mul_14, rsqrt_2, sub_3, var_mean_2, view_30, view_31
# Graph fragment:
#   %convolution_6 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_6]
#   %getitem_15 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf106 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf106]
#   %bmm_7 : Tensor "bf16[128, 256, 256][65536, 256, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %addmm_7 : Tensor "bf16[128, 6400][6400, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %add_11 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=add_11]
#   %convert_element_type_120 : Tensor "bf16[128, 256, 16, 16][65536, 256, 16, 1]cuda:0" = PlaceHolder[target=convert_element_type_120]
#   %convert_element_type_112 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %clone_6 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_112,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [128, 32, 8, 256]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_3 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_15), kwargs = {})
#   %mul_14 : Tensor "f32[128, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %view_31 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [128, 256, 16, 16]), kwargs = {})
#   %split_with_sizes_2 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_7, [3072, 3072, 256], 1), kwargs = {})
#   %view_34 : Tensor "bf16[128, 256, 1, 1][6400, 1, 1, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_18, [128, 256, 1, 1]), kwargs = {})
#   %view_35 : Tensor "f32[128, 256, 256][65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_31, [128, 256, -1]), kwargs = {})
#   %add_10 : Tensor "f32[128, 256, 256][65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %bmm_7), kwargs = {})
#   %view_36 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [128, 256, 16, 16]), kwargs = {})
#   %add_11 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_36, %view_34), kwargs = {})
#   %sigmoid_9 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_9), kwargs = {})
#   %convert_element_type_118 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_44, torch.bfloat16), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %convolution_7 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %convert_element_type_119, %convert_element_type_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %full_default_12 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([128, 256, 16, 16], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_96 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_12, %sigmoid_9), kwargs = {})
#   %mul_394 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sub_96), kwargs = {})
#   %add_197 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_394, 1), kwargs = {})
#   %mul_395 : Tensor "f32[128, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_9, %add_197), kwargs = {})
#   return %add_11,%convert_element_type_120,%mul_395,%buf116
triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 117440512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 256)
    x2 = xindex // 65536
    x4 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*x0 + 65536*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (6144 + x1 + 6400*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 2048.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp17
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22 + tmp20
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp24, None)
    tl.store(out_ptr3 + (x1 + 256*x0 + 65536*x2), tmp19, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/bb/cbbpqgupkzetf2alyeu25vbwbwvzcpwfkcxyeiwimg2h5bfbkkui.py
# Topologically Sorted Source Nodes: [x_9, add_9, h_4], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_9 => add_12
#   h_4 => convert_element_type_121, convert_element_type_122, mul_16, sigmoid_10
#   x_9 => convert_element_type_118, convolution_7
# Graph fragment:
#   %buf117 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf117]
#   %buf115 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf115]
#   %convolution_7 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_7]
#   %convolution_5 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %convert_element_type_118 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_44, torch.bfloat16), kwargs = {})
#   %convolution_7 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %convert_element_type_119, %convert_element_type_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %convolution_5), kwargs = {})
#   %convert_element_type_121 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %sigmoid_10 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_121,), kwargs = {})
#   %mul_16 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_121, %sigmoid_10), kwargs = {})
#   %convert_element_type_122 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
#   return %convolution_7,%convert_element_type_122
triton_poi_fused__to_copy_add_convolution_silu_40 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663808}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_40(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/fk/cfkyoinvsvktzubteog7ycih3fqvepdqavhlb2upfhlud44tr5bt.py
# Topologically Sorted Source Nodes: [x_13, add_13, h_5], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_13 => add_17
#   h_5 => convert_element_type_151, convert_element_type_152, mul_20, sigmoid_12
#   x_13 => convert_element_type_148, convolution_9
# Graph fragment:
#   %buf154 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf154]
#   %buf152 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf152]
#   %convert_element_type_122 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convert_element_type_122]
#   %convert_element_type_148 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_52, torch.bfloat16), kwargs = {})
#   %convolution_9 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_150, %convert_element_type_149, %convert_element_type_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_17 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %convert_element_type_122), kwargs = {})
#   %convert_element_type_151 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.float32), kwargs = {})
#   %sigmoid_12 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_151,), kwargs = {})
#   %mul_20 : Tensor "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_151, %sigmoid_12), kwargs = {})
#   %convert_element_type_152 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_20, torch.bfloat16), kwargs = {})
#   return %convert_element_type_152
triton_poi_fused__to_copy_add_convolution_silu_41 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67109376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/5y/c5ygb3jqe2czv6ckqg4oahccdnvnqkvdk4uwlvsy4j75y37ylmt2.py
# Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_6 => convert_element_type_154
# Graph fragment:
#   %primals_53 : Tensor "f32[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=primals_53]
#   %convert_element_type_154 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_53, torch.bfloat16), kwargs = {})
#   return %convert_element_type_154
triton_poi_fused__to_copy_42 = async_compile.triton('triton_poi_fused__to_copy_42', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/is/cisffrlwlhlp6z37bnyquebv6fwxdherc7s72coyhwef6gv7he2g.py
# Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   h_6 => convert_element_type_153, convolution_10
# Graph fragment:
#   %buf158 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf158]
#   %buf157 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf157]
#   %convert_element_type_153 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_54, torch.bfloat16), kwargs = {})
#   %convolution_10 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_152, %convert_element_type_154, %convert_element_type_153, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_10
triton_poi_fused__to_copy_convolution_43 = async_compile.triton('triton_poi_fused__to_copy_convolution_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25166848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_43(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/fg/cfg7kdhvo4lbtzsrypxdikyojjmjlvdwxtdjl3o7g2dpv62pnfsc.py
# Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_4 => clone_12
#   q_7 => view_62
#   q_8 => permute_37
#   view_31 => view_63
# Graph fragment:
#   %mm_4 : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %view_62 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [128, 64, 512]), kwargs = {})
#   %view_63 : Tensor "bf16[128, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_62, [128, 64, 8, 64]), kwargs = {})
#   %permute_37 : Tensor "bf16[128, 8, 64, 64][32768, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_63, [0, 2, 1, 3]), kwargs = {})
#   %clone_12 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_12
triton_poi_fused__unsafe_view_clone_transpose_view_44 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_44', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/uh/cuhjumfu2lalq5fgojj25ofyufbftvzmmattcdjxz4dphcg4jcot.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   k_5 => permute_38
#   kv_2 => unsqueeze_4
#   matmul_4 => clone_13
#   transpose_22 => permute_40
#   view_32 => view_64
# Graph fragment:
#   %mm_5 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "bf16[128, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_64 : Tensor "bf16[128, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_28, [128, 1, 8, 64]), kwargs = {})
#   %permute_38 : Tensor "bf16[128, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_64, [0, 2, 1, 3]), kwargs = {})
#   %permute_40 : Tensor "bf16[128, 8, 64, 1][1024, 64, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_38, [0, 1, 3, 2]), kwargs = {})
#   %clone_13 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_13
triton_poi_fused_clone_split_transpose_unsqueeze_view_45 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/wp/cwpxldq3fttjnljrdb2fagupcck2d5gowkjg7cmk5de4bkkznpyo.py
# Topologically Sorted Source Nodes: [matmul_4, attn_5, matmul_5], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_5 => div_3, exp_3, sum_3
#   matmul_4 => view_68
#   matmul_5 => convert_element_type_164
# Graph fragment:
#   %bmm_12 : Tensor "bf16[1024, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=bmm_12]
#   %view_68 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_12, [128, 8, 64, 1]), kwargs = {})
#   %convert_element_type_default_33 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_68, torch.float32), kwargs = {})
#   %mul_tensor_14 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_33, 1), kwargs = {})
#   %amax_default_7 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_14, [-1], True), kwargs = {})
#   %sub_tensor_7 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_14, %amax_default_7), kwargs = {})
#   %mul_tensor_15 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_7, 0.125), kwargs = {})
#   %exp_3 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_15,), kwargs = {})
#   %sum_3 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [-1], True), kwargs = {})
#   %div_3 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_3, %sum_3), kwargs = {})
#   %convert_element_type_164 : Tensor "bf16[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   return %expand_10
triton_poi_fused__softmax__to_copy_amax_mul_sub_view_46 = async_compile.triton('triton_poi_fused__softmax__to_copy_amax_mul_sub_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_amax_mul_sub_view_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__to_copy_amax_mul_sub_view_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 - tmp3
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = (tmp7 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/we/cwewnosh25prm44soz425ragmnrvbaofwbuil4xrk2qgvucalw26.py
# Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   chunk_2 => split_2
#   kv_2 => unsqueeze_4
#   matmul_5 => clone_14
#   v_13 => permute_39
#   view_33 => view_65
# Graph fragment:
#   %mm_5 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %unsqueeze_4 : Tensor "bf16[128, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_5, 1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%unsqueeze_4, 512, -1), kwargs = {})
#   %view_65 : Tensor "bf16[128, 1, 8, 64][1024, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_29, [128, 1, 8, 64]), kwargs = {})
#   %permute_39 : Tensor "bf16[128, 8, 1, 64][1024, 64, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_65, [0, 2, 1, 3]), kwargs = {})
#   %clone_14 : Tensor "bf16[128, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_14
triton_poi_fused_clone_split_transpose_unsqueeze_view_47 = async_compile.triton('triton_poi_fused_clone_split_transpose_unsqueeze_view_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_split_transpose_unsqueeze_view_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_unsqueeze_view_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + 1024*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/jk/cjkqqwj3ffnozuajowuvxabved5qe5ecdhcqaz654ui25gub43ow.py
# Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_5 => view_71
#   out_12 => clone_15
#   transpose_23 => permute_41
# Graph fragment:
#   %bmm_13 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %view_71 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_13, [128, 8, 64, 64]), kwargs = {})
#   %permute_41 : Tensor "bf16[128, 64, 8, 64][32768, 64, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_71, [0, 2, 1, 3]), kwargs = {})
#   %clone_15 : Tensor "bf16[128, 64, 8, 64][32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_15
triton_poi_fused_clone_transpose_view_48 = async_compile.triton('triton_poi_fused_clone_transpose_view_48', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_view_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/6s/c6siu6svnkyloxlxj3kejndt5yla72wpya2dqomcqhrcpkzvoyz4.py
# Topologically Sorted Source Nodes: [transpose_24, out_13, x_14], Original ATen: [aten.transpose, aten.view, aten.add]
# Source node to ATen node mapping:
#   out_13 => view_75
#   transpose_24 => permute_43
#   x_14 => add_18
# Graph fragment:
#   %convolution_10 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %getitem_30 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=getitem_30]
#   %permute_43 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_30, [0, 2, 1]), kwargs = {})
#   %view_75 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [128, 512, 8, 8]), kwargs = {})
#   %add_18 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %view_75), kwargs = {})
#   return %add_18
triton_poi_fused_add_transpose_view_49 = async_compile.triton('triton_poi_fused_add_transpose_view_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_transpose_view_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33554432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_transpose_view_49(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/oe/coezc5zsiocsw577jrxnoqqrvtyh6lttxkmsjerr35yohklmsjak.py
# Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_15 => convert_element_type_173
# Graph fragment:
#   %primals_59 : Tensor "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=primals_59]
#   %convert_element_type_173 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_59, torch.bfloat16), kwargs = {})
#   return %convert_element_type_173
triton_poi_fused__to_copy_50 = async_compile.triton('triton_poi_fused__to_copy_50', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/gz/cgz7d64ghdhku5sllxgby7ktyrojnaic46tdcvjnpaiaprlyxnl3.py
# Topologically Sorted Source Nodes: [x_norm_4], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   x_norm_4 => add_19, clone_16, convert_element_type_174, rsqrt_4, var_mean_4, view_76
# Graph fragment:
#   %convolution_11 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_11]
#   %buf183 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf183]
#   %convert_element_type_174 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_16 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [128, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_19 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   return %getitem_33,%buf183,%rsqrt_4
triton_red_fused__to_copy_clone_native_group_norm_51 = async_compile.triton('triton_red_fused__to_copy_clone_native_group_norm_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_native_group_norm_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_clone_native_group_norm_51(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
    tmp9 = 1024.0
    tmp10 = (tmp7 / tmp9)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.store(out_ptr2 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/vs/cvsltj2uhxzvqovs6inv6zt4s5biy5utkehnlhgdijymvhtm2o5d.py
# Topologically Sorted Source Nodes: [x_norm_4, x_flat_4, v_t_x_4], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   v_t_x_4 => convert_element_type_175
#   x_flat_4 => view_81
#   x_norm_4 => add_19, clone_16, convert_element_type_174, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
# Graph fragment:
#   %convolution_11 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_11]
#   %getitem_33 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_33]
#   %buf183 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf183]
#   %convert_element_type_175 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0" = PlaceHolder[target=convert_element_type_175]
#   %convert_element_type_174 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_16 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [128, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_19 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %sub_7 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_33), kwargs = {})
#   %mul_22 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [128, 512, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[128, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [128, 512, -1]), kwargs = {})
#   %convert_element_type_175 : Tensor "bf16[128, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_81, torch.bfloat16), kwargs = {})
#   %permute_395 : Tensor "bf16[128, 64, 512][32768, 1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_175, [0, 2, 1]), kwargs = {})
#   return %convert_element_type_175,%permute_395
triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_52 = async_compile.triton('triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 16777216, 'x': 25165824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_52(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 512*y0 + 4096*(((y0 % 8)) // 8) + 32768*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*y1 + (x2 // 16)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*y1 + (x2 // 16)), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2 + 512*y3), tmp11, xmask)
    tl.store(out_ptr1 + (y0 + 64*x2 + 32768*y1), tmp11, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/qu/cqu6o6kqftip3yl6bbyvtuivrek7il4thq2nyfjsoatksqbggyx5.py
# Topologically Sorted Source Nodes: [x_norm_4, split_4, shift_9, x_flat_4, out_14, view_39, out_15, x_16, x_17], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   out_14 => add_20
#   out_15 => add_21
#   shift_9 => view_80
#   split_4 => split_with_sizes_4
#   view_39 => view_82
#   x_16 => mul_23, sigmoid_13
#   x_17 => convert_element_type_180, convert_element_type_182, convolution_12
#   x_flat_4 => view_81
#   x_norm_4 => add_19, clone_16, convert_element_type_174, mul_22, rsqrt_4, sub_7, var_mean_4, view_76, view_77
# Graph fragment:
#   %convolution_11 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_11]
#   %getitem_33 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=getitem_33]
#   %buf183 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf183]
#   %bmm_15 : Tensor "bf16[128, 512, 64][32768, 64, 1]cuda:0" = PlaceHolder[target=bmm_15]
#   %addmm_9 : Tensor "bf16[128, 12800][12800, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %add_21 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=add_21]
#   %convert_element_type_182 : Tensor "bf16[128, 512, 8, 8][32768, 64, 8, 1]cuda:0" = PlaceHolder[target=convert_element_type_182]
#   %convert_element_type_174 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %clone_16 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_174,), kwargs = {memory_format: torch.contiguous_format})
#   %view_76 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [128, 32, 16, 64]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_76, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_19 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %sub_7 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_76, %getitem_33), kwargs = {})
#   %mul_22 : Tensor "f32[128, 32, 16, 64][32768, 1024, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %view_77 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_22, [128, 512, 8, 8]), kwargs = {})
#   %split_with_sizes_4 : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%addmm_9, [6144, 6144, 512], 1), kwargs = {})
#   %view_80 : Tensor "bf16[128, 512, 1, 1][12800, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_36, [128, 512, 1, 1]), kwargs = {})
#   %view_81 : Tensor "f32[128, 512, 64][32768, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_77, [128, 512, -1]), kwargs = {})
#   %add_20 : Tensor "f32[128, 512, 64][32768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_81, %bmm_15), kwargs = {})
#   %view_82 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_20, [128, 512, 8, 8]), kwargs = {})
#   %add_21 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_82, %view_80), kwargs = {})
#   %sigmoid_13 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_23 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_13), kwargs = {})
#   %convert_element_type_180 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_62, torch.bfloat16), kwargs = {})
#   %convert_element_type_182 : Tensor "bf16[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
#   %convolution_12 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_182, %convert_element_type_181, %convert_element_type_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %full_default_26 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([128, 512, 8, 8], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_88 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_26, %sigmoid_13), kwargs = {})
#   %mul_350 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sub_88), kwargs = {})
#   %add_178 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_350, 1), kwargs = {})
#   %mul_351 : Tensor "f32[128, 512, 8, 8][32768, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_13, %add_178), kwargs = {})
#   return %add_21,%convert_element_type_182,%mul_351,%buf193
triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_53 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 58720256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 512)
    x2 = xindex // 32768
    x4 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512*x0 + 32768*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x4 // 16), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x4 // 16), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (12288 + x1 + 12800*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1024.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp17
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22 + tmp20
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp24, None)
    tl.store(out_ptr3 + (x1 + 512*x0 + 32768*x2), tmp19, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ld/cldckj3yhy6pcopu542velbwpkn5pjmzxd4bavbl5k76xsr26kek.py
# Topologically Sorted Source Nodes: [x_17, add_17, h_7], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
# Source node to ATen node mapping:
#   add_17 => add_22
#   h_7 => convert_element_type_183, convert_element_type_184, mul_24, sigmoid_14
#   x_17 => convert_element_type_180, convolution_12
# Graph fragment:
#   %buf194 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf194]
#   %buf192 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf192]
#   %convolution_12 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_12]
#   %convolution_10 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %convert_element_type_180 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_62, torch.bfloat16), kwargs = {})
#   %convolution_12 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_182, %convert_element_type_181, %convert_element_type_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_22 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_10), kwargs = {})
#   %convert_element_type_183 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.float32), kwargs = {})
#   %sigmoid_14 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_183,), kwargs = {})
#   %mul_24 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_183, %sigmoid_14), kwargs = {})
#   %convert_element_type_184 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
#   return %convolution_12,%convert_element_type_184
triton_poi_fused__to_copy_add_convolution_silu_54 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_silu_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_silu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50332672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_silu_54(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/cw/ccw2fykmuhxrhohxgdxojvaujckrjzfzwep6eaymy7ktqylkji6c.py
# Topologically Sorted Source Nodes: [x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.view, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_21 => add_27
#   h_8 => convert_element_type_213, convert_element_type_214, mul_28, sigmoid_16
#   qkv => convert_element_type_218
#   view_50 => view_106
#   x_21 => convert_element_type_210, convolution_14
#   x_flat_6 => permute_56
#   x_norm_6 => add_28, add_29, convert_element_type_215, mul_29, mul_30, rsqrt_6, sub_10, var_mean_6
# Graph fragment:
#   %buf231 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=buf231]
#   %buf229 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=buf229]
#   %convert_element_type_184 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convert_element_type_184]
#   %add_27 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %buf234 : Tensor "f32[128, 64, 1][64, 1, 8192]cuda:0" = PlaceHolder[target=buf234]
#   %getitem_47 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_47]
#   %rsqrt_6 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %primals_71 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_71]
#   %primals_72 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_72]
#   %convert_element_type_210 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_70, torch.bfloat16), kwargs = {})
#   %convolution_14 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_212, %convert_element_type_211, %convert_element_type_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_27 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convert_element_type_184), kwargs = {})
#   %convert_element_type_213 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_28 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [128, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_215 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_56, torch.float32), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_215, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_28 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %sub_10 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_215, %getitem_47), kwargs = {})
#   %mul_29 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_6), kwargs = {})
#   %mul_30 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %primals_71), kwargs = {})
#   %add_29 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %primals_72), kwargs = {})
#   %convert_element_type_218 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_29, torch.bfloat16), kwargs = {})
#   return %add_27,%getitem_47,%buf234,%rsqrt_6,%convert_element_type_218
triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_55 = async_compile.triton('triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_55', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_55', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 50336768}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_55(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp10 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp23 = 512.0
    tmp24 = (tmp22 / tmp23)
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp9 - tmp17
    tmp29 = tmp28 * tmp27
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp27, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp34, None)
    tl.store(out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/aw/cawqjnrhbadlfxm6wsoi5c3xno5nfmqu2ginq7nnbnq2t72ule5y.py
# Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   qkv => convert_element_type_217, permute_57
# Graph fragment:
#   %primals_73 : Tensor "f32[1536, 512][512, 1]cuda:0" = PlaceHolder[target=primals_73]
#   %convert_element_type_217 : Tensor "bf16[1536, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_73, torch.bfloat16), kwargs = {})
#   %permute_57 : Tensor "bf16[512, 1536][1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_217, [1, 0]), kwargs = {})
#   return %permute_57
triton_poi_fused__to_copy_t_56 = async_compile.triton('triton_poi_fused__to_copy_t_56', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/uf/cufm2i4zi425mkrny6xr7njcsyyqjdjrvwsdwnhbxabyidz75feu.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   matmul_8 => clone_22
#   q_12 => select
#   qkv => add_tensor_1, convert_element_type_216, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
# Graph fragment:
#   %mm_default_1 : Tensor "bf16[8192, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_74 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=primals_74]
#   %convert_element_type_216 : Tensor "bf16[1536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_74, torch.bfloat16), kwargs = {})
#   %add_tensor_1 : Tensor "bf16[8192, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %convert_element_type_216), kwargs = {})
#   %view_108 : Tensor "bf16[128, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [128, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "bf16[128, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [128, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "bf16[3, 128, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select : Tensor "bf16[128, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 0), kwargs = {})
#   %clone_22 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_16,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_22
triton_poi_fused__to_copy_addmm_clone_permute_select_view_57 = async_compile.triton('triton_poi_fused__to_copy_addmm_clone_permute_select_view_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_clone_permute_select_view_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25167872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_clone_permute_select_view_57(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/cc/cccwmzgvf4o7lmphebqzrjfqu7g3w255ggmu5rg3pwtyarebdiyu.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   k_8 => select_1
#   matmul_8 => clone_23
#   qkv => add_tensor_1, convert_element_type_216, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   transpose_35 => permute_59
# Graph fragment:
#   %mm_default_1 : Tensor "bf16[8192, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_74 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=primals_74]
#   %convert_element_type_216 : Tensor "bf16[1536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_74, torch.bfloat16), kwargs = {})
#   %add_tensor_1 : Tensor "bf16[8192, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %convert_element_type_216), kwargs = {})
#   %view_108 : Tensor "bf16[128, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [128, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "bf16[128, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [128, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "bf16[3, 128, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_1 : Tensor "bf16[128, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 1), kwargs = {})
#   %permute_59 : Tensor "bf16[128, 8, 64, 64][98304, 64, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_23 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_17,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_23
triton_poi_fused__to_copy_addmm_clone_permute_select_transpose_view_58 = async_compile.triton('triton_poi_fused__to_copy_addmm_clone_permute_select_transpose_view_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_clone_permute_select_transpose_view_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 8390656, 'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_clone_permute_select_transpose_view_58(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/zi/czi67unynxhm2fkxzd3zys335maqacplblyyfpvht7xbhbmjtd4h.py
# Topologically Sorted Source Nodes: [matmul_8, attn_9, out_20], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
# Source node to ATen node mapping:
#   attn_9 => div_5, exp_5, sum_5
#   matmul_8 => view_112
#   out_20 => convert_element_type_225
# Graph fragment:
#   %bmm_20 : Tensor "bf16[1024, 64, 64][4096, 64, 1]cuda:0" = PlaceHolder[target=bmm_20]
#   %amax_default_5 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0" = PlaceHolder[target=amax_default_5]
#   %sum_5 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0" = PlaceHolder[target=sum_5]
#   %view_112 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_20, [128, 8, 64, 64]), kwargs = {})
#   %convert_element_type_default_31 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_112, torch.float32), kwargs = {})
#   %mul_tensor_10 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_31, 1), kwargs = {})
#   %amax_default_5 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_10, [-1], True), kwargs = {})
#   %sub_tensor_5 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_10, %amax_default_5), kwargs = {})
#   %mul_tensor_11 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_5, 0.125), kwargs = {})
#   %exp_5 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_11,), kwargs = {})
#   %sum_5 : Tensor "f32[128, 8, 64, 1][512, 64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1], True), kwargs = {})
#   %div_5 : Tensor "f32[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_5), kwargs = {})
#   %convert_element_type_225 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_5, torch.bfloat16), kwargs = {})
#   return %amax_default_5,%sum_5,%expand_18
triton_per_fused__softmax__to_copy_amax_mul_sub_view_59 = async_compile.triton('triton_per_fused__softmax__to_copy_amax_mul_sub_view_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_amax_mul_sub_view_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576, 'r0_': 25165824}}
)
@triton.jit
def triton_per_fused__softmax__to_copy_amax_mul_sub_view_59(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = triton_helpers.max2(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = tmp3 - tmp6
    tmp8 = 0.125
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = (tmp10 / tmp13)
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 64*x0), tmp15, None)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/2d/c2d3nribqa736d265wx7lmvblv2km2y7d3bfz3rx4d5vgemwmhhz.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
# Source node to ATen node mapping:
#   out_20 => clone_24
#   qkv => add_tensor_1, convert_element_type_216, view_108
#   qkv_1 => view_109
#   qkv_2 => permute_58
#   v_20 => select_2
# Graph fragment:
#   %mm_default_1 : Tensor "bf16[8192, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_74 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=primals_74]
#   %convert_element_type_216 : Tensor "bf16[1536][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_74, torch.bfloat16), kwargs = {})
#   %add_tensor_1 : Tensor "bf16[8192, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %convert_element_type_216), kwargs = {})
#   %view_108 : Tensor "bf16[128, 64, 1536][98304, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [128, 64, 1536]), kwargs = {})
#   %view_109 : Tensor "bf16[128, 64, 3, 8, 64][98304, 1536, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_108, [128, 64, 3, 8, 64]), kwargs = {})
#   %permute_58 : Tensor "bf16[3, 128, 8, 64, 64][512, 98304, 64, 1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_109, [2, 0, 3, 1, 4]), kwargs = {})
#   %select_2 : Tensor "bf16[128, 8, 64, 64][98304, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%permute_58, 0, 2), kwargs = {})
#   %clone_24 : Tensor "bf16[128, 8, 64, 64][32768, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_19,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_24
triton_poi_fused__to_copy_addmm_clone_permute_select_view_60 = async_compile.triton('triton_poi_fused__to_copy_addmm_clone_permute_select_view_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_clone_permute_select_view_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25167872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_clone_permute_select_view_60(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/jw/cjwkuynkslq6yl4szerwpcfmnxr3c564kb65qrlk33txadjobb7m.py
# Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_8 => convert_element_type_213, convert_element_type_214, mul_28, sigmoid_16
#   h_9 => convert_element_type_236
#   out_22 => add_tensor, convert_element_type_228, view_118
#   out_23 => add_30
#   out_24 => add_31, add_32, convert_element_type_233, mul_32, mul_33, rsqrt_7, sub_12, var_mean_7
#   out_25 => view_119
#   transpose_37 => permute_62
#   view_50 => view_106
#   x_flat_6 => permute_56
# Graph fragment:
#   %add_27 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=add_27]
#   %mm_default : Tensor "bf16[8192, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %primals_76 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_76]
#   %add_30 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0" = PlaceHolder[target=add_30]
#   %buf253 : Tensor "f32[128, 64, 1][64, 1, 8192]cuda:0" = PlaceHolder[target=buf253]
#   %getitem_49 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_49]
#   %rsqrt_7 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_7]
#   %primals_77 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_77]
#   %primals_78 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_78]
#   %convert_element_type_213 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %sigmoid_16 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_213,), kwargs = {})
#   %mul_28 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_213, %sigmoid_16), kwargs = {})
#   %convert_element_type_214 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
#   %view_106 : Tensor "bf16[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_214, [128, 512, 64]), kwargs = {})
#   %permute_56 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_106, [0, 2, 1]), kwargs = {})
#   %convert_element_type_228 : Tensor "bf16[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_76, torch.bfloat16), kwargs = {})
#   %add_tensor : Tensor "bf16[8192, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %convert_element_type_228), kwargs = {})
#   %view_118 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [128, 64, 512]), kwargs = {})
#   %add_30 : Tensor "bf16[128, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_56, %view_118), kwargs = {})
#   %convert_element_type_233 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_233, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_31 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[128, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %sub_12 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_233, %getitem_49), kwargs = {})
#   %mul_32 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_7), kwargs = {})
#   %mul_33 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %primals_77), kwargs = {})
#   %add_32 : Tensor "f32[128, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %primals_78), kwargs = {})
#   %permute_62 : Tensor "f32[128, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_119 : Tensor "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [128, 512, 8, 8]), kwargs = {})
#   %convert_element_type_236 : Tensor "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_119, torch.bfloat16), kwargs = {})
#   return %add_30,%getitem_49,%buf253,%rsqrt_7,%convert_element_type_236
triton_per_fused__to_copy_add_addmm_native_layer_norm_silu_transpose_view_61 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_silu_transpose_view_61', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_silu_transpose_view_61', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 50337792}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_silu_transpose_view_61(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp11 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp24 = 512.0
    tmp25 = (tmp23 / tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp10 - tmp18
    tmp30 = tmp29 * tmp28
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp9, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp28, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp35, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/6n/c6n5dl5gnaq6i3s24zbtb5acat2yletin5znlh5mrtrece4vvlnm.py
# Topologically Sorted Source Nodes: [h_9], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   h_9 => convert_element_type_235
# Graph fragment:
#   %primals_79 : Tensor "f32[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0" = PlaceHolder[target=primals_79]
#   %convert_element_type_235 : Tensor "bf16[512, 256, 4, 4][4096, 1, 1024, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_79, torch.bfloat16), kwargs = {})
#   return %convert_element_type_235
triton_poi_fused__to_copy_62 = async_compile.triton('triton_poi_fused__to_copy_62', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_62(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/mg/cmgz6avacsvpz4fpfxk6tqazneoe4dqepr5xjplpfauyqkgzcpzg.py
# Topologically Sorted Source Nodes: [h_9, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten._to_copy, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   gate => unsqueeze_7
#   h_10 => add_33
#   h_16_gated => mul_35
#   h_9 => convert_element_type_234, convolution_15
#   input_27 => sigmoid_18
#   unsqueeze_4 => unsqueeze_6
# Graph fragment:
#   %buf259 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf259]
#   %buf258 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf258]
#   %convert_element_type_152 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convert_element_type_152]
#   %addmm_17 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %convert_element_type_234 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_80, torch.bfloat16), kwargs = {})
#   %convolution_15 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_236, %convert_element_type_235, %convert_element_type_234, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %sigmoid_18 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_17,), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[128, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_18, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %mul_35 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_152, %unsqueeze_7), kwargs = {})
#   %add_33 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %mul_35), kwargs = {})
#   return %add_33
triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_63 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_63', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67174912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 65536
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0 + 256*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/na/cnaqcclcr52q6mxwsthqimvgmswnqb5l7eartz2xgjun5bmwd7s5.py
# Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_40 => convert_element_type_409
# Graph fragment:
#   %primals_129 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=primals_129]
#   %convert_element_type_409 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_129, torch.bfloat16), kwargs = {})
#   return %convert_element_type_409
triton_poi_fused__to_copy_64 = async_compile.triton('triton_poi_fused__to_copy_64', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/i7/ci762456nqcwcexb5ucbesg5bpyrbxtyc73g7bpilos2wlekc75p.py
# Topologically Sorted Source Nodes: [h_16, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten._to_copy, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   gate_1 => unsqueeze_14
#   h_16 => convert_element_type_399, convolution_26
#   h_17 => add_59
#   h_32_gated => mul_57
#   input_41 => sigmoid_30
#   unsqueeze_11 => unsqueeze_13
# Graph fragment:
#   %buf450 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf450]
#   %buf449 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf449]
#   %convert_element_type_90 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_90]
#   %addmm_24 : Tensor "bf16[128, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_24]
#   %convert_element_type_399 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_126, torch.bfloat16), kwargs = {})
#   %convolution_26 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_398, %convert_element_type_400, %convert_element_type_399, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %sigmoid_30 : Tensor "bf16[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_24,), kwargs = {})
#   %unsqueeze_13 : Tensor "bf16[128, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sigmoid_30, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_57 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_90, %unsqueeze_14), kwargs = {})
#   %add_59 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %mul_57), kwargs = {})
#   return %add_59
triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_65 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134250752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = tmp2 + tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ud/cudebx4cp3mm7jidu4o2g5mzx5hylkrjfns4cdu4elgabwbp6xwm.py
# Topologically Sorted Source Nodes: [x_50, add_53, h_20], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.fill, aten.sigmoid, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   add_53 => add_71
#   h_20 => convert_element_type_450, convert_element_type_451, mul_66, sigmoid_36
#   x_50 => convert_element_type_447, convolution_32
# Graph fragment:
#   %buf511 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf511]
#   %buf509 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf509]
#   %convert_element_type_438 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_438]
#   %convert_element_type_447 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_142, torch.bfloat16), kwargs = {})
#   %convolution_32 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_449, %convert_element_type_448, %convert_element_type_447, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_71 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_32, %convert_element_type_438), kwargs = {})
#   %convert_element_type_450 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.float32), kwargs = {})
#   %sigmoid_36 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_450,), kwargs = {})
#   %mul_66 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_450, %sigmoid_36), kwargs = {})
#   %convert_element_type_451 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_66, torch.bfloat16), kwargs = {})
#   %full_1 : Tensor "bf16[128, 32, 32, 128][131072, 4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 32, 32, 128], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_129 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.permute.default](args = (%full_1, [0, 3, 1, 2]), kwargs = {})
#   %sigmoid_47 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_71,), kwargs = {})
#   %sub_41 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_129, %sigmoid_47), kwargs = {})
#   %mul_131 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_71, %sub_41), kwargs = {})
#   %add_98 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_131, 1), kwargs = {})
#   %mul_132 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_47, %add_98), kwargs = {})
#   return %convert_element_type_451,%mul_132
triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_silu_sub_66 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_silu_sub_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_silu_sub_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201326848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_silu_sub_66(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.sigmoid(tmp4)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp4 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/j2/cj2lxlub75ou6xhkobwlqjjhesvgaitqakhlimktganli34omvig.py
# Topologically Sorted Source Nodes: [x_56, add_59], Original ATen: [aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_59 => add_79
#   x_56 => convert_element_type_473, convolution_36
# Graph fragment:
#   %buf547 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf547]
#   %buf545 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf545]
#   %convert_element_type_464 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_464]
#   %convert_element_type_473 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_150, torch.bfloat16), kwargs = {})
#   %convolution_36 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_475, %convert_element_type_474, %convert_element_type_473, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_79 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_36, %convert_element_type_464), kwargs = {})
#   return %add_79
triton_poi_fused__to_copy_add_convolution_67 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134217984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_67(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/do/cdoiab6j5sno27xlmhqkzqfhq3wwsv5bzxh4rfv7rs4eow6torvz.py
# Topologically Sorted Source Nodes: [h_22, input_42], Original ATen: [aten.silu, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   h_22 => convert_element_type_476, mul_72, sigmoid_40
#   input_42 => add_80, clone_56, rsqrt_18, var_mean_18, view_270
# Graph fragment:
#   %add_79 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_79]
#   %buf550 : Tensor "f32[128, 32, 1, 1][32, 1, 4096, 4096]cuda:0" = PlaceHolder[target=buf550]
#   %convert_element_type_476 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_72 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_56 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_72,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_56, [128, 32, 4, 1024]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_270, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_80 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_120, 1e-05), kwargs = {})
#   %rsqrt_18 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   return %getitem_121,%buf550,%rsqrt_18
triton_red_fused_clone_native_group_norm_silu_68 = async_compile.triton('triton_red_fused_clone_native_group_norm_silu_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_silu_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 65536, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_silu_68(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
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
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tl.store(out_ptr0 + (x4), tmp5, None)
    tmp11 = 4096.0
    tmp12 = (tmp9 / tmp11)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/xf/cxffe7kahw2b5hem25wcsjl4llmu4keg2an6snv4ctkgmp7z3ijy.py
# Topologically Sorted Source Nodes: [h_22, input_42, input_43, input_44], Original ATen: [aten.silu, aten.clone, aten.native_group_norm, aten._to_copy]
# Source node to ATen node mapping:
#   h_22 => convert_element_type_476, mul_72, sigmoid_40
#   input_42 => add_81, clone_56, mul_73, mul_74, sub_28, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, view_270, view_271
#   input_43 => mul_75, sigmoid_41
#   input_44 => convert_element_type_481
# Graph fragment:
#   %add_79 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_79]
#   %getitem_121 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=getitem_121]
#   %rsqrt_18 : Tensor "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_18]
#   %primals_151 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_151]
#   %primals_152 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_152]
#   %add_81 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=add_81]
#   %convert_element_type_476 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %sigmoid_40 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_476,), kwargs = {})
#   %mul_72 : Tensor "f32[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_476, %sigmoid_40), kwargs = {})
#   %clone_56 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%mul_72,), kwargs = {memory_format: torch.contiguous_format})
#   %view_270 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_56, [128, 32, 4, 1024]), kwargs = {})
#   %sub_28 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_270, %getitem_121), kwargs = {})
#   %mul_73 : Tensor "f32[128, 32, 4, 1024][131072, 4096, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_18), kwargs = {})
#   %view_271 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_73, [128, 128, 32, 32]), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_151, 0), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, 2), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 3), kwargs = {})
#   %mul_74 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_271, %unsqueeze_17), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_152, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 3), kwargs = {})
#   %add_81 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_20), kwargs = {})
#   %sigmoid_41 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_75 : Tensor "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_41), kwargs = {})
#   %convert_element_type_481 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.bfloat16), kwargs = {})
#   return %add_81,%convert_element_type_481
triton_poi_fused__to_copy_clone_native_group_norm_silu_69 = async_compile.triton('triton_poi_fused__to_copy_clone_native_group_norm_silu_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_native_group_norm_silu_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 33555456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_native_group_norm_silu_69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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
    tmp4 = tl.load(in_ptr1 + (32*y1 + (x2 // 4)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (32*y1 + (x2 // 4)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr1 + (y0 + 1024*x2 + 131072*y1), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/ao/caovo3shry4uge2z6xd4nybmach6s6er76kxqe2uuhc2ug3mhlv7.py
# Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_44 => convert_element_type_479, convolution_37
# Graph fragment:
#   %primals_154 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=primals_154]
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_154, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[128, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf556
triton_poi_fused__to_copy_convolution_70 = async_compile.triton('triton_poi_fused__to_copy_convolution_70', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_70', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_70(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/te/cteqzitssldqolqalw6fylyexuvalyhmafbi6itepz7tcuue2qq3.py
# Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_44 => convert_element_type_479, convolution_37
# Graph fragment:
#   %convert_element_type_481 : Tensor "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = PlaceHolder[target=convert_element_type_481]
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_154, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[128, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf557
triton_poi_fused__to_copy_convolution_71 = async_compile.triton('triton_poi_fused__to_copy_convolution_71', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_71', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 33554432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_71(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 131072*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/uc/cucwa4i6rlg22xnce53hphoiykuhrmbezlly4mkfz3vcghmagbka.py
# Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_44 => convert_element_type_479, convolution_37
# Graph fragment:
#   %buf558 : Tensor "bf16[128, 4, 32, 32][4096, 1, 128, 4]cuda:0" = PlaceHolder[target=buf558]
#   %buf556 : Tensor "bf16[4][1]cuda:0" = PlaceHolder[target=buf556]
#   %convert_element_type_479 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_154, torch.bfloat16), kwargs = {})
#   %convolution_37 : Tensor "bf16[128, 4, 32, 32][4096, 1, 128, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_481, %convert_element_type_480, %convert_element_type_479, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_37
triton_poi_fused__to_copy_convolution_72 = async_compile.triton('triton_poi_fused__to_copy_convolution_72', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_72', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_72(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/kz/ckznuirqrz5qu7mf3zl4yyrfy657yjmziuczhmeou3w2aapnpdhi.py
# Topologically Sorted Source Nodes: [x_47, add_50], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   add_50 => add_67
#   x_47 => convert_element_type_434, convolution_30
# Graph fragment:
#   %buf493 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=buf493]
#   %buf491 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=buf491]
#   %convert_element_type_425 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convert_element_type_425]
#   %convert_element_type_434 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_138, torch.bfloat16), kwargs = {})
#   %convolution_30 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_436, %convert_element_type_435, %convert_element_type_434, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_67 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convert_element_type_425), kwargs = {})
#   %full_1 : Tensor "bf16[128, 32, 32, 128][131072, 4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 32, 32, 128], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_129 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.permute.default](args = (%full_1, [0, 3, 1, 2]), kwargs = {})
#   %sigmoid_49 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_67,), kwargs = {})
#   %sub_45 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_129, %sigmoid_49), kwargs = {})
#   %mul_149 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %sub_45), kwargs = {})
#   %add_105 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_149, 1), kwargs = {})
#   %mul_150 : Tensor "bf16[128, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_49, %add_105), kwargs = {})
#   return %mul_150
triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_73 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_73', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134217984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_73(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = 1.0
    tmp7 = tmp6 - tmp5
    tmp8 = tmp4 * tmp7
    tmp9 = tmp8 + tmp6
    tmp10 = tmp5 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /mnt/g/GitHub/VAE_ca_proj/Thermal/.compile_cache/pu/cpuxb6pusfn6kiqwldkgevtt7omc7ev54a4bl5fk5w4z5ajg4vyq.py
# Topologically Sorted Source Nodes: [x_41, add_43], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.sigmoid, aten.fill, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   add_43 => add_58
#   x_41 => convert_element_type_394, convolution_25
# Graph fragment:
#   %buf446 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf446]
#   %buf444 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=buf444]
#   %convert_element_type_368 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convert_element_type_368]
#   %convert_element_type_394 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_124, torch.bfloat16), kwargs = {})
#   %convolution_25 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_396, %convert_element_type_395, %convert_element_type_394, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convert_element_type_368), kwargs = {})
#   %sigmoid_54 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_58,), kwargs = {})
#   %full_17 : Tensor "bf16[128, 16, 16, 256][65536, 4096, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 16, 16, 256], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %permute_178 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.permute.default](args = (%full_17, [0, 3, 1, 2]), kwargs = {})
#   %sub_55 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_178, %sigmoid_54), kwargs = {})
#   %mul_192 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %sub_55), kwargs = {})
#   %add_120 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_192, 1), kwargs = {})
#   %mul_193 : Tensor "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_54, %add_120), kwargs = {})
#   return %mul_193
triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67109376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = 1.0
    tmp7 = tmp6 - tmp5
    tmp8 = tmp4 * tmp7
    tmp9 = tmp8 + tmp6
    tmp10 = tmp5 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154 = args
        args.clear()
        assert_size_stride(primals_1, (128, ), (1, ))
        assert_size_stride(primals_2, (1024, 256), (256, 1))
        assert_size_stride(primals_3, (1024, ), (1, ))
        assert_size_stride(primals_4, (256, 1024), (1024, 1))
        assert_size_stride(primals_5, (256, ), (1, ))
        assert_size_stride(primals_6, (4, 256), (256, 1))
        assert_size_stride(primals_7, (128, ), (1, ))
        assert_size_stride(primals_8, (512, 512), (512, 1))
        assert_size_stride(primals_9, (512, ), (1, ))
        assert_size_stride(primals_10, (256, 512), (512, 1))
        assert_size_stride(primals_11, (256, ), (1, ))
        assert_size_stride(primals_12, (256, 256), (256, 1))
        assert_size_stride(primals_13, (256, ), (1, ))
        assert_size_stride(primals_14, (3200, 256), (256, 1))
        assert_size_stride(primals_15, (3200, ), (1, ))
        assert_size_stride(primals_16, (256, 256), (256, 1))
        assert_size_stride(primals_17, (256, ), (1, ))
        assert_size_stride(primals_18, (6400, 256), (256, 1))
        assert_size_stride(primals_19, (6400, ), (1, ))
        assert_size_stride(primals_20, (256, 256), (256, 1))
        assert_size_stride(primals_21, (256, ), (1, ))
        assert_size_stride(primals_22, (12800, 256), (256, 1))
        assert_size_stride(primals_23, (12800, ), (1, ))
        assert_size_stride(primals_24, (128, 4, 3, 3), (36, 1, 12, 4))
        assert_size_stride(primals_25, (128, ), (1, ))
        assert_size_stride(primals_26, (128, 4, 32, 32), (4096, 1, 128, 4))
        assert_size_stride(primals_27, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_28, (128, ), (1, ))
        assert_size_stride(primals_29, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(primals_30, (128, ), (1, ))
        assert_size_stride(primals_31, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_32, (128, ), (1, ))
        assert_size_stride(primals_33, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(primals_34, (128, ), (1, ))
        assert_size_stride(primals_35, (256, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_36, (256, ), (1, ))
        assert_size_stride(primals_37, (256, 256), (256, 1))
        assert_size_stride(primals_38, (512, 256), (256, 1))
        assert_size_stride(primals_39, (256, 256), (256, 1))
        assert_size_stride(primals_40, (256, ), (1, ))
        assert_size_stride(primals_41, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_42, (256, ), (1, ))
        assert_size_stride(primals_43, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(primals_44, (256, ), (1, ))
        assert_size_stride(primals_45, (256, 256), (256, 1))
        assert_size_stride(primals_46, (512, 256), (256, 1))
        assert_size_stride(primals_47, (256, 256), (256, 1))
        assert_size_stride(primals_48, (256, ), (1, ))
        assert_size_stride(primals_49, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_50, (256, ), (1, ))
        assert_size_stride(primals_51, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(primals_52, (256, ), (1, ))
        assert_size_stride(primals_53, (512, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_54, (512, ), (1, ))
        assert_size_stride(primals_55, (512, 512), (512, 1))
        assert_size_stride(primals_56, (1024, 256), (256, 1))
        assert_size_stride(primals_57, (512, 512), (512, 1))
        assert_size_stride(primals_58, (512, ), (1, ))
        assert_size_stride(primals_59, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(primals_60, (512, ), (1, ))
        assert_size_stride(primals_61, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(primals_62, (512, ), (1, ))
        assert_size_stride(primals_63, (512, 512), (512, 1))
        assert_size_stride(primals_64, (1024, 256), (256, 1))
        assert_size_stride(primals_65, (512, 512), (512, 1))
        assert_size_stride(primals_66, (512, ), (1, ))
        assert_size_stride(primals_67, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(primals_68, (512, ), (1, ))
        assert_size_stride(primals_69, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(primals_70, (512, ), (1, ))
        assert_size_stride(primals_71, (512, ), (1, ))
        assert_size_stride(primals_72, (512, ), (1, ))
        assert_size_stride(primals_73, (1536, 512), (512, 1))
        assert_size_stride(primals_74, (1536, ), (1, ))
        assert_size_stride(primals_75, (512, 512), (512, 1))
        assert_size_stride(primals_76, (512, ), (1, ))
        assert_size_stride(primals_77, (512, ), (1, ))
        assert_size_stride(primals_78, (512, ), (1, ))
        assert_size_stride(primals_79, (512, 256, 4, 4), (4096, 1, 1024, 256))
        assert_size_stride(primals_80, (256, ), (1, ))
        assert_size_stride(primals_81, (256, 256), (256, 1))
        assert_size_stride(primals_82, (256, ), (1, ))
        assert_size_stride(primals_83, (256, 256), (256, 1))
        assert_size_stride(primals_84, (256, ), (1, ))
        assert_size_stride(primals_85, (256, 256), (256, 1))
        assert_size_stride(primals_86, (512, 256), (256, 1))
        assert_size_stride(primals_87, (256, 256), (256, 1))
        assert_size_stride(primals_88, (256, ), (1, ))
        assert_size_stride(primals_89, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_90, (256, ), (1, ))
        assert_size_stride(primals_91, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(primals_92, (256, ), (1, ))
        assert_size_stride(primals_93, (256, 256), (256, 1))
        assert_size_stride(primals_94, (512, 256), (256, 1))
        assert_size_stride(primals_95, (256, 256), (256, 1))
        assert_size_stride(primals_96, (256, ), (1, ))
        assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_98, (256, ), (1, ))
        assert_size_stride(primals_99, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(primals_100, (256, ), (1, ))
        assert_size_stride(primals_101, (256, 256), (256, 1))
        assert_size_stride(primals_102, (512, 256), (256, 1))
        assert_size_stride(primals_103, (256, 256), (256, 1))
        assert_size_stride(primals_104, (256, ), (1, ))
        assert_size_stride(primals_105, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_106, (256, ), (1, ))
        assert_size_stride(primals_107, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(primals_108, (256, ), (1, ))
        assert_size_stride(primals_109, (256, 256), (256, 1))
        assert_size_stride(primals_110, (512, 256), (256, 1))
        assert_size_stride(primals_111, (256, 256), (256, 1))
        assert_size_stride(primals_112, (256, ), (1, ))
        assert_size_stride(primals_113, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_114, (256, ), (1, ))
        assert_size_stride(primals_115, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(primals_116, (256, ), (1, ))
        assert_size_stride(primals_117, (256, 256), (256, 1))
        assert_size_stride(primals_118, (512, 256), (256, 1))
        assert_size_stride(primals_119, (256, 256), (256, 1))
        assert_size_stride(primals_120, (256, ), (1, ))
        assert_size_stride(primals_121, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_122, (256, ), (1, ))
        assert_size_stride(primals_123, (256, 256, 1, 1), (256, 1, 256, 256))
        assert_size_stride(primals_124, (256, ), (1, ))
        assert_size_stride(primals_125, (256, 128, 4, 4), (2048, 1, 512, 128))
        assert_size_stride(primals_126, (128, ), (1, ))
        assert_size_stride(primals_127, (256, 256), (256, 1))
        assert_size_stride(primals_128, (256, ), (1, ))
        assert_size_stride(primals_129, (128, 256), (256, 1))
        assert_size_stride(primals_130, (128, ), (1, ))
        assert_size_stride(primals_131, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_132, (128, ), (1, ))
        assert_size_stride(primals_133, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(primals_134, (128, ), (1, ))
        assert_size_stride(primals_135, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_136, (128, ), (1, ))
        assert_size_stride(primals_137, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(primals_138, (128, ), (1, ))
        assert_size_stride(primals_139, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_140, (128, ), (1, ))
        assert_size_stride(primals_141, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(primals_142, (128, ), (1, ))
        assert_size_stride(primals_143, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_144, (128, ), (1, ))
        assert_size_stride(primals_145, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(primals_146, (128, ), (1, ))
        assert_size_stride(primals_147, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_148, (128, ), (1, ))
        assert_size_stride(primals_149, (128, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(primals_150, (128, ), (1, ))
        assert_size_stride(primals_151, (128, ), (1, ))
        assert_size_stride(primals_152, (128, ), (1, ))
        assert_size_stride(primals_153, (4, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_154, (4, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [arange, mul, truediv, freqs, getitem, getitem_1, args, cos, sin, emb, input_1], Original ATen: [aten.arange, aten.mul, aten.div, aten.exp, aten.unsqueeze, aten.cos, aten.sin, aten.cat, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_0.run(primals_1, buf0, 32768, stream=stream0)
            del primals_1
            buf1 = empty_strided_cuda((1024, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_2, buf1, 262144, stream=stream0)
            del primals_2
            buf2 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_3, buf2, 1024, stream=stream0)
            del primals_3
            buf3 = empty_strided_cuda((128, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf2, buf0, reinterpret_tensor(buf1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf3)
            del buf2
            buf4 = empty_strided_cuda((128, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_silu_3.run(buf3, buf4, 131072, stream=stream0)
            buf5 = reinterpret_tensor(buf1, (256, 1024), (1024, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_4, buf5, 262144, stream=stream0)
            del primals_4
            buf6 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_5, buf6, 256, stream=stream0)
            del primals_5
            buf7 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf6, buf4, reinterpret_tensor(buf5, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf7)
            buf8 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [s_emb, cat_1, input_4], Original ATen: [aten.embedding, aten.cat, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_cat_embedding_5.run(buf7, primals_7, primals_6, buf8, 65536, stream=stream0)
            del primals_6
            buf9 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_8, buf9, 262144, stream=stream0)
            del primals_8
            buf10 = empty_strided_cuda((512, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_9, buf10, 512, stream=stream0)
            del primals_9
            buf11 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf10, buf8, reinterpret_tensor(buf9, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf11)
            buf12 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_silu_7.run(buf11, buf12, 65536, stream=stream0)
            buf13 = empty_strided_cuda((256, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_10, buf13, 131072, stream=stream0)
            del primals_10
            buf14 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_11, buf14, 256, stream=stream0)
            del primals_11
            buf15 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf14, buf12, reinterpret_tensor(buf13, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf15)
            buf16 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_12, buf16, 65536, stream=stream0)
            del primals_12
            buf17 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_13, buf17, 256, stream=stream0)
            del primals_13
            buf18 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf17, buf15, reinterpret_tensor(buf16, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf18)
            buf19 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf18, buf19, 32768, stream=stream0)
            buf20 = empty_strided_cuda((3200, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_11.run(primals_14, buf20, 819200, stream=stream0)
            del primals_14
            buf21 = empty_strided_cuda((3200, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(primals_15, buf21, 3200, stream=stream0)
            del primals_15
            buf22 = empty_strided_cuda((128, 3200), (3200, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf21, buf19, reinterpret_tensor(buf20, (256, 3200), (1, 256), 0), alpha=1, beta=1, out=buf22)
            del buf21
            buf23 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_16, buf23, 65536, stream=stream0)
            del primals_16
            buf24 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_17, buf24, 256, stream=stream0)
            del primals_17
            buf25 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf24, buf15, reinterpret_tensor(buf23, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf25)
            buf26 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf25, buf26, 32768, stream=stream0)
            buf27 = empty_strided_cuda((6400, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(primals_18, buf27, 1638400, stream=stream0)
            del primals_18
            buf28 = empty_strided_cuda((6400, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_14.run(primals_19, buf28, 6400, stream=stream0)
            del primals_19
            buf29 = empty_strided_cuda((128, 6400), (6400, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf28, buf26, reinterpret_tensor(buf27, (256, 6400), (1, 256), 0), alpha=1, beta=1, out=buf29)
            del buf28
            buf30 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_20, buf30, 65536, stream=stream0)
            del primals_20
            buf31 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_21, buf31, 256, stream=stream0)
            del primals_21
            buf32 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf31, buf15, reinterpret_tensor(buf30, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf32)
            buf33 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf32, buf33, 32768, stream=stream0)
            buf34 = empty_strided_cuda((12800, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_15.run(primals_22, buf34, 3276800, stream=stream0)
            del primals_22
            buf35 = empty_strided_cuda((12800, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_16.run(primals_23, buf35, 12800, stream=stream0)
            del primals_23
            buf36 = empty_strided_cuda((128, 12800), (12800, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf35, buf33, reinterpret_tensor(buf34, (256, 12800), (1, 256), 0), alpha=1, beta=1, out=buf36)
            del buf35
            buf37 = empty_strided_cuda((128, 4, 3, 3), (36, 1, 12, 4), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_17.run(primals_24, buf37, 4608, stream=stream0)
            del primals_24
            buf38 = empty_strided_cuda((128, 4, 32, 32), (4096, 1, 128, 4), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(primals_26, buf38, 524288, stream=stream0)
            del primals_26
            buf39 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_25, buf39, 128, stream=stream0)
            del primals_25
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            buf40 = extern_kernels.convolution(buf38, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf40, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [h], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf41, buf39, 16777216, stream=stream0)
            buf42 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(primals_27, buf42, 147456, stream=stream0)
            del primals_27
            buf43 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_28, buf43, 128, stream=stream0)
            del primals_28
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            buf44 = extern_kernels.convolution(buf41, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf44, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf45 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf45, buf43, 16777216, stream=stream0)
            buf46 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf47 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf49 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_22.run(buf45, buf46, buf47, buf49, 4096, 4096, stream=stream0)
            buf50 = empty_strided_cuda((128, 128, 1024), (131072, 1, 128), torch.bfloat16)
            buf602 = empty_strided_cuda((128, 1024, 128), (131072, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm, x_flat, v_t_x], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23.run(buf45, buf46, buf47, buf50, buf602, 131072, 128, stream=stream0)
            buf51 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, v_1, transpose, v_t_x], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 1536), buf50, out=buf51)
            buf52 = reinterpret_tensor(buf50, (128, 128, 1024), (131072, 1024, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 0), buf51, out=buf52)
            buf55 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            buf601 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
            buf57 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm, split, shift_1, x_flat, out, view_4, out_1, x_1, x_2], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24.run(buf45, buf46, buf47, buf52, buf22, buf55, buf601, buf57, 16777216, stream=stream0)
            buf54 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(primals_29, buf54, 16384, stream=stream0)
            del primals_29
            buf56 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_30, buf56, 128, stream=stream0)
            del primals_30
            # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._to_copy, aten.convolution]
            buf58 = extern_kernels.convolution(buf57, buf54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf58, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf59 = buf58; del buf58  # reuse
            buf60 = buf57; del buf57  # reuse
            # Topologically Sorted Source Nodes: [x_2, add_2, h_1], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_26.run(buf59, buf56, buf41, buf60, 16777216, stream=stream0)
            buf61 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(primals_31, buf61, 147456, stream=stream0)
            del primals_31
            buf62 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_32, buf62, 128, stream=stream0)
            del primals_32
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.convolution]
            buf63 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf63, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf64 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf64, buf62, 16777216, stream=stream0)
            buf65 = buf47; del buf47  # reuse
            buf66 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf68 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_1], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_22.run(buf64, buf65, buf66, buf68, 4096, 4096, stream=stream0)
            buf69 = reinterpret_tensor(buf52, (128, 128, 1024), (131072, 1, 128), 0); del buf52  # reuse
            buf600 = empty_strided_cuda((128, 1024, 128), (131072, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_1, x_flat_1, v_t_x_1], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23.run(buf64, buf65, buf66, buf69, buf600, 131072, 128, stream=stream0)
            buf70 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, v_1, transpose, v_t_x_1], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 1536), buf69, out=buf70)
            buf71 = reinterpret_tensor(buf69, (128, 128, 1024), (131072, 1024, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed_1], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 0), buf70, out=buf71)
            buf74 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            buf599 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
            buf76 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, shift_1, x_norm_1, x_flat_1, out_2, view_9, out_3, x_4, x_5], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24.run(buf64, buf65, buf66, buf71, buf22, buf74, buf599, buf76, 16777216, stream=stream0)
            buf73 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(primals_33, buf73, 16384, stream=stream0)
            del primals_33
            buf75 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_34, buf75, 128, stream=stream0)
            del primals_34
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._to_copy, aten.convolution]
            buf77 = extern_kernels.convolution(buf76, buf73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf77, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf78 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [x_5, add_5, h_2], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_27.run(buf77, buf75, buf60, buf78, 16777216, stream=stream0)
            buf79 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_28.run(primals_35, buf79, 294912, stream=stream0)
            del primals_35
            buf80 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_36, buf80, 256, stream=stream0)
            del primals_36
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            buf81 = extern_kernels.convolution(buf78, buf79, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf81, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf82 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf82, buf80, 8388608, stream=stream0)
            buf83 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_1], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_37, buf83, 65536, stream=stream0)
            del primals_37
            buf84 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_10, q, q_1], Original ATen: [aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf82, (32768, 256), (256, 1), 0), buf83, out=buf84)
            buf85 = empty_strided_cuda((256, 512), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_38, buf85, 131072, stream=stream0)
            del primals_38
            buf86 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf85, out=buf86)
            buf87 = empty_strided_cuda((128, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_1, view_11, q_2, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf84, buf87, 8388608, stream=stream0)
            buf88 = empty_strided_cuda((128, 4, 64, 1), (256, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv, chunk, view_12, k_1, transpose_6, matmul], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf86, buf88, 32768, stream=stream0)
            buf89 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_1, kv, chunk, view_11, q_2, view_12, k_1, transpose_6, matmul], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf87, (512, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf88, (512, 64, 1), (64, 1, 0), 0), out=buf89)
            buf90 = empty_strided_cuda((128, 4, 256, 1), (1024, 256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, attn_1, matmul_1], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32.run(buf89, buf90, 131072, stream=stream0)
            buf91 = empty_strided_cuda((128, 4, 1, 64), (256, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf86, buf91, 32768, stream=stream0)
            buf92 = reinterpret_tensor(buf84, (512, 256, 64), (16384, 64, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [kv, chunk, view_13, v_5, matmul, attn_1, matmul_1], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf90, (512, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf91, (512, 1, 64), (64, 0, 1), 0), out=buf92)
            buf93 = empty_strided_cuda((128, 256, 4, 64), (65536, 256, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_34.run(buf92, buf93, 8388608, stream=stream0)
            buf94 = reinterpret_tensor(buf86, (256, 256), (256, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_39, buf94, 65536, stream=stream0)
            del primals_39
            buf95 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_40, buf95, 256, stream=stream0)
            del primals_40
            buf96 = reinterpret_tensor(buf92, (32768, 256), (256, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [matmul_1, transpose_7, out_4, input_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf95, reinterpret_tensor(buf93, (32768, 256), (256, 1), 0), reinterpret_tensor(buf94, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf96)
            # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.view, aten.native_dropout]
            buf97 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf96, (128, 256, 256), (65536, 256, 1), 0), 0.1, True)
            buf98 = buf97[0]
            assert_size_stride(buf98, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf98, 16, 'torch.ops.aten.native_dropout.default')
            buf99 = buf97[1]
            assert_size_stride(buf99, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf99, 16, 'torch.ops.aten.native_dropout.default')
            del buf97
            buf100 = reinterpret_tensor(buf98, (128, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [transpose_8, out_5, x_6], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_35.run(buf100, buf82, 8388608, stream=stream0)
            buf101 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(primals_41, buf101, 589824, stream=stream0)
            del primals_41
            buf102 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_42, buf102, 256, stream=stream0)
            del primals_42
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._to_copy, aten.convolution]
            buf103 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf103, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf104 = buf103; del buf103  # reuse
            # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf104, buf102, 8388608, stream=stream0)
            buf105 = buf66; del buf66  # reuse
            buf106 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf108 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_2], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_37.run(buf104, buf105, buf106, buf108, 4096, 2048, stream=stream0)
            buf109 = reinterpret_tensor(buf96, (128, 256, 256), (65536, 1, 256), 0); del buf96  # reuse
            buf597 = empty_strided_cuda((128, 256, 256), (65536, 1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_2, x_flat_2, v_t_x_2], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38.run(buf104, buf105, buf106, buf109, buf597, 32768, 256, stream=stream0)
            buf110 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, v_7, transpose_9, v_t_x_2], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 3072), buf109, out=buf110)
            buf111 = reinterpret_tensor(buf109, (128, 256, 256), (65536, 256, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_2], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 0), buf110, out=buf111)
            buf114 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            buf596 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
            buf116 = empty_strided_cuda((128, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_2, split_2, shift_5, x_flat_2, out_6, view_19, out_7, x_8, x_9], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39.run(buf104, buf105, buf106, buf111, buf29, buf114, buf596, buf116, 8388608, stream=stream0)
            buf113 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_43, buf113, 65536, stream=stream0)
            del primals_43
            buf115 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_44, buf115, 256, stream=stream0)
            del primals_44
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy, aten.convolution]
            buf117 = extern_kernels.convolution(buf116, buf113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf117, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf118 = buf117; del buf117  # reuse
            buf120 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [x_9, add_9, h_4], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_40.run(buf118, buf115, buf82, buf120, 8388608, stream=stream0)
            buf119 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_4], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_45, buf119, 65536, stream=stream0)
            del primals_45
            buf121 = reinterpret_tensor(buf111, (32768, 256), (256, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [add_9, h_4, view_20, q_3, q_4], Original ATen: [aten.add, aten.silu, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf120, (32768, 256), (256, 1), 0), buf119, out=buf121)
            buf122 = empty_strided_cuda((256, 512), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_46, buf122, 131072, stream=stream0)
            del primals_46
            buf123 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf122, out=buf123)
            buf124 = empty_strided_cuda((128, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_4, view_21, q_5, matmul_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf121, buf124, 8388608, stream=stream0)
            buf125 = empty_strided_cuda((128, 4, 64, 1), (256, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_22, k_3, transpose_14, matmul_2], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf123, buf125, 32768, stream=stream0)
            buf126 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_4, kv_1, chunk_1, view_21, q_5, view_22, k_3, transpose_14, matmul_2], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf124, (512, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf125, (512, 64, 1), (64, 1, 0), 0), out=buf126)
            buf127 = empty_strided_cuda((128, 4, 256, 1), (1024, 256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_2, attn_3, matmul_3], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32.run(buf126, buf127, 131072, stream=stream0)
            buf128 = empty_strided_cuda((128, 4, 1, 64), (256, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_23, v_9, matmul_3], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf123, buf128, 32768, stream=stream0)
            buf129 = reinterpret_tensor(buf121, (512, 256, 64), (16384, 64, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [kv_1, chunk_1, view_23, v_9, matmul_2, attn_3, matmul_3], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf127, (512, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf128, (512, 1, 64), (64, 0, 1), 0), out=buf129)
            buf130 = empty_strided_cuda((128, 256, 4, 64), (65536, 256, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_34.run(buf129, buf130, 8388608, stream=stream0)
            buf131 = reinterpret_tensor(buf123, (256, 256), (256, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_47, buf131, 65536, stream=stream0)
            del primals_47
            buf132 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_48, buf132, 256, stream=stream0)
            del primals_48
            buf133 = reinterpret_tensor(buf129, (32768, 256), (256, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [matmul_3, transpose_15, out_8, input_18], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf132, reinterpret_tensor(buf130, (32768, 256), (256, 1), 0), reinterpret_tensor(buf131, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf133)
            # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten.view, aten.native_dropout]
            buf134 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf133, (128, 256, 256), (65536, 256, 1), 0), 0.1, True)
            buf135 = buf134[0]
            assert_size_stride(buf135, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf135, 16, 'torch.ops.aten.native_dropout.default')
            buf136 = buf134[1]
            assert_size_stride(buf136, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf136, 16, 'torch.ops.aten.native_dropout.default')
            del buf134
            buf137 = reinterpret_tensor(buf135, (128, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [transpose_16, out_9, x_10], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_35.run(buf137, buf120, 8388608, stream=stream0)
            buf138 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(primals_49, buf138, 589824, stream=stream0)
            del primals_49
            buf139 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_50, buf139, 256, stream=stream0)
            del primals_50
            # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._to_copy, aten.convolution]
            buf140 = extern_kernels.convolution(buf137, buf138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf140, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf141 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf141, buf139, 8388608, stream=stream0)
            buf142 = buf106; del buf106  # reuse
            buf143 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf145 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_3], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_37.run(buf141, buf142, buf143, buf145, 4096, 2048, stream=stream0)
            buf146 = reinterpret_tensor(buf133, (128, 256, 256), (65536, 1, 256), 0); del buf133  # reuse
            buf595 = empty_strided_cuda((128, 256, 256), (65536, 1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_3, x_flat_3, v_t_x_3], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38.run(buf141, buf142, buf143, buf146, buf595, 32768, 256, stream=stream0)
            buf147 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, v_7, transpose_9, v_t_x_3], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 3072), buf146, out=buf147)
            buf148 = reinterpret_tensor(buf146, (128, 256, 256), (65536, 256, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_3], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 0), buf147, out=buf148)
            buf151 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            buf594 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
            buf153 = empty_strided_cuda((128, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, shift_5, x_norm_3, x_flat_3, out_10, view_29, out_11, x_12, x_13], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39.run(buf141, buf142, buf143, buf148, buf29, buf151, buf594, buf153, 8388608, stream=stream0)
            buf150 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_51, buf150, 65536, stream=stream0)
            del primals_51
            buf152 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_52, buf152, 256, stream=stream0)
            del primals_52
            # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten._to_copy, aten.convolution]
            buf154 = extern_kernels.convolution(buf153, buf150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf154, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf155 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [x_13, add_13, h_5], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_41.run(buf154, buf152, buf120, buf155, 8388608, stream=stream0)
            buf156 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(primals_53, buf156, 1179648, stream=stream0)
            del primals_53
            buf157 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_54, buf157, 512, stream=stream0)
            del primals_54
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            buf158 = extern_kernels.convolution(buf155, buf156, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf158, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf159 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [h_6], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_43.run(buf159, buf157, 4194304, stream=stream0)
            buf160 = empty_strided_cuda((512, 512), (1, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_7], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_55, buf160, 262144, stream=stream0)
            del primals_55
            buf161 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_30, q_6, q_7], Original ATen: [aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf159, (8192, 512), (512, 1), 0), buf160, out=buf161)
            buf162 = empty_strided_cuda((256, 1024), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_56, buf162, 262144, stream=stream0)
            del primals_56
            buf163 = empty_strided_cuda((128, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf162, out=buf163)
            buf164 = empty_strided_cuda((128, 8, 64, 64), (32768, 4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_7, view_31, q_8, matmul_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_44.run(buf161, buf164, 4194304, stream=stream0)
            buf165 = empty_strided_cuda((128, 8, 64, 1), (512, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_45.run(buf163, buf165, 65536, stream=stream0)
            buf166 = empty_strided_cuda((1024, 64, 1), (64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_7, kv_2, chunk_2, view_31, q_8, view_32, k_5, transpose_22, matmul_4], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf164, (1024, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf165, (1024, 64, 1), (64, 1, 0), 0), out=buf166)
            buf167 = empty_strided_cuda((128, 8, 64, 1), (512, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_4, attn_5, matmul_5], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_46.run(buf166, buf167, 65536, stream=stream0)
            buf168 = empty_strided_cuda((128, 8, 1, 64), (512, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_47.run(buf163, buf168, 65536, stream=stream0)
            buf169 = reinterpret_tensor(buf161, (1024, 64, 64), (4096, 64, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [kv_2, chunk_2, view_33, v_13, matmul_4, attn_5, matmul_5], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf167, (1024, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf168, (1024, 1, 64), (64, 0, 1), 0), out=buf169)
            buf170 = empty_strided_cuda((128, 64, 8, 64), (32768, 512, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_48.run(buf169, buf170, 4194304, stream=stream0)
            buf171 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_57, buf171, 262144, stream=stream0)
            del primals_57
            buf172 = buf157; del buf157  # reuse
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_58, buf172, 512, stream=stream0)
            del primals_58
            buf173 = reinterpret_tensor(buf169, (8192, 512), (512, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [matmul_5, transpose_23, out_12, input_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf172, reinterpret_tensor(buf170, (8192, 512), (512, 1), 0), reinterpret_tensor(buf171, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf173)
            # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.view, aten.native_dropout]
            buf174 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf173, (128, 64, 512), (32768, 512, 1), 0), 0.1, True)
            buf175 = buf174[0]
            assert_size_stride(buf175, (128, 64, 512), (32768, 512, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf175, 16, 'torch.ops.aten.native_dropout.default')
            buf176 = buf174[1]
            assert_size_stride(buf176, (128, 64, 512), (32768, 512, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf176, 16, 'torch.ops.aten.native_dropout.default')
            del buf174
            buf177 = reinterpret_tensor(buf175, (128, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf175  # reuse
            # Topologically Sorted Source Nodes: [transpose_24, out_13, x_14], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_49.run(buf177, buf159, 4194304, stream=stream0)
            buf178 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_50.run(primals_59, buf178, 2359296, stream=stream0)
            del primals_59
            buf179 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_60, buf179, 512, stream=stream0)
            del primals_60
            # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._to_copy, aten.convolution]
            buf180 = extern_kernels.convolution(buf177, buf178, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf180, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf181 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_43.run(buf181, buf179, 4194304, stream=stream0)
            buf182 = buf143; del buf143  # reuse
            buf183 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf185 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_4], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_51.run(buf181, buf182, buf183, buf185, 4096, 1024, stream=stream0)
            buf186 = reinterpret_tensor(buf173, (128, 512, 64), (32768, 1, 512), 0); del buf173  # reuse
            buf592 = empty_strided_cuda((128, 64, 512), (32768, 1, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_4, x_flat_4, v_t_x_4], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_52.run(buf181, buf182, buf183, buf186, buf592, 8192, 512, stream=stream0)
            buf187 = empty_strided_cuda((128, 12, 64), (768, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_4, v_15, transpose_25, v_t_x_4], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf36, (128, 12, 512), (12800, 1, 12), 6144), buf186, out=buf187)
            buf188 = reinterpret_tensor(buf186, (128, 512, 64), (32768, 64, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [split_4, u_9, mixed_4], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf36, (128, 512, 12), (12800, 12, 1), 0), buf187, out=buf188)
            buf191 = empty_strided_cuda((128, 512, 8, 8), (32768, 64, 8, 1), torch.bfloat16)
            buf591 = empty_strided_cuda((128, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
            buf193 = empty_strided_cuda((128, 512, 8, 8), (32768, 1, 4096, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_4, split_4, shift_9, x_flat_4, out_14, view_39, out_15, x_16, x_17], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.split_with_sizes, aten.view, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_53.run(buf181, buf182, buf183, buf188, buf36, buf191, buf591, buf193, 4194304, stream=stream0)
            buf190 = empty_strided_cuda((512, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_61, buf190, 262144, stream=stream0)
            del primals_61
            buf192 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_62, buf192, 512, stream=stream0)
            del primals_62
            # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._to_copy, aten.convolution]
            buf194 = extern_kernels.convolution(buf193, buf190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf194, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf195 = buf194; del buf194  # reuse
            buf197 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [x_17, add_17, h_7], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_54.run(buf195, buf192, buf159, buf197, 4194304, stream=stream0)
            buf196 = empty_strided_cuda((512, 512), (1, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_10], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_63, buf196, 262144, stream=stream0)
            del primals_63
            buf198 = reinterpret_tensor(buf188, (8192, 512), (512, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [add_17, h_7, view_40, q_9, q_10], Original ATen: [aten.add, aten.silu, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (8192, 512), (512, 1), 0), buf196, out=buf198)
            buf199 = empty_strided_cuda((256, 1024), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_64, buf199, 262144, stream=stream0)
            del primals_64
            buf200 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf199, out=buf200)
            buf201 = empty_strided_cuda((128, 8, 64, 64), (32768, 4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_10, view_41, q_11, matmul_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_44.run(buf198, buf201, 4194304, stream=stream0)
            buf202 = empty_strided_cuda((128, 8, 64, 1), (512, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_42, k_7, transpose_30, matmul_6], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_45.run(buf200, buf202, 65536, stream=stream0)
            buf203 = empty_strided_cuda((1024, 64, 1), (64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_10, kv_3, chunk_3, view_41, q_11, view_42, k_7, transpose_30, matmul_6], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf201, (1024, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf202, (1024, 64, 1), (64, 1, 0), 0), out=buf203)
            buf204 = empty_strided_cuda((128, 8, 64, 1), (512, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_6, attn_7, matmul_7], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_46.run(buf203, buf204, 65536, stream=stream0)
            buf205 = empty_strided_cuda((128, 8, 1, 64), (512, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_43, v_17, matmul_7], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_47.run(buf200, buf205, 65536, stream=stream0)
            buf206 = reinterpret_tensor(buf198, (1024, 64, 64), (4096, 64, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [kv_3, chunk_3, view_43, v_17, matmul_6, attn_7, matmul_7], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf204, (1024, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf205, (1024, 1, 64), (64, 0, 1), 0), out=buf206)
            buf207 = empty_strided_cuda((128, 64, 8, 64), (32768, 512, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_48.run(buf206, buf207, 4194304, stream=stream0)
            buf208 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_65, buf208, 262144, stream=stream0)
            del primals_65
            buf209 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_66, buf209, 512, stream=stream0)
            del primals_66
            buf210 = reinterpret_tensor(buf206, (8192, 512), (512, 1), 0); del buf206  # reuse
            # Topologically Sorted Source Nodes: [matmul_7, transpose_31, out_16, input_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf209, reinterpret_tensor(buf207, (8192, 512), (512, 1), 0), reinterpret_tensor(buf208, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf210)
            # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.view, aten.native_dropout]
            buf211 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf210, (128, 64, 512), (32768, 512, 1), 0), 0.1, True)
            buf212 = buf211[0]
            assert_size_stride(buf212, (128, 64, 512), (32768, 512, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf212, 16, 'torch.ops.aten.native_dropout.default')
            buf213 = buf211[1]
            assert_size_stride(buf213, (128, 64, 512), (32768, 512, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf213, 16, 'torch.ops.aten.native_dropout.default')
            del buf211
            buf214 = reinterpret_tensor(buf212, (128, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [transpose_32, out_17, x_18], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_49.run(buf214, buf197, 4194304, stream=stream0)
            buf215 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_50.run(primals_67, buf215, 2359296, stream=stream0)
            del primals_67
            buf216 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_68, buf216, 512, stream=stream0)
            del primals_68
            # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten._to_copy, aten.convolution]
            buf217 = extern_kernels.convolution(buf214, buf215, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf217, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf218 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_43.run(buf218, buf216, 4194304, stream=stream0)
            buf219 = buf183; del buf183  # reuse
            buf220 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf222 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_5], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_51.run(buf218, buf219, buf220, buf222, 4096, 1024, stream=stream0)
            buf223 = reinterpret_tensor(buf210, (128, 512, 64), (32768, 1, 512), 0); del buf210  # reuse
            buf590 = empty_strided_cuda((128, 64, 512), (32768, 1, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_5, x_flat_5, v_t_x_5], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_52.run(buf218, buf219, buf220, buf223, buf590, 8192, 512, stream=stream0)
            buf224 = empty_strided_cuda((128, 12, 64), (768, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_4, v_15, transpose_25, v_t_x_5], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf36, (128, 12, 512), (12800, 1, 12), 6144), buf223, out=buf224)
            buf225 = reinterpret_tensor(buf223, (128, 512, 64), (32768, 64, 1), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [split_4, u_9, mixed_5], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf36, (128, 512, 12), (12800, 12, 1), 0), buf224, out=buf225)
            buf228 = empty_strided_cuda((128, 512, 8, 8), (32768, 64, 8, 1), torch.bfloat16)
            buf589 = empty_strided_cuda((128, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
            buf230 = empty_strided_cuda((128, 512, 8, 8), (32768, 1, 4096, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_4, shift_9, x_norm_5, x_flat_5, out_18, view_49, out_19, x_20, x_21], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_53.run(buf218, buf219, buf220, buf225, buf36, buf228, buf589, buf230, 4194304, stream=stream0)
            buf227 = empty_strided_cuda((512, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_69, buf227, 262144, stream=stream0)
            del primals_69
            buf229 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_6.run(primals_70, buf229, 512, stream=stream0)
            del primals_70
            # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten._to_copy, aten.convolution]
            buf231 = extern_kernels.convolution(buf230, buf227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf231, (128, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf232 = buf231; del buf231  # reuse
            buf233 = empty_strided_cuda((128, 64, 1), (64, 1, 1), torch.float32)
            buf234 = empty_strided_cuda((128, 64, 1), (64, 1, 8192), torch.float32)
            buf236 = reinterpret_tensor(buf234, (128, 64, 1), (64, 1, 1), 0); del buf234  # reuse
            buf237 = reinterpret_tensor(buf230, (128, 64, 512), (32768, 512, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [x_21, add_21, h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.view, aten.transpose, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_convolution_native_layer_norm_silu_transpose_view_55.run(buf232, buf236, buf229, buf197, primals_71, primals_72, buf233, buf237, 8192, 512, stream=stream0)
            del buf229
            del primals_72
            buf238 = empty_strided_cuda((512, 1536), (1, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_56.run(primals_73, buf238, 786432, stream=stream0)
            del primals_73
            buf239 = empty_strided_cuda((8192, 1536), (1536, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, x_norm_6, qkv], Original ATen: [aten.silu, aten.view, aten.transpose, aten._to_copy, aten.native_layer_norm, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf237, (8192, 512), (512, 1), 0), buf238, out=buf239)
            buf240 = reinterpret_tensor(buf225, (128, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_clone_permute_select_view_57.run(buf239, primals_74, buf240, 4194304, stream=stream0)
            buf241 = empty_strided_cuda((128, 8, 64, 64), (32768, 4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, k_8, transpose_35, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_clone_permute_select_transpose_view_58.run(buf239, primals_74, buf241, 65536, 64, stream=stream0)
            buf242 = empty_strided_cuda((1024, 64, 64), (4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, q_12, k_8, transpose_35, matmul_8], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf240, (1024, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf241, (1024, 64, 64), (4096, 64, 1), 0), out=buf242)
            buf243 = empty_strided_cuda((128, 8, 64, 1), (512, 64, 1, 1), torch.float32)
            buf244 = empty_strided_cuda((128, 8, 64, 1), (512, 64, 1, 1), torch.float32)
            buf245 = empty_strided_cuda((128, 8, 64, 64), (32768, 4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_8, attn_9, out_20], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_amax_mul_sub_view_59.run(buf242, buf243, buf244, buf245, 65536, 64, stream=stream0)
            buf246 = empty_strided_cuda((128, 8, 64, 64), (32768, 4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, out_20], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_addmm_clone_permute_select_view_60.run(buf239, primals_74, buf246, 4194304, stream=stream0)
            del buf239
            del primals_74
            buf247 = empty_strided_cuda((1024, 64, 64), (4096, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, qkv_2, v_20, matmul_8, attn_9, out_20], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.permute, aten.select, aten.mul, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf245, (1024, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf246, (1024, 64, 64), (4096, 64, 1), 0), out=buf247)
            buf248 = empty_strided_cuda((128, 64, 8, 64), (32768, 512, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_48.run(buf247, buf248, 4194304, stream=stream0)
            buf249 = empty_strided_cuda((512, 512), (1, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_75, buf249, 262144, stream=stream0)
            del primals_75
            buf250 = reinterpret_tensor(buf247, (8192, 512), (512, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [out_20, transpose_36, out_21, out_22], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf248, (8192, 512), (512, 1), 0), buf249, out=buf250)
            buf251 = reinterpret_tensor(buf250, (128, 64, 512), (32768, 512, 1), 0); del buf250  # reuse
            buf252 = empty_strided_cuda((128, 64, 1), (64, 1, 1), torch.float32)
            buf253 = empty_strided_cuda((128, 64, 1), (64, 1, 8192), torch.float32)
            buf255 = reinterpret_tensor(buf253, (128, 64, 1), (64, 1, 1), 0); del buf253  # reuse
            buf257 = empty_strided_cuda((128, 512, 8, 8), (32768, 1, 4096, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_8, view_50, x_flat_6, out_22, out_23, out_24, transpose_37, out_25, h_9], Original ATen: [aten.silu, aten.view, aten.transpose, aten._to_copy, aten.addmm, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_addmm_native_layer_norm_silu_transpose_view_61.run(buf251, buf255, buf232, primals_76, primals_77, primals_78, buf252, buf257, 8192, 512, stream=stream0)
            del primals_76
            del primals_78
            buf256 = empty_strided_cuda((512, 256, 4, 4), (4096, 1, 1024, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_9], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_62.run(primals_79, buf256, 2097152, stream=stream0)
            del primals_79
            buf258 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_9], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_80, buf258, 256, stream=stream0)
            del primals_80
            # Topologically Sorted Source Nodes: [h_9], Original ATen: [aten._to_copy, aten.convolution]
            buf259 = extern_kernels.convolution(buf257, buf256, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf259, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf260 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_81, buf260, 65536, stream=stream0)
            del primals_81
            buf261 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_82, buf261, 256, stream=stream0)
            del primals_82
            buf262 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf261, buf15, reinterpret_tensor(buf260, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf262)
            buf263 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf262, buf263, 32768, stream=stream0)
            buf264 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_83, buf264, 65536, stream=stream0)
            del primals_83
            buf265 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_84, buf265, 256, stream=stream0)
            del primals_84
            buf266 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf265, buf263, reinterpret_tensor(buf264, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf266)
            buf267 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_14], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_85, buf267, 65536, stream=stream0)
            del primals_85
            buf268 = buf259; del buf259  # reuse
            # Topologically Sorted Source Nodes: [h_9, input_27, unsqueeze_4, gate, h_16_gated, h_10], Original ATen: [aten._to_copy, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_63.run(buf268, buf258, buf155, buf266, 8388608, stream=stream0)
            buf269 = reinterpret_tensor(buf148, (32768, 256), (256, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [h_9, input_27, unsqueeze_4, gate, h_16_gated, h_10, view_52, q_13, q_14], Original ATen: [aten._to_copy, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf268, (32768, 256), (256, 1), 0), buf267, out=buf269)
            buf270 = reinterpret_tensor(buf200, (256, 512), (1, 256), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_86, buf270, 131072, stream=stream0)
            del primals_86
            buf271 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf270, out=buf271)
            buf272 = empty_strided_cuda((128, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_14, view_53, q_15, matmul_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf269, buf272, 8388608, stream=stream0)
            buf273 = empty_strided_cuda((128, 4, 64, 1), (256, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_54, k_10, transpose_42, matmul_10], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf271, buf273, 32768, stream=stream0)
            buf274 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_14, kv_4, chunk_4, view_53, q_15, view_54, k_10, transpose_42, matmul_10], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf272, (512, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf273, (512, 64, 1), (64, 1, 0), 0), out=buf274)
            buf275 = empty_strided_cuda((128, 4, 256, 1), (1024, 256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_10, attn_11, matmul_11], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32.run(buf274, buf275, 131072, stream=stream0)
            buf276 = empty_strided_cuda((128, 4, 1, 64), (256, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_55, v_22, matmul_11], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf271, buf276, 32768, stream=stream0)
            buf277 = reinterpret_tensor(buf269, (512, 256, 64), (16384, 64, 1), 0); del buf269  # reuse
            # Topologically Sorted Source Nodes: [kv_4, chunk_4, view_55, v_22, matmul_10, attn_11, matmul_11], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf275, (512, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf276, (512, 1, 64), (64, 0, 1), 0), out=buf277)
            buf278 = empty_strided_cuda((128, 256, 4, 64), (65536, 256, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_34.run(buf277, buf278, 8388608, stream=stream0)
            buf279 = reinterpret_tensor(buf271, (256, 256), (256, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_87, buf279, 65536, stream=stream0)
            del primals_87
            buf280 = buf258; del buf258  # reuse
            # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_88, buf280, 256, stream=stream0)
            del primals_88
            buf281 = reinterpret_tensor(buf277, (32768, 256), (256, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [matmul_11, transpose_43, out_26, input_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf280, reinterpret_tensor(buf278, (32768, 256), (256, 1), 0), reinterpret_tensor(buf279, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf281)
            # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.view, aten.native_dropout]
            buf282 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf281, (128, 256, 256), (65536, 256, 1), 0), 0.1, True)
            buf283 = buf282[0]
            assert_size_stride(buf283, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf283, 16, 'torch.ops.aten.native_dropout.default')
            buf284 = buf282[1]
            assert_size_stride(buf284, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf284, 16, 'torch.ops.aten.native_dropout.default')
            del buf282
            buf285 = reinterpret_tensor(buf283, (128, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [transpose_44, out_27, x_22], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_35.run(buf285, buf268, 8388608, stream=stream0)
            buf286 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(primals_89, buf286, 589824, stream=stream0)
            del primals_89
            buf287 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_90, buf287, 256, stream=stream0)
            del primals_90
            # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._to_copy, aten.convolution]
            buf288 = extern_kernels.convolution(buf285, buf286, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf288, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf289 = buf288; del buf288  # reuse
            # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf289, buf287, 8388608, stream=stream0)
            buf290 = buf220; del buf220  # reuse
            buf291 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf293 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_7], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_37.run(buf289, buf290, buf291, buf293, 4096, 2048, stream=stream0)
            buf294 = reinterpret_tensor(buf281, (128, 256, 256), (65536, 1, 256), 0); del buf281  # reuse
            buf588 = empty_strided_cuda((128, 256, 256), (65536, 1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_7, x_flat_7, v_t_x_6], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38.run(buf289, buf290, buf291, buf294, buf588, 32768, 256, stream=stream0)
            buf295 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, v_7, transpose_9, v_t_x_6], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 3072), buf294, out=buf295)
            buf296 = reinterpret_tensor(buf294, (128, 256, 256), (65536, 256, 1), 0); del buf294  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_6], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 0), buf295, out=buf296)
            buf299 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            buf587 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
            buf301 = empty_strided_cuda((128, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, shift_5, x_norm_7, x_flat_7, out_28, view_61, out_29, x_24, x_25], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39.run(buf289, buf290, buf291, buf296, buf29, buf299, buf587, buf301, 8388608, stream=stream0)
            buf298 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_91, buf298, 65536, stream=stream0)
            del primals_91
            buf300 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_92, buf300, 256, stream=stream0)
            del primals_92
            # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten._to_copy, aten.convolution]
            buf302 = extern_kernels.convolution(buf301, buf298, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf302, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf303 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_17], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_93, buf303, 65536, stream=stream0)
            del primals_93
            buf304 = buf301; del buf301  # reuse
            # Topologically Sorted Source Nodes: [x_25, add_27, h_11], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_41.run(buf302, buf300, buf268, buf304, 8388608, stream=stream0)
            buf305 = reinterpret_tensor(buf296, (32768, 256), (256, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [x_25, add_27, h_11, view_62, q_16, q_17], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf304, (32768, 256), (256, 1), 0), buf303, out=buf305)
            buf306 = empty_strided_cuda((256, 512), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_94, buf306, 131072, stream=stream0)
            del primals_94
            buf307 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf306, out=buf307)
            buf308 = empty_strided_cuda((128, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_17, view_63, q_18, matmul_12], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf305, buf308, 8388608, stream=stream0)
            buf309 = empty_strided_cuda((128, 4, 64, 1), (256, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_64, k_12, transpose_50, matmul_12], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf307, buf309, 32768, stream=stream0)
            buf310 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_17, kv_5, chunk_5, view_63, q_18, view_64, k_12, transpose_50, matmul_12], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf308, (512, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf309, (512, 64, 1), (64, 1, 0), 0), out=buf310)
            buf311 = empty_strided_cuda((128, 4, 256, 1), (1024, 256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_12, attn_13, matmul_13], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32.run(buf310, buf311, 131072, stream=stream0)
            buf312 = empty_strided_cuda((128, 4, 1, 64), (256, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_65, v_26, matmul_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf307, buf312, 32768, stream=stream0)
            buf313 = reinterpret_tensor(buf305, (512, 256, 64), (16384, 64, 1), 0); del buf305  # reuse
            # Topologically Sorted Source Nodes: [kv_5, chunk_5, view_65, v_26, matmul_12, attn_13, matmul_13], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf311, (512, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf312, (512, 1, 64), (64, 0, 1), 0), out=buf313)
            buf314 = empty_strided_cuda((128, 256, 4, 64), (65536, 256, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_34.run(buf313, buf314, 8388608, stream=stream0)
            buf315 = reinterpret_tensor(buf307, (256, 256), (256, 1), 0); del buf307  # reuse
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_95, buf315, 65536, stream=stream0)
            del primals_95
            buf316 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_96, buf316, 256, stream=stream0)
            del primals_96
            buf317 = reinterpret_tensor(buf313, (32768, 256), (256, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [matmul_13, transpose_51, out_30, input_30], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf316, reinterpret_tensor(buf314, (32768, 256), (256, 1), 0), reinterpret_tensor(buf315, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf317)
            # Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten.view, aten.native_dropout]
            buf318 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf317, (128, 256, 256), (65536, 256, 1), 0), 0.1, True)
            buf319 = buf318[0]
            assert_size_stride(buf319, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf319, 16, 'torch.ops.aten.native_dropout.default')
            buf320 = buf318[1]
            assert_size_stride(buf320, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf320, 16, 'torch.ops.aten.native_dropout.default')
            del buf318
            buf321 = reinterpret_tensor(buf319, (128, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [transpose_52, out_31, x_26], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_35.run(buf321, buf304, 8388608, stream=stream0)
            buf322 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(primals_97, buf322, 589824, stream=stream0)
            del primals_97
            buf323 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_98, buf323, 256, stream=stream0)
            del primals_98
            # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten._to_copy, aten.convolution]
            buf324 = extern_kernels.convolution(buf321, buf322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf324, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf325 = buf324; del buf324  # reuse
            # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf325, buf323, 8388608, stream=stream0)
            buf326 = buf291; del buf291  # reuse
            buf327 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf329 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_8], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_37.run(buf325, buf326, buf327, buf329, 4096, 2048, stream=stream0)
            buf330 = reinterpret_tensor(buf317, (128, 256, 256), (65536, 1, 256), 0); del buf317  # reuse
            buf585 = empty_strided_cuda((128, 256, 256), (65536, 1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_8, x_flat_8, v_t_x_7], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38.run(buf325, buf326, buf327, buf330, buf585, 32768, 256, stream=stream0)
            buf331 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, v_7, transpose_9, v_t_x_7], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 3072), buf330, out=buf331)
            buf332 = reinterpret_tensor(buf330, (128, 256, 256), (65536, 256, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_7], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 0), buf331, out=buf332)
            buf335 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            buf584 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
            buf337 = empty_strided_cuda((128, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, shift_5, x_norm_8, x_flat_8, out_32, view_71, out_33, x_28, x_29], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39.run(buf325, buf326, buf327, buf332, buf29, buf335, buf584, buf337, 8388608, stream=stream0)
            buf334 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_99, buf334, 65536, stream=stream0)
            del primals_99
            buf336 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_100, buf336, 256, stream=stream0)
            del primals_100
            # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten._to_copy, aten.convolution]
            buf338 = extern_kernels.convolution(buf337, buf334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf338, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf339 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_20], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_101, buf339, 65536, stream=stream0)
            del primals_101
            buf340 = buf337; del buf337  # reuse
            # Topologically Sorted Source Nodes: [x_29, add_31, h_12], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_41.run(buf338, buf336, buf304, buf340, 8388608, stream=stream0)
            buf341 = reinterpret_tensor(buf332, (32768, 256), (256, 1), 0); del buf332  # reuse
            # Topologically Sorted Source Nodes: [x_29, add_31, h_12, view_72, q_19, q_20], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf340, (32768, 256), (256, 1), 0), buf339, out=buf341)
            buf342 = empty_strided_cuda((256, 512), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_102, buf342, 131072, stream=stream0)
            del primals_102
            buf343 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf342, out=buf343)
            buf344 = empty_strided_cuda((128, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_20, view_73, q_21, matmul_14], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf341, buf344, 8388608, stream=stream0)
            buf345 = empty_strided_cuda((128, 4, 64, 1), (256, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_74, k_14, transpose_58, matmul_14], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf343, buf345, 32768, stream=stream0)
            buf346 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_20, kv_6, chunk_6, view_73, q_21, view_74, k_14, transpose_58, matmul_14], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf344, (512, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf345, (512, 64, 1), (64, 1, 0), 0), out=buf346)
            buf347 = empty_strided_cuda((128, 4, 256, 1), (1024, 256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_14, attn_15, matmul_15], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32.run(buf346, buf347, 131072, stream=stream0)
            buf348 = empty_strided_cuda((128, 4, 1, 64), (256, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_75, v_30, matmul_15], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf343, buf348, 32768, stream=stream0)
            buf349 = reinterpret_tensor(buf341, (512, 256, 64), (16384, 64, 1), 0); del buf341  # reuse
            # Topologically Sorted Source Nodes: [kv_6, chunk_6, view_75, v_30, matmul_14, attn_15, matmul_15], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf347, (512, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf348, (512, 1, 64), (64, 0, 1), 0), out=buf349)
            buf350 = empty_strided_cuda((128, 256, 4, 64), (65536, 256, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_34.run(buf349, buf350, 8388608, stream=stream0)
            buf351 = reinterpret_tensor(buf343, (256, 256), (256, 1), 0); del buf343  # reuse
            # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_103, buf351, 65536, stream=stream0)
            del primals_103
            buf352 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_104, buf352, 256, stream=stream0)
            del primals_104
            buf353 = reinterpret_tensor(buf349, (32768, 256), (256, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [matmul_15, transpose_59, out_34, input_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf352, reinterpret_tensor(buf350, (32768, 256), (256, 1), 0), reinterpret_tensor(buf351, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf353)
            # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten.view, aten.native_dropout]
            buf354 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf353, (128, 256, 256), (65536, 256, 1), 0), 0.1, True)
            buf355 = buf354[0]
            assert_size_stride(buf355, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf355, 16, 'torch.ops.aten.native_dropout.default')
            buf356 = buf354[1]
            assert_size_stride(buf356, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf356, 16, 'torch.ops.aten.native_dropout.default')
            del buf354
            buf357 = reinterpret_tensor(buf355, (128, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf355  # reuse
            # Topologically Sorted Source Nodes: [transpose_60, out_35, x_30], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_35.run(buf357, buf340, 8388608, stream=stream0)
            buf358 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(primals_105, buf358, 589824, stream=stream0)
            del primals_105
            buf359 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_106, buf359, 256, stream=stream0)
            del primals_106
            # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._to_copy, aten.convolution]
            buf360 = extern_kernels.convolution(buf357, buf358, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf360, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf361 = buf360; del buf360  # reuse
            # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf361, buf359, 8388608, stream=stream0)
            buf362 = buf327; del buf327  # reuse
            buf363 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf365 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_9], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_37.run(buf361, buf362, buf363, buf365, 4096, 2048, stream=stream0)
            buf366 = reinterpret_tensor(buf353, (128, 256, 256), (65536, 1, 256), 0); del buf353  # reuse
            buf582 = empty_strided_cuda((128, 256, 256), (65536, 1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_9, x_flat_9, v_t_x_8], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38.run(buf361, buf362, buf363, buf366, buf582, 32768, 256, stream=stream0)
            buf367 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, v_7, transpose_9, v_t_x_8], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 3072), buf366, out=buf367)
            buf368 = reinterpret_tensor(buf366, (128, 256, 256), (65536, 256, 1), 0); del buf366  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_8], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 0), buf367, out=buf368)
            buf371 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            buf581 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
            buf373 = empty_strided_cuda((128, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, shift_5, x_norm_9, x_flat_9, out_36, view_81, out_37, x_32, x_33], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39.run(buf361, buf362, buf363, buf368, buf29, buf371, buf581, buf373, 8388608, stream=stream0)
            buf370 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_107, buf370, 65536, stream=stream0)
            del primals_107
            buf372 = buf359; del buf359  # reuse
            # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_108, buf372, 256, stream=stream0)
            del primals_108
            # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten._to_copy, aten.convolution]
            buf374 = extern_kernels.convolution(buf373, buf370, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf374, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf375 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_23], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_109, buf375, 65536, stream=stream0)
            del primals_109
            buf376 = buf373; del buf373  # reuse
            # Topologically Sorted Source Nodes: [x_33, add_35, h_13], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_41.run(buf374, buf372, buf340, buf376, 8388608, stream=stream0)
            buf377 = reinterpret_tensor(buf368, (32768, 256), (256, 1), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [x_33, add_35, h_13, view_82, q_22, q_23], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf376, (32768, 256), (256, 1), 0), buf375, out=buf377)
            buf378 = empty_strided_cuda((256, 512), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_110, buf378, 131072, stream=stream0)
            del primals_110
            buf379 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf378, out=buf379)
            buf380 = empty_strided_cuda((128, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_23, view_83, q_24, matmul_16], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf377, buf380, 8388608, stream=stream0)
            buf381 = empty_strided_cuda((128, 4, 64, 1), (256, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_84, k_16, transpose_66, matmul_16], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf379, buf381, 32768, stream=stream0)
            buf382 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_23, kv_7, chunk_7, view_83, q_24, view_84, k_16, transpose_66, matmul_16], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf380, (512, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf381, (512, 64, 1), (64, 1, 0), 0), out=buf382)
            buf383 = empty_strided_cuda((128, 4, 256, 1), (1024, 256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_16, attn_17, matmul_17], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32.run(buf382, buf383, 131072, stream=stream0)
            buf384 = empty_strided_cuda((128, 4, 1, 64), (256, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_85, v_34, matmul_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf379, buf384, 32768, stream=stream0)
            buf385 = reinterpret_tensor(buf377, (512, 256, 64), (16384, 64, 1), 0); del buf377  # reuse
            # Topologically Sorted Source Nodes: [kv_7, chunk_7, view_85, v_34, matmul_16, attn_17, matmul_17], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf383, (512, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf384, (512, 1, 64), (64, 0, 1), 0), out=buf385)
            buf386 = empty_strided_cuda((128, 256, 4, 64), (65536, 256, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_34.run(buf385, buf386, 8388608, stream=stream0)
            buf387 = reinterpret_tensor(buf379, (256, 256), (256, 1), 0); del buf379  # reuse
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_111, buf387, 65536, stream=stream0)
            del primals_111
            buf388 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_112, buf388, 256, stream=stream0)
            del primals_112
            buf389 = reinterpret_tensor(buf385, (32768, 256), (256, 1), 0); del buf385  # reuse
            # Topologically Sorted Source Nodes: [matmul_17, transpose_67, out_38, input_34], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf388, reinterpret_tensor(buf386, (32768, 256), (256, 1), 0), reinterpret_tensor(buf387, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf389)
            # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.view, aten.native_dropout]
            buf390 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf389, (128, 256, 256), (65536, 256, 1), 0), 0.1, True)
            buf391 = buf390[0]
            assert_size_stride(buf391, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf391, 16, 'torch.ops.aten.native_dropout.default')
            buf392 = buf390[1]
            assert_size_stride(buf392, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf392, 16, 'torch.ops.aten.native_dropout.default')
            del buf390
            buf393 = reinterpret_tensor(buf391, (128, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf391  # reuse
            # Topologically Sorted Source Nodes: [transpose_68, out_39, x_34], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_35.run(buf393, buf376, 8388608, stream=stream0)
            buf394 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(primals_113, buf394, 589824, stream=stream0)
            del primals_113
            buf395 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_114, buf395, 256, stream=stream0)
            del primals_114
            # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._to_copy, aten.convolution]
            buf396 = extern_kernels.convolution(buf393, buf394, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf396, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf397 = buf396; del buf396  # reuse
            # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf397, buf395, 8388608, stream=stream0)
            buf398 = buf363; del buf363  # reuse
            buf399 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf401 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_10], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_37.run(buf397, buf398, buf399, buf401, 4096, 2048, stream=stream0)
            buf402 = reinterpret_tensor(buf389, (128, 256, 256), (65536, 1, 256), 0); del buf389  # reuse
            buf579 = empty_strided_cuda((128, 256, 256), (65536, 1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_10, x_flat_10, v_t_x_9], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38.run(buf397, buf398, buf399, buf402, buf579, 32768, 256, stream=stream0)
            buf403 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, v_7, transpose_9, v_t_x_9], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 3072), buf402, out=buf403)
            buf404 = reinterpret_tensor(buf402, (128, 256, 256), (65536, 256, 1), 0); del buf402  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_9], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 0), buf403, out=buf404)
            buf407 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            buf578 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
            buf409 = empty_strided_cuda((128, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, shift_5, x_norm_10, x_flat_10, out_40, view_91, out_41, x_36, x_37], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39.run(buf397, buf398, buf399, buf404, buf29, buf407, buf578, buf409, 8388608, stream=stream0)
            buf406 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_115, buf406, 65536, stream=stream0)
            del primals_115
            buf408 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_116, buf408, 256, stream=stream0)
            del primals_116
            # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten._to_copy, aten.convolution]
            buf410 = extern_kernels.convolution(buf409, buf406, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf410, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf411 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_26], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_117, buf411, 65536, stream=stream0)
            del primals_117
            buf412 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [x_37, add_39, h_14], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_41.run(buf410, buf408, buf376, buf412, 8388608, stream=stream0)
            buf413 = reinterpret_tensor(buf404, (32768, 256), (256, 1), 0); del buf404  # reuse
            # Topologically Sorted Source Nodes: [x_37, add_39, h_14, view_92, q_25, q_26], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.view, aten.transpose, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf412, (32768, 256), (256, 1), 0), buf411, out=buf413)
            buf414 = empty_strided_cuda((256, 512), (1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_8.run(primals_118, buf414, 131072, stream=stream0)
            del primals_118
            buf415 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, buf414, out=buf415)
            buf416 = empty_strided_cuda((128, 4, 256, 64), (65536, 16384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_26, view_93, q_27, matmul_18], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_30.run(buf413, buf416, 8388608, stream=stream0)
            buf417 = empty_strided_cuda((128, 4, 64, 1), (256, 64, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_94, k_18, transpose_74, matmul_18], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_31.run(buf415, buf417, 32768, stream=stream0)
            buf418 = empty_strided_cuda((512, 256, 1), (256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q_26, kv_8, chunk_8, view_93, q_27, view_94, k_18, transpose_74, matmul_18], Original ATen: [aten._unsafe_view, aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf416, (512, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf417, (512, 64, 1), (64, 1, 0), 0), out=buf418)
            buf419 = empty_strided_cuda((128, 4, 256, 1), (1024, 256, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_18, attn_19, matmul_19], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax]
            stream0 = get_raw_stream(0)
            triton_poi_fused__softmax__to_copy_amax_mul_sub_view_32.run(buf418, buf419, 131072, stream=stream0)
            buf420 = empty_strided_cuda((128, 4, 1, 64), (256, 64, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_95, v_38, matmul_19], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_unsqueeze_view_33.run(buf415, buf420, 32768, stream=stream0)
            buf421 = reinterpret_tensor(buf413, (512, 256, 64), (16384, 64, 1), 0); del buf413  # reuse
            # Topologically Sorted Source Nodes: [kv_8, chunk_8, view_95, v_38, matmul_18, attn_19, matmul_19], Original ATen: [aten.unsqueeze, aten.split, aten.view, aten.transpose, aten._to_copy, aten.mul, aten.amax, aten.sub, aten._softmax, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf419, (512, 256, 1), (256, 1, 0), 0), reinterpret_tensor(buf420, (512, 1, 64), (64, 0, 1), 0), out=buf421)
            buf422 = empty_strided_cuda((128, 256, 4, 64), (65536, 256, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42], Original ATen: [aten.view, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_34.run(buf421, buf422, 8388608, stream=stream0)
            buf423 = reinterpret_tensor(buf415, (256, 256), (256, 1), 0); del buf415  # reuse
            # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_119, buf423, 65536, stream=stream0)
            del primals_119
            buf424 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_120, buf424, 256, stream=stream0)
            del primals_120
            buf425 = reinterpret_tensor(buf421, (32768, 256), (256, 1), 0); del buf421  # reuse
            # Topologically Sorted Source Nodes: [matmul_19, transpose_75, out_42, input_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf424, reinterpret_tensor(buf422, (32768, 256), (256, 1), 0), reinterpret_tensor(buf423, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf425)
            # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.view, aten.native_dropout]
            buf426 = torch.ops.aten.native_dropout.default(reinterpret_tensor(buf425, (128, 256, 256), (65536, 256, 1), 0), 0.1, True)
            buf427 = buf426[0]
            assert_size_stride(buf427, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf427, 16, 'torch.ops.aten.native_dropout.default')
            buf428 = buf426[1]
            assert_size_stride(buf428, (128, 256, 256), (65536, 256, 1), 'torch.ops.aten.native_dropout.default')
            assert_alignment(buf428, 16, 'torch.ops.aten.native_dropout.default')
            del buf426
            buf429 = reinterpret_tensor(buf427, (128, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf427  # reuse
            # Topologically Sorted Source Nodes: [transpose_76, out_43, x_38], Original ATen: [aten.transpose, aten.view, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_transpose_view_35.run(buf429, buf412, 8388608, stream=stream0)
            buf430 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(primals_121, buf430, 589824, stream=stream0)
            del primals_121
            buf431 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_122, buf431, 256, stream=stream0)
            del primals_122
            # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._to_copy, aten.convolution]
            buf432 = extern_kernels.convolution(buf429, buf430, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf432, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf433 = buf432; del buf432  # reuse
            # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_29.run(buf433, buf431, 8388608, stream=stream0)
            buf434 = buf399; del buf399  # reuse
            buf435 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf437 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_11], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_37.run(buf433, buf434, buf435, buf437, 4096, 2048, stream=stream0)
            buf438 = reinterpret_tensor(buf425, (128, 256, 256), (65536, 1, 256), 0); del buf425  # reuse
            buf576 = empty_strided_cuda((128, 256, 256), (65536, 1, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_11, x_flat_11, v_t_x_10], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_38.run(buf433, buf434, buf435, buf438, buf576, 32768, 256, stream=stream0)
            buf439 = empty_strided_cuda((128, 12, 256), (3072, 256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, v_7, transpose_9, v_t_x_10], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 3072), buf438, out=buf439)
            buf440 = reinterpret_tensor(buf438, (128, 256, 256), (65536, 256, 1), 0); del buf438  # reuse
            # Topologically Sorted Source Nodes: [split_2, u_5, mixed_10], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 0), buf439, out=buf440)
            buf443 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.bfloat16)
            buf575 = empty_strided_cuda((128, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
            buf445 = empty_strided_cuda((128, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split_2, shift_5, x_norm_11, x_flat_11, out_44, view_101, out_45, x_40, x_41], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_39.run(buf433, buf434, buf435, buf440, buf29, buf443, buf575, buf445, 8388608, stream=stream0)
            del buf440
            buf442 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_123, buf442, 65536, stream=stream0)
            del primals_123
            buf444 = buf431; del buf431  # reuse
            # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_124, buf444, 256, stream=stream0)
            del primals_124
            # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten._to_copy, aten.convolution]
            buf446 = extern_kernels.convolution(buf445, buf442, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf446, (128, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf447 = buf445; del buf445  # reuse
            # Topologically Sorted Source Nodes: [x_41, add_43, h_15], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_41.run(buf446, buf444, buf412, buf447, 8388608, stream=stream0)
            buf448 = empty_strided_cuda((256, 128, 4, 4), (2048, 1, 512, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_16], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(primals_125, buf448, 524288, stream=stream0)
            del primals_125
            buf449 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [h_16], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_126, buf449, 128, stream=stream0)
            del primals_126
            # Topologically Sorted Source Nodes: [h_16], Original ATen: [aten._to_copy, aten.convolution]
            buf450 = extern_kernels.convolution(buf447, buf448, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf450, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf451 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(primals_127, buf451, 65536, stream=stream0)
            del primals_127
            buf452 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(primals_128, buf452, 256, stream=stream0)
            del primals_128
            buf453 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf452, buf15, reinterpret_tensor(buf451, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf453)
            del buf452
            buf454 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf453, buf454, 32768, stream=stream0)
            buf455 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_64.run(primals_129, buf455, 32768, stream=stream0)
            del primals_129
            buf456 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_130, buf456, 128, stream=stream0)
            del primals_130
            buf457 = empty_strided_cuda((128, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy, aten.t, aten.addmm]
            extern_kernels.addmm(buf456, buf454, reinterpret_tensor(buf455, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf457)
            buf458 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [h_16, input_41, unsqueeze_11, gate_1, h_32_gated, h_17], Original ATen: [aten._to_copy, aten.convolution, aten.sigmoid, aten.unsqueeze, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_mul_sigmoid_unsqueeze_65.run(buf458, buf449, buf78, buf457, 16777216, stream=stream0)
            buf459 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(primals_131, buf459, 147456, stream=stream0)
            del primals_131
            buf460 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_132, buf460, 128, stream=stream0)
            del primals_132
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.convolution]
            buf461 = extern_kernels.convolution(buf458, buf459, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf461, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf462 = buf461; del buf461  # reuse
            # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf462, buf460, 16777216, stream=stream0)
            buf463 = buf435; del buf435  # reuse
            buf464 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf466 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_12], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_22.run(buf462, buf463, buf464, buf466, 4096, 4096, stream=stream0)
            buf467 = reinterpret_tensor(buf71, (128, 128, 1024), (131072, 1, 128), 0); del buf71  # reuse
            buf573 = empty_strided_cuda((128, 1024, 128), (131072, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_12, x_flat_12, v_t_x_11], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23.run(buf462, buf463, buf464, buf467, buf573, 131072, 128, stream=stream0)
            buf468 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, v_1, transpose, v_t_x_11], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 1536), buf467, out=buf468)
            buf469 = reinterpret_tensor(buf467, (128, 128, 1024), (131072, 1024, 1), 0); del buf467  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed_11], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 0), buf468, out=buf469)
            buf472 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            buf572 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
            buf474 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, shift_1, x_norm_12, x_flat_12, out_46, view_106, out_47, x_43, x_44], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24.run(buf462, buf463, buf464, buf469, buf22, buf472, buf572, buf474, 16777216, stream=stream0)
            buf471 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(primals_133, buf471, 16384, stream=stream0)
            del primals_133
            buf473 = buf460; del buf460  # reuse
            # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_134, buf473, 128, stream=stream0)
            del primals_134
            # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten._to_copy, aten.convolution]
            buf475 = extern_kernels.convolution(buf474, buf471, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf475, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf476 = buf474; del buf474  # reuse
            # Topologically Sorted Source Nodes: [x_44, add_47, h_18], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_27.run(buf475, buf473, buf458, buf476, 16777216, stream=stream0)
            buf477 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(primals_135, buf477, 147456, stream=stream0)
            del primals_135
            buf478 = buf456; del buf456  # reuse
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_136, buf478, 128, stream=stream0)
            del primals_136
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy, aten.convolution]
            buf479 = extern_kernels.convolution(buf476, buf477, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf479, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf480 = buf479; del buf479  # reuse
            # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf480, buf478, 16777216, stream=stream0)
            buf481 = buf464; del buf464  # reuse
            buf482 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf484 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_13], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_22.run(buf480, buf481, buf482, buf484, 4096, 4096, stream=stream0)
            buf485 = reinterpret_tensor(buf469, (128, 128, 1024), (131072, 1, 128), 0); del buf469  # reuse
            buf570 = empty_strided_cuda((128, 1024, 128), (131072, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_13, x_flat_13, v_t_x_12], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23.run(buf480, buf481, buf482, buf485, buf570, 131072, 128, stream=stream0)
            buf486 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, v_1, transpose, v_t_x_12], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 1536), buf485, out=buf486)
            buf487 = reinterpret_tensor(buf485, (128, 128, 1024), (131072, 1024, 1), 0); del buf485  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed_12], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 0), buf486, out=buf487)
            buf490 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            buf569 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
            buf492 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, shift_1, x_norm_13, x_flat_13, out_48, view_111, out_49, x_46, x_47], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24.run(buf480, buf481, buf482, buf487, buf22, buf490, buf569, buf492, 16777216, stream=stream0)
            buf489 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(primals_137, buf489, 16384, stream=stream0)
            del primals_137
            buf491 = buf478; del buf478  # reuse
            # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_138, buf491, 128, stream=stream0)
            del primals_138
            # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._to_copy, aten.convolution]
            buf493 = extern_kernels.convolution(buf492, buf489, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf493, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf494 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [x_47, add_50, h_19], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_silu_27.run(buf493, buf491, buf476, buf494, 16777216, stream=stream0)
            buf495 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(primals_139, buf495, 147456, stream=stream0)
            del primals_139
            buf496 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_140, buf496, 128, stream=stream0)
            del primals_140
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.convolution]
            buf497 = extern_kernels.convolution(buf494, buf495, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf497, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf498 = buf497; del buf497  # reuse
            # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf498, buf496, 16777216, stream=stream0)
            buf499 = buf482; del buf482  # reuse
            buf500 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf502 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_14], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_22.run(buf498, buf499, buf500, buf502, 4096, 4096, stream=stream0)
            buf503 = reinterpret_tensor(buf487, (128, 128, 1024), (131072, 1, 128), 0); del buf487  # reuse
            buf567 = empty_strided_cuda((128, 1024, 128), (131072, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_14, x_flat_14, v_t_x_13], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23.run(buf498, buf499, buf500, buf503, buf567, 131072, 128, stream=stream0)
            buf504 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, v_1, transpose, v_t_x_13], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 1536), buf503, out=buf504)
            buf505 = reinterpret_tensor(buf503, (128, 128, 1024), (131072, 1024, 1), 0); del buf503  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed_13], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 0), buf504, out=buf505)
            buf508 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            buf566 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
            buf510 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, shift_1, x_norm_14, x_flat_14, out_50, view_116, out_51, x_49, x_50], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24.run(buf498, buf499, buf500, buf505, buf22, buf508, buf566, buf510, 16777216, stream=stream0)
            buf507 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(primals_141, buf507, 16384, stream=stream0)
            del primals_141
            buf509 = buf496; del buf496  # reuse
            # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_142, buf509, 128, stream=stream0)
            del primals_142
            # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten._to_copy, aten.convolution]
            buf511 = extern_kernels.convolution(buf510, buf507, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf511, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf512 = buf510; del buf510  # reuse
            buf565 = reinterpret_tensor(buf505, (128, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [x_50, add_53, h_20], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_silu_sub_66.run(buf511, buf509, buf494, buf512, buf565, 16777216, stream=stream0)
            buf513 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(primals_143, buf513, 147456, stream=stream0)
            del primals_143
            buf514 = buf509; del buf509  # reuse
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_144, buf514, 128, stream=stream0)
            del primals_144
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy, aten.convolution]
            buf515 = extern_kernels.convolution(buf512, buf513, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf515, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf516 = buf515; del buf515  # reuse
            # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf516, buf514, 16777216, stream=stream0)
            buf517 = buf500; del buf500  # reuse
            buf518 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf520 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_15], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_22.run(buf516, buf517, buf518, buf520, 4096, 4096, stream=stream0)
            buf521 = reinterpret_tensor(buf511, (128, 128, 1024), (131072, 1, 128), 0); del buf511  # reuse
            buf564 = empty_strided_cuda((128, 1024, 128), (131072, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_15, x_flat_15, v_t_x_14], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23.run(buf516, buf517, buf518, buf521, buf564, 131072, 128, stream=stream0)
            buf522 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, v_1, transpose, v_t_x_14], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 1536), buf521, out=buf522)
            buf523 = reinterpret_tensor(buf521, (128, 128, 1024), (131072, 1024, 1), 0); del buf521  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed_14], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 0), buf522, out=buf523)
            buf526 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            buf563 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
            buf528 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, shift_1, x_norm_15, x_flat_15, out_52, view_121, out_53, x_52, x_53], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24.run(buf516, buf517, buf518, buf523, buf22, buf526, buf563, buf528, 16777216, stream=stream0)
            buf525 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(primals_145, buf525, 16384, stream=stream0)
            del primals_145
            buf527 = buf514; del buf514  # reuse
            # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_146, buf527, 128, stream=stream0)
            del primals_146
            # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten._to_copy, aten.convolution]
            buf529 = extern_kernels.convolution(buf528, buf525, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf529, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf530 = buf528; del buf528  # reuse
            buf562 = reinterpret_tensor(buf523, (128, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf523  # reuse
            # Topologically Sorted Source Nodes: [x_53, add_56, h_21], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.silu, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_silu_sub_66.run(buf529, buf527, buf512, buf530, buf562, 16777216, stream=stream0)
            buf531 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(primals_147, buf531, 147456, stream=stream0)
            del primals_147
            buf532 = buf527; del buf527  # reuse
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_148, buf532, 128, stream=stream0)
            del primals_148
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.convolution]
            buf533 = extern_kernels.convolution(buf530, buf531, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf533, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf534 = buf533; del buf533  # reuse
            # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_20.run(buf534, buf532, 16777216, stream=stream0)
            buf535 = buf518; del buf518  # reuse
            buf536 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf538 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            # Topologically Sorted Source Nodes: [x_norm_16], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_native_group_norm_22.run(buf534, buf535, buf536, buf538, 4096, 4096, stream=stream0)
            buf539 = reinterpret_tensor(buf529, (128, 128, 1024), (131072, 1, 128), 0); del buf529  # reuse
            buf561 = empty_strided_cuda((128, 1024, 128), (131072, 1, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_norm_16, x_flat_16, v_t_x_15], Original ATen: [aten._to_copy, aten.clone, aten.native_group_norm, aten.view, aten.transpose]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_transpose_view_23.run(buf534, buf535, buf536, buf539, buf561, 131072, 128, stream=stream0)
            buf540 = empty_strided_cuda((128, 12, 1024), (12288, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, v_1, transpose, v_t_x_15], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 1536), buf539, out=buf540)
            buf541 = reinterpret_tensor(buf539, (128, 128, 1024), (131072, 1024, 1), 0); del buf539  # reuse
            # Topologically Sorted Source Nodes: [split, u_1, mixed_15], Original ATen: [aten.split_with_sizes, aten.view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 0), buf540, out=buf541)
            buf544 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.bfloat16)
            buf560 = empty_strided_cuda((128, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
            buf546 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, shift_1, x_norm_16, x_flat_16, out_54, view_126, out_55, x_55, x_56], Original ATen: [aten.split_with_sizes, aten.view, aten._to_copy, aten.clone, aten.native_group_norm, aten.add, aten.silu, aten.convolution, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_fill_mul_native_group_norm_silu_split_with_sizes_sub_view_24.run(buf534, buf535, buf536, buf541, buf22, buf544, buf560, buf546, 16777216, stream=stream0)
            del buf541
            buf543 = empty_strided_cuda((128, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_25.run(primals_149, buf543, 16384, stream=stream0)
            del primals_149
            buf545 = buf532; del buf532  # reuse
            # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_19.run(primals_150, buf545, 128, stream=stream0)
            del primals_150
            # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten._to_copy, aten.convolution]
            buf547 = extern_kernels.convolution(buf546, buf543, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf547, (128, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf548 = buf547; del buf547  # reuse
            # Topologically Sorted Source Nodes: [x_56, add_59], Original ATen: [aten._to_copy, aten.convolution, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_67.run(buf548, buf545, buf530, 16777216, stream=stream0)
            del buf545
            buf549 = reinterpret_tensor(buf536, (128, 32, 1, 1), (32, 1, 1, 1), 0); del buf536  # reuse
            buf550 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 4096, 4096), torch.float32)
            buf552 = reinterpret_tensor(buf550, (128, 32, 1, 1), (32, 1, 1, 1), 0); del buf550  # reuse
            # Topologically Sorted Source Nodes: [h_22, input_42], Original ATen: [aten.silu, aten.clone, aten.native_group_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_clone_native_group_norm_silu_68.run(buf552, buf548, buf549, 4096, 4096, stream=stream0)
            buf555 = reinterpret_tensor(buf546, (128, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf546  # reuse
            # Topologically Sorted Source Nodes: [h_22, input_42, input_43, input_44], Original ATen: [aten.silu, aten.clone, aten.native_group_norm, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_native_group_norm_silu_69.run(buf548, buf549, buf552, primals_151, primals_152, buf555, 131072, 128, stream=stream0)
            buf554 = empty_strided_cuda((4, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_17.run(primals_153, buf554, 4608, stream=stream0)
            del primals_153
            buf556 = empty_strided_cuda((4, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_70.run(primals_154, buf556, 4, stream=stream0)
            del primals_154
            buf557 = empty_strided_cuda((128, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_71.run(buf555, buf557, 16384, 1024, stream=stream0)
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy, aten.convolution]
            buf558 = extern_kernels.convolution(buf557, buf554, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf558, (128, 4, 32, 32), (4096, 1, 128, 4), 'torch.ops.aten.convolution.default')
            del buf557
            buf559 = buf558; del buf558  # reuse
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_72.run(buf559, buf556, 524288, stream=stream0)
            del buf556
            buf568 = buf493; del buf493  # reuse
            # Topologically Sorted Source Nodes: [x_47, add_50], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_73.run(buf568, buf491, buf476, 16777216, stream=stream0)
            del buf491
            buf571 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [x_44, add_47], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_73.run(buf571, buf473, buf458, 16777216, stream=stream0)
            del buf473
            buf574 = buf446; del buf446  # reuse
            # Topologically Sorted Source Nodes: [x_41, add_43], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.sigmoid, aten.fill, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74.run(buf574, buf444, buf412, 8388608, stream=stream0)
            del buf444
            buf577 = buf410; del buf410  # reuse
            # Topologically Sorted Source Nodes: [x_37, add_39], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74.run(buf577, buf408, buf376, 8388608, stream=stream0)
            del buf408
            buf580 = buf374; del buf374  # reuse
            # Topologically Sorted Source Nodes: [x_33, add_35], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74.run(buf580, buf372, buf340, 8388608, stream=stream0)
            del buf372
            buf583 = buf338; del buf338  # reuse
            # Topologically Sorted Source Nodes: [x_29, add_31], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74.run(buf583, buf336, buf304, 8388608, stream=stream0)
            del buf336
            buf586 = buf302; del buf302  # reuse
            # Topologically Sorted Source Nodes: [x_25, add_27], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74.run(buf586, buf300, buf268, 8388608, stream=stream0)
            del buf300
            buf593 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [x_13, add_13], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_74.run(buf593, buf152, buf120, 8388608, stream=stream0)
            del buf152
            buf598 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [x_5, add_5], Original ATen: [aten._to_copy, aten.convolution, aten.add, aten.fill, aten.sigmoid, aten.sub, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_fill_mul_sigmoid_sub_73.run(buf598, buf75, buf60, 16777216, stream=stream0)
            del buf75
        return (buf559, primals_7, primals_71, primals_77, primals_151, primals_152, buf0, buf3, buf4, buf8, buf11, buf12, buf15, buf18, buf19, buf25, buf26, buf32, buf33, buf37, buf38, buf41, buf42, buf45, reinterpret_tensor(buf46, (128, 32), (32, 1), 0), reinterpret_tensor(buf49, (128, 32), (32, 1), 0), buf54, buf55, buf59, buf60, buf61, buf64, reinterpret_tensor(buf65, (128, 32), (32, 1), 0), reinterpret_tensor(buf68, (128, 32), (32, 1), 0), buf73, buf74, buf78, buf79, buf82, buf89, reinterpret_tensor(buf93, (32768, 256), (256, 1), 0), buf99, buf100, buf101, buf104, reinterpret_tensor(buf105, (128, 32), (32, 1), 0), reinterpret_tensor(buf108, (128, 32), (32, 1), 0), buf113, buf114, buf118, reinterpret_tensor(buf120, (32768, 256), (256, 1), 0), buf126, reinterpret_tensor(buf130, (32768, 256), (256, 1), 0), buf136, buf137, buf138, buf141, reinterpret_tensor(buf142, (128, 32), (32, 1), 0), reinterpret_tensor(buf145, (128, 32), (32, 1), 0), buf150, buf151, buf155, buf156, buf159, buf166, reinterpret_tensor(buf170, (8192, 512), (512, 1), 0), buf176, buf177, buf178, buf181, reinterpret_tensor(buf182, (128, 32), (32, 1), 0), reinterpret_tensor(buf185, (128, 32), (32, 1), 0), buf190, buf191, buf195, reinterpret_tensor(buf197, (8192, 512), (512, 1), 0), buf203, reinterpret_tensor(buf207, (8192, 512), (512, 1), 0), buf213, buf214, buf215, buf218, reinterpret_tensor(buf219, (128, 32), (32, 1), 0), reinterpret_tensor(buf222, (128, 32), (32, 1), 0), buf227, buf228, buf232, buf233, buf236, reinterpret_tensor(buf237, (8192, 512), (512, 1), 0), buf242, buf243, buf244, reinterpret_tensor(buf248, (8192, 512), (512, 1), 0), buf251, buf252, buf255, buf256, buf257, buf262, buf263, buf266, reinterpret_tensor(buf268, (32768, 256), (256, 1), 0), buf274, reinterpret_tensor(buf278, (32768, 256), (256, 1), 0), buf284, buf285, buf286, buf289, reinterpret_tensor(buf290, (128, 32), (32, 1), 0), reinterpret_tensor(buf293, (128, 32), (32, 1), 0), buf298, buf299, reinterpret_tensor(buf304, (32768, 256), (256, 1), 0), buf310, reinterpret_tensor(buf314, (32768, 256), (256, 1), 0), buf320, buf321, buf322, buf325, reinterpret_tensor(buf326, (128, 32), (32, 1), 0), reinterpret_tensor(buf329, (128, 32), (32, 1), 0), buf334, buf335, reinterpret_tensor(buf340, (32768, 256), (256, 1), 0), buf346, reinterpret_tensor(buf350, (32768, 256), (256, 1), 0), buf356, buf357, buf358, buf361, reinterpret_tensor(buf362, (128, 32), (32, 1), 0), reinterpret_tensor(buf365, (128, 32), (32, 1), 0), buf370, buf371, reinterpret_tensor(buf376, (32768, 256), (256, 1), 0), buf382, reinterpret_tensor(buf386, (32768, 256), (256, 1), 0), buf392, buf393, buf394, buf397, reinterpret_tensor(buf398, (128, 32), (32, 1), 0), reinterpret_tensor(buf401, (128, 32), (32, 1), 0), buf406, buf407, reinterpret_tensor(buf412, (32768, 256), (256, 1), 0), buf418, reinterpret_tensor(buf422, (32768, 256), (256, 1), 0), buf428, buf429, buf430, buf433, reinterpret_tensor(buf434, (128, 32), (32, 1), 0), reinterpret_tensor(buf437, (128, 32), (32, 1), 0), buf442, buf443, buf447, buf448, buf453, buf454, buf457, buf458, buf459, buf462, reinterpret_tensor(buf463, (128, 32), (32, 1), 0), reinterpret_tensor(buf466, (128, 32), (32, 1), 0), buf471, buf472, buf476, buf477, buf480, reinterpret_tensor(buf481, (128, 32), (32, 1), 0), reinterpret_tensor(buf484, (128, 32), (32, 1), 0), buf489, buf490, buf494, buf495, buf498, reinterpret_tensor(buf499, (128, 32), (32, 1), 0), reinterpret_tensor(buf502, (128, 32), (32, 1), 0), buf507, buf508, buf512, buf513, buf516, reinterpret_tensor(buf517, (128, 32), (32, 1), 0), reinterpret_tensor(buf520, (128, 32), (32, 1), 0), buf525, buf526, buf530, buf531, buf534, reinterpret_tensor(buf535, (128, 32), (32, 1), 0), reinterpret_tensor(buf538, (128, 32), (32, 1), 0), buf543, buf544, buf548, buf549, buf552, buf554, buf555, buf560, reinterpret_tensor(buf22, (128, 12, 128), (3200, 1, 12), 0), reinterpret_tensor(buf540, (128, 1024, 12), (12288, 1, 1024), 0), reinterpret_tensor(buf22, (128, 128, 12), (3200, 12, 1), 1536), buf561, buf562, buf563, reinterpret_tensor(buf522, (128, 1024, 12), (12288, 1, 1024), 0), buf564, buf565, buf566, reinterpret_tensor(buf504, (128, 1024, 12), (12288, 1, 1024), 0), buf567, buf568, buf569, reinterpret_tensor(buf486, (128, 1024, 12), (12288, 1, 1024), 0), buf570, buf571, buf572, reinterpret_tensor(buf468, (128, 1024, 12), (12288, 1, 1024), 0), buf573, buf455, buf451, buf574, buf575, reinterpret_tensor(buf29, (128, 12, 256), (6400, 1, 12), 0), reinterpret_tensor(buf439, (128, 256, 12), (3072, 1, 256), 0), reinterpret_tensor(buf29, (128, 256, 12), (6400, 12, 1), 3072), buf576, buf423, reinterpret_tensor(buf419, (512, 1, 256), (256, 1, 1), 0), reinterpret_tensor(buf420, (512, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf416, (512, 64, 256), (16384, 1, 64), 0), reinterpret_tensor(buf417, (512, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf414, (512, 256), (256, 1), 0), reinterpret_tensor(buf411, (256, 256), (256, 1), 0), buf577, buf578, reinterpret_tensor(buf403, (128, 256, 12), (3072, 1, 256), 0), buf579, buf387, reinterpret_tensor(buf383, (512, 1, 256), (256, 1, 1), 0), reinterpret_tensor(buf384, (512, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf380, (512, 64, 256), (16384, 1, 64), 0), reinterpret_tensor(buf381, (512, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf378, (512, 256), (256, 1), 0), reinterpret_tensor(buf375, (256, 256), (256, 1), 0), buf580, buf581, reinterpret_tensor(buf367, (128, 256, 12), (3072, 1, 256), 0), buf582, buf351, reinterpret_tensor(buf347, (512, 1, 256), (256, 1, 1), 0), reinterpret_tensor(buf348, (512, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf344, (512, 64, 256), (16384, 1, 64), 0), reinterpret_tensor(buf345, (512, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf342, (512, 256), (256, 1), 0), reinterpret_tensor(buf339, (256, 256), (256, 1), 0), buf583, buf584, reinterpret_tensor(buf331, (128, 256, 12), (3072, 1, 256), 0), buf585, buf315, reinterpret_tensor(buf311, (512, 1, 256), (256, 1, 1), 0), reinterpret_tensor(buf312, (512, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf308, (512, 64, 256), (16384, 1, 64), 0), reinterpret_tensor(buf309, (512, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf306, (512, 256), (256, 1), 0), reinterpret_tensor(buf303, (256, 256), (256, 1), 0), buf586, buf587, reinterpret_tensor(buf295, (128, 256, 12), (3072, 1, 256), 0), buf588, buf279, reinterpret_tensor(buf275, (512, 1, 256), (256, 1, 1), 0), reinterpret_tensor(buf276, (512, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf272, (512, 64, 256), (16384, 1, 64), 0), reinterpret_tensor(buf273, (512, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf270, (512, 256), (256, 1), 0), reinterpret_tensor(buf267, (256, 256), (256, 1), 0), buf264, buf260, reinterpret_tensor(buf249, (512, 512), (512, 1), 0), reinterpret_tensor(buf245, (1024, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf246, (1024, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf240, (1024, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf241, (1024, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf238, (1536, 512), (512, 1), 0), buf589, reinterpret_tensor(buf36, (128, 12, 512), (12800, 1, 12), 0), reinterpret_tensor(buf224, (128, 64, 12), (768, 1, 64), 0), reinterpret_tensor(buf36, (128, 512, 12), (12800, 12, 1), 6144), buf590, buf208, reinterpret_tensor(buf204, (1024, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf205, (1024, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf201, (1024, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf202, (1024, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf199, (1024, 256), (256, 1), 0), reinterpret_tensor(buf196, (512, 512), (512, 1), 0), buf591, reinterpret_tensor(buf187, (128, 64, 12), (768, 1, 64), 0), buf592, buf171, reinterpret_tensor(buf167, (1024, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf168, (1024, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf164, (1024, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf165, (1024, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf162, (1024, 256), (256, 1), 0), reinterpret_tensor(buf160, (512, 512), (512, 1), 0), buf593, buf594, reinterpret_tensor(buf147, (128, 256, 12), (3072, 1, 256), 0), buf595, buf131, reinterpret_tensor(buf127, (512, 1, 256), (256, 1, 1), 0), reinterpret_tensor(buf128, (512, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf124, (512, 64, 256), (16384, 1, 64), 0), reinterpret_tensor(buf125, (512, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf122, (512, 256), (256, 1), 0), reinterpret_tensor(buf119, (256, 256), (256, 1), 0), buf596, reinterpret_tensor(buf110, (128, 256, 12), (3072, 1, 256), 0), buf597, buf94, reinterpret_tensor(buf90, (512, 1, 256), (256, 1, 1), 0), reinterpret_tensor(buf91, (512, 64, 1), (64, 1, 64), 0), reinterpret_tensor(buf87, (512, 64, 256), (16384, 1, 64), 0), reinterpret_tensor(buf88, (512, 1, 64), (64, 1, 1), 0), reinterpret_tensor(buf85, (512, 256), (256, 1), 0), reinterpret_tensor(buf83, (256, 256), (256, 1), 0), buf598, buf599, reinterpret_tensor(buf70, (128, 1024, 12), (12288, 1, 1024), 0), buf600, buf601, reinterpret_tensor(buf51, (128, 1024, 12), (12288, 1, 1024), 0), buf602, buf34, buf30, buf27, buf23, buf20, buf16, buf13, buf9, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_8 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3200, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((3200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((6400, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((6400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((12800, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((12800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 4, 3, 3), (36, 1, 12, 4), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 4, 32, 32), (4096, 1, 128, 4), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((4, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
