
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_66', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 125830656}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_split_with_sizes_view_66(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
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
