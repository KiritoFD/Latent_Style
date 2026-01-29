
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
