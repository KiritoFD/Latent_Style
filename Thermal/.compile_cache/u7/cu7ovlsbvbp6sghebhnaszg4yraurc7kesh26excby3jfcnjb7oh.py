
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83886080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
