
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26214400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_71(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
