
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_fill_mul_native_group_norm_native_group_norm_backward_sigmoid_silu_sub_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 134218240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_fill_mul_native_group_norm_native_group_norm_backward_sigmoid_silu_sub_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y5 = yindex
    x3 = xindex // 4
    y1 = yindex // 1024
    y0 = (yindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x4 + 128*y5), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x4 + 128*y5), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x4 + 128*y5), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x3 + 32*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 1.0
    tmp5 = tmp4 - tmp3
    tmp6 = tmp2 * tmp5
    tmp7 = tmp6 + tmp4
    tmp8 = tmp3 * tmp7
    tmp9 = tmp1 * tmp8
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp10
    tmp20 = tmp19 * tmp10
    tmp21 = tmp20 * tmp10
    tmp22 = 0.000244140625
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp13 + tmp24
    tmp26 = -tmp23
    tmp28 = tmp26 * tmp27
    tmp30 = tmp29 * tmp10
    tmp31 = tmp30 * tmp22
    tmp32 = tmp28 - tmp31
    tmp33 = tmp25 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.sigmoid(tmp14)
    tmp36 = tmp4 - tmp35
    tmp37 = tmp14 * tmp36
    tmp38 = tmp37 + tmp4
    tmp39 = tmp35 * tmp38
    tmp40 = tmp34 * tmp39
    tl.store(out_ptr1 + (y0 + 1024*x4 + 131072*y1), tmp40, xmask & ymask)
