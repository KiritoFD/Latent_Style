
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mul_silu_std_sub_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 294912, 'x': 98312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mul_silu_std_sub_70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
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
