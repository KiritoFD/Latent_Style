
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
