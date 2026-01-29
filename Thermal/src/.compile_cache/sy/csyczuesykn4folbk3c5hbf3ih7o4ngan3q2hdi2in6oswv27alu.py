
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 36710400}}
)
@triton.jit
def triton_per_fused_add_addmm_convolution_native_group_norm_native_layer_norm_silu_split_with_sizes_transpose_view_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
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
