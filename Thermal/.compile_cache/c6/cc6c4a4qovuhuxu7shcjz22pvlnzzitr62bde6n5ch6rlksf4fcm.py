
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 335872}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_convolution_native_layer_norm_silu_transpose_view_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp9 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = (tmp22 / tmp24)
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = (tmp31 / tmp33)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 512*x0), tmp43, xmask)
