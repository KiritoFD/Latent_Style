
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*fp32', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'in_ptr14': '*fp32', 'in_ptr15': '*bf16', 'in_ptr16': '*bf16', 'in_ptr17': '*fp32', 'in_ptr18': '*bf16', 'in_ptr19': '*bf16', 'in_ptr20': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_84', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18841600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_cat_clone_transpose_view_84(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3200)
    x1 = xindex // 3200
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1536, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 3072, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 3200, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp11, tmp15, tmp16)
    tmp18 = tl.where(tmp9, tmp10, tmp17)
    tmp19 = tl.where(tmp4, tmp5, tmp18)
    tmp20 = tl.load(in_ptr3 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp11, tmp23, tmp24)
    tmp26 = tl.where(tmp9, tmp21, tmp25)
    tmp27 = tl.where(tmp4, tmp20, tmp26)
    tmp28 = tmp19 + tmp27
    tmp29 = tl.load(in_ptr6 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr7 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr8 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp11, tmp32, tmp33)
    tmp35 = tl.where(tmp9, tmp30, tmp34)
    tmp36 = tl.where(tmp4, tmp29, tmp35)
    tmp37 = tmp28 + tmp36
    tmp38 = tl.load(in_ptr9 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr10 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr11 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp11, tmp41, tmp42)
    tmp44 = tl.where(tmp9, tmp39, tmp43)
    tmp45 = tl.where(tmp4, tmp38, tmp44)
    tmp46 = tmp37 + tmp45
    tmp47 = tl.load(in_ptr12 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr13 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp49 = tl.load(in_ptr14 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp11, tmp50, tmp51)
    tmp53 = tl.where(tmp9, tmp48, tmp52)
    tmp54 = tl.where(tmp4, tmp47, tmp53)
    tmp55 = tmp46 + tmp54
    tmp56 = tl.load(in_ptr15 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp57 = tl.load(in_ptr16 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp58 = tl.load(in_ptr17 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp11, tmp59, tmp60)
    tmp62 = tl.where(tmp9, tmp57, tmp61)
    tmp63 = tl.where(tmp4, tmp56, tmp62)
    tmp64 = tmp55 + tmp63
    tmp65 = tl.load(in_ptr18 + (1536*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp66 = tl.load(in_ptr19 + (128*((((-1536) + x0) % 12)) + 1536*x1 + (((((-1536) + x0) // 12) % 128))), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr20 + (128*x1 + ((-3072) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp11, tmp68, tmp69)
    tmp71 = tl.where(tmp9, tmp66, tmp70)
    tmp72 = tl.where(tmp4, tmp65, tmp71)
    tmp73 = tmp64 + tmp72
    tl.store(in_out_ptr0 + (x2), tmp73, None)
