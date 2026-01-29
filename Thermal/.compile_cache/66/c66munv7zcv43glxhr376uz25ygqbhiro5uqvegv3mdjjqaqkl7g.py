
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 167772160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_mul_native_group_norm_native_group_norm_backward_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 1024*y3), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x2 + 1024*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (y3 // 4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0 + 128*x2 + 131072*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (y3 // 4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (y3 // 4), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13 * tmp7
    tmp15 = tmp14 * tmp7
    tmp16 = tmp15 * tmp7
    tmp17 = 0.000244140625
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp10 + tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 1024*y3), tmp23, xmask)
