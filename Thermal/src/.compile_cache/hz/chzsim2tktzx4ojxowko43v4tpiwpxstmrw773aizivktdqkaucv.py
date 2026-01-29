# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: ./.compile_cache/dz/cdzfimgmtg2oaefuhhyjjunyogqgcz5ppqmjjlbovmtkjmr2qvvk.py
# Topologically Sorted Source Nodes: [sigma, wrapped_sqrt, mul, mul_1, x_next], Original ATen: [aten.view, aten.sqrt, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   sigma => view
#   wrapped_sqrt => full_default
#   x_next => add
# Graph fragment:
#   %arg2_1 : Tensor "f32[30, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg1_1 : Tensor "f32[30][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %randn : Tensor "f32[30, 4, 32, 32][4096, 1024, 32, 1]cuda:0" = PlaceHolder[target=randn]
#   %view : Tensor "f32[30, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [-1, 1, 1, 1]), kwargs = {})
#   %full_default : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.2581988897471611), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul : Tensor "f32[30, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %full_default), kwargs = {})
#   %mul_1 : Tensor "f32[30, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %randn), kwargs = {})
#   %add : Tensor "f32[30, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mul_1), kwargs = {})
#   return %add
triton_poi_fused_add_mul_sqrt_view_0 = async_compile.triton('triton_poi_fused_add_mul_sqrt_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sqrt_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1966080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sqrt_view_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = 0.2581988897471611
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')

def partition_0(args):
    arg2_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg2_1, (30, 4, 32, 32), (4096, 1024, 32, 1))
    assert_size_stride(arg1_1, (30, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [noise], Original ATen: [aten.randn_like]
        buf0 = torch.ops.aten.randn.default([30, 4, 32, 32], dtype=torch.float32, device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        assert_size_stride(buf1, (30, 4, 32, 32), (4096, 1024, 32, 1), 'torch.ops.aten.randn.default')
        assert_alignment(buf1, 16, 'torch.ops.aten.randn.default')
        del buf0
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [sigma, wrapped_sqrt, mul, mul_1, x_next], Original ATen: [aten.view, aten.sqrt, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sqrt_view_0.run(buf2, arg2_1, arg1_1, 122880, stream=stream0)
        del arg1_1
        del arg2_1
    return (buf2, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        partition0_args = [arg2_1, arg1_1]
        del arg2_1, arg1_1
        (buf2,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf2, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((30, 4, 32, 32), (4096, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
