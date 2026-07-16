# AOT ID: ['0_inference']
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
_frozen_param0 = None  # device(type='cuda', index=0) torch.float16 (4, 4, 1, 1) (4, 1, 4, 4) 22b2bb65360
_frozen_param2 = None  # device(type='cuda', index=0) torch.float16 (512, 4, 3, 3) (36, 1, 12, 4) 22b7eb1eb20
_frozen_param3 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2bb66120
_frozen_param6 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1cff0
_frozen_param7 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1cf50
_frozen_param10 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1e8a0
_frozen_param11 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1ea80
_frozen_param15 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e990
_frozen_param17 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e850
_frozen_param19 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1ceb0
_frozen_param21 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2bb66e40
_frozen_param24 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1cd70
_frozen_param25 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1cfa0
_frozen_param28 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1cbe0
_frozen_param29 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e6c0
_frozen_param32 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1e440
_frozen_param33 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1cd20
_frozen_param36 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1ca50
_frozen_param37 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e490
_frozen_param40 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1c960
_frozen_param41 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1caf0
_frozen_param44 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1e670
_frozen_param45 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1c9b0
_frozen_param48 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1f7a0
_frozen_param49 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e2b0
_frozen_param52 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2b9d7660
_frozen_param53 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1f7f0
_frozen_param54 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2b9d7840
_frozen_param55 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1c4b0
_frozen_param58 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7def0
_frozen_param59 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2b9d77a0
_frozen_param62 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f020
_frozen_param63 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7c870
_frozen_param66 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f2a0
_frozen_param67 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f070
_frozen_param70 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f480
_frozen_param71 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7eee0
_frozen_param74 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f5c0
_frozen_param75 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f2f0
_frozen_param78 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7fa70
_frozen_param79 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f9d0
_frozen_param80 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7fb60
_frozen_param81 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f660
_frozen_param84 = None  # device(type='cuda', index=0) torch.float16 (256, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7fcf0
_frozen_param85 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fb10
_frozen_param88 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2ba7fe80
_frozen_param89 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fca0
_frozen_param90 = None  # device(type='cuda', index=0) torch.float16 (256, 512, 1, 1) (512, 1, 512, 512) 22b2ba7ff70
_frozen_param91 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fe30
_frozen_param94 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb64140
_frozen_param95 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fd40
_frozen_param98 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb642d0
_frozen_param99 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb640f0
_frozen_param102 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb64460
_frozen_param103 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb64190
_frozen_param106 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb645f0
_frozen_param107 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb64410
_frozen_param108 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb646e0
_frozen_param109 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb644b0
_frozen_param112 = None  # device(type='cuda', index=0) torch.float16 (128, 256, 3, 3) (2304, 1, 768, 256) 22b2bb64820
_frozen_param113 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64690
_frozen_param116 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb649b0
_frozen_param117 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb647d0
_frozen_param118 = None  # device(type='cuda', index=0) torch.float16 (128, 256, 1, 1) (256, 1, 256, 256) 22b2bb65f40
_frozen_param119 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64960
_frozen_param122 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb64c30
_frozen_param123 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64870
_frozen_param126 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb64dc0
_frozen_param127 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64be0
_frozen_param130 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb64f50
_frozen_param131 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64c80
_frozen_param134 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb650e0
_frozen_param135 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64f00
_frozen_param140 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1b80
_frozen_param141 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f37f0
_frozen_param142 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1c70
_frozen_param143 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1cc0
_frozen_param144 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1) (512, 1, 1) 22b367f1a90
_frozen_param145 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1) (512, 1, 1) 22b367f19a0
_frozen_param146 = None  # device(type='cuda', index=0) torch.float16 (8, 512, 512) (0, 1, 512) 22b367f1130
_frozen_param147 = None  # device(type='cuda', index=0) torch.float16 (8, 512, 512) (0, 1, 512) 22b367f1a40
_frozen_param148 = None  # device(type='cuda', index=0) torch.float16 (8, 512, 512) (0, 1, 512) 22b367f1950
_frozen_param149 = None  # device(type='cuda', index=0) torch.float16 (512, 512) (1, 512) 22b367f1400
_frozen_param150 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0c80
_frozen_param151 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1b30
_frozen_param152 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1860
_frozen_param153 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1770
_frozen_param154 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1720
_frozen_param155 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1900
_frozen_param156 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f16d0
_frozen_param157 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1680
_frozen_param158 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1590
_frozen_param159 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1630
_frozen_param160 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f13b0
_frozen_param161 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f14f0
_frozen_param162 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0be0
_frozen_param163 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f14a0
_frozen_param164 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1ae0
_frozen_param165 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f19f0
_frozen_param166 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f15e0
_frozen_param167 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1310
_frozen_param168 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0cd0
_frozen_param169 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1360
_frozen_param170 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1450
_frozen_param171 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1220
_frozen_param172 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f11d0
_frozen_param173 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1180
_frozen_param174 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1090
_frozen_param175 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1810
_frozen_param176 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0ff0
_frozen_param177 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f2d00
_frozen_param178 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0f00
_frozen_param179 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0780
_frozen_param180 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f2c60
_frozen_param181 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f07d0
_frozen_param182 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f08c0
_frozen_param183 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f09b0
_frozen_param184 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0b40
_frozen_param185 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0a50
_frozen_param186 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0aa0
_frozen_param187 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0c30
_frozen_param188 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f2b70
_frozen_param189 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0d70
_frozen_param190 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f02d0
_frozen_param191 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f00a0
_frozen_param192 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f06e0
_frozen_param193 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2a30
_frozen_param194 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f12c0
_frozen_param195 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f0dc0
_frozen_param196 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f0af0
_frozen_param197 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f1040
_frozen_param198 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2a80
_frozen_param199 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2ad0
_frozen_param200 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2cb0
_frozen_param201 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f24e0
_frozen_param202 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2b20
_frozen_param203 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f34d0
_frozen_param205 = None  # device(type='cuda', index=0) torch.float16 (3, 128, 3, 3) (1152, 1, 384, 128) 22b368118b0
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

_frozen_param0 = None  # device(type='cuda', index=0) torch.float16 (4, 4, 1, 1) (4, 1, 4, 4) 22b2bb65360
_frozen_param2 = None  # device(type='cuda', index=0) torch.float16 (512, 4, 3, 3) (36, 1, 12, 4) 22b7eb1eb20
_frozen_param3 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2bb66120
_frozen_param6 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1cff0
_frozen_param7 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1cf50
_frozen_param10 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1e8a0
_frozen_param11 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1ea80
_frozen_param15 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e990
_frozen_param17 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e850
_frozen_param19 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1ceb0
_frozen_param21 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2bb66e40
_frozen_param24 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1cd70
_frozen_param25 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1cfa0
_frozen_param28 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1cbe0
_frozen_param29 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e6c0
_frozen_param32 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1e440
_frozen_param33 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1cd20
_frozen_param36 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1ca50
_frozen_param37 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e490
_frozen_param40 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1c960
_frozen_param41 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1caf0
_frozen_param44 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1e670
_frozen_param45 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1c9b0
_frozen_param48 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b7eb1f7a0
_frozen_param49 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1e2b0
_frozen_param52 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2b9d7660
_frozen_param53 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1f7f0
_frozen_param54 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2b9d7840
_frozen_param55 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b7eb1c4b0
_frozen_param58 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7def0
_frozen_param59 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2b9d77a0
_frozen_param62 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f020
_frozen_param63 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7c870
_frozen_param66 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f2a0
_frozen_param67 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f070
_frozen_param70 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f480
_frozen_param71 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7eee0
_frozen_param74 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7f5c0
_frozen_param75 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f2f0
_frozen_param78 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7fa70
_frozen_param79 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f9d0
_frozen_param80 = None  # device(type='cuda', index=0) torch.float16 (512, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7fb60
_frozen_param81 = None  # device(type='cuda', index=0) torch.float16 (512,) (1,) 22b2ba7f660
_frozen_param84 = None  # device(type='cuda', index=0) torch.float16 (256, 512, 3, 3) (4608, 1, 1536, 512) 22b2ba7fcf0
_frozen_param85 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fb10
_frozen_param88 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2ba7fe80
_frozen_param89 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fca0
_frozen_param90 = None  # device(type='cuda', index=0) torch.float16 (256, 512, 1, 1) (512, 1, 512, 512) 22b2ba7ff70
_frozen_param91 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fe30
_frozen_param94 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb64140
_frozen_param95 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2ba7fd40
_frozen_param98 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb642d0
_frozen_param99 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb640f0
_frozen_param102 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb64460
_frozen_param103 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb64190
_frozen_param106 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb645f0
_frozen_param107 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb64410
_frozen_param108 = None  # device(type='cuda', index=0) torch.float16 (256, 256, 3, 3) (2304, 1, 768, 256) 22b2bb646e0
_frozen_param109 = None  # device(type='cuda', index=0) torch.float16 (256,) (1,) 22b2bb644b0
_frozen_param112 = None  # device(type='cuda', index=0) torch.float16 (128, 256, 3, 3) (2304, 1, 768, 256) 22b2bb64820
_frozen_param113 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64690
_frozen_param116 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb649b0
_frozen_param117 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb647d0
_frozen_param118 = None  # device(type='cuda', index=0) torch.float16 (128, 256, 1, 1) (256, 1, 256, 256) 22b2bb65f40
_frozen_param119 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64960
_frozen_param122 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb64c30
_frozen_param123 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64870
_frozen_param126 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb64dc0
_frozen_param127 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64be0
_frozen_param130 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb64f50
_frozen_param131 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64c80
_frozen_param134 = None  # device(type='cuda', index=0) torch.float16 (128, 128, 3, 3) (1152, 1, 384, 128) 22b2bb650e0
_frozen_param135 = None  # device(type='cuda', index=0) torch.float16 (128,) (1,) 22b2bb64f00
_frozen_param140 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1b80
_frozen_param141 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f37f0
_frozen_param142 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1c70
_frozen_param143 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1cc0
_frozen_param144 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1) (512, 1, 1) 22b367f1a90
_frozen_param145 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1) (512, 1, 1) 22b367f19a0
_frozen_param146 = None  # device(type='cuda', index=0) torch.float16 (8, 512, 512) (0, 1, 512) 22b367f1130
_frozen_param147 = None  # device(type='cuda', index=0) torch.float16 (8, 512, 512) (0, 1, 512) 22b367f1a40
_frozen_param148 = None  # device(type='cuda', index=0) torch.float16 (8, 512, 512) (0, 1, 512) 22b367f1950
_frozen_param149 = None  # device(type='cuda', index=0) torch.float16 (512, 512) (1, 512) 22b367f1400
_frozen_param150 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0c80
_frozen_param151 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1b30
_frozen_param152 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1860
_frozen_param153 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1770
_frozen_param154 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1720
_frozen_param155 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1900
_frozen_param156 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f16d0
_frozen_param157 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1680
_frozen_param158 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1590
_frozen_param159 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1630
_frozen_param160 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f13b0
_frozen_param161 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f14f0
_frozen_param162 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0be0
_frozen_param163 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f14a0
_frozen_param164 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1ae0
_frozen_param165 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f19f0
_frozen_param166 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f15e0
_frozen_param167 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1310
_frozen_param168 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0cd0
_frozen_param169 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1360
_frozen_param170 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1450
_frozen_param171 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1220
_frozen_param172 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f11d0
_frozen_param173 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1180
_frozen_param174 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1090
_frozen_param175 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f1810
_frozen_param176 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0ff0
_frozen_param177 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f2d00
_frozen_param178 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0f00
_frozen_param179 = None  # device(type='cuda', index=0) torch.float16 (1, 512, 1, 1) (512, 1, 1, 1) 22b367f0780
_frozen_param180 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f2c60
_frozen_param181 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f07d0
_frozen_param182 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f08c0
_frozen_param183 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f09b0
_frozen_param184 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0b40
_frozen_param185 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0a50
_frozen_param186 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0aa0
_frozen_param187 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0c30
_frozen_param188 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f2b70
_frozen_param189 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f0d70
_frozen_param190 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f02d0
_frozen_param191 = None  # device(type='cuda', index=0) torch.float16 (1, 256, 1, 1) (256, 1, 1, 1) 22b367f00a0
_frozen_param192 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f06e0
_frozen_param193 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2a30
_frozen_param194 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f12c0
_frozen_param195 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f0dc0
_frozen_param196 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f0af0
_frozen_param197 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f1040
_frozen_param198 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2a80
_frozen_param199 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2ad0
_frozen_param200 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2cb0
_frozen_param201 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f24e0
_frozen_param202 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f2b20
_frozen_param203 = None  # device(type='cuda', index=0) torch.float16 (1, 128, 1, 1) (128, 1, 1, 1) 22b367f34d0
_frozen_param205 = None  # device(type='cuda', index=0) torch.float16 (3, 128, 3, 3) (1152, 1, 384, 128) 22b368118b0


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\bi\cbi5cunxeiselitrdw3uuyvvbjlm7blcgpmpoms243rvmrc7axdt.py
# Topologically Sorted Source Nodes: [z, z_1], Original ATen: [aten.div, aten.convolution]
# Source node to ATen node mapping:
#   z => div
#   z_1 => convolution
# Graph fragment:
#   %arg140_1 : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0" = PlaceHolder[target=arg140_1]
#   %div : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg140_1, 0.18215), kwargs = {})
#   %convolution : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_div_0 = async_compile.triton('triton_poi_fused_convolution_div_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_div_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 5.489980785067252, tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\qq\cqqejdg3w7xrdb6firttlohtmq6xotblbtsylpwhbnyxe7la47hc.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], -0.0128173828125, tl.float32)
    tmp6 = tl.full([1], -0.07159423828125, tl.float32)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1], 3, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.full([1], -0.1956787109375, tl.float32)
    tmp11 = tl.full([1], 0.13525390625, tl.float32)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp7, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\32\c323gvdmhyc4hyv4gacgidpesk5crzsompvodot746wuy224fcls.py
# Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states], Original ATen: [aten.div, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => clone, convert_element_type, var_mean, view
#   sample => convolution_1
#   z => div
#   z_1 => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf3]
#   %_frozen_param3 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param3]
#   %div : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg140_1, 0.18215), kwargs = {})
#   %convolution : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg2_1, %arg3_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %view : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf5
triton_red_fused_clone_convolution_div_native_group_norm_2 = async_compile.triton('triton_red_fused_clone_convolution_div_native_group_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_convolution_div_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 33555456}}
)
@triton.jit
def triton_red_fused_clone_convolution_div_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 65536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 2097152*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ta\ctajocrperq3aw4elemum3ionwub6rfvccmfs64aqnfgmhuonhpd.py
# Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states, hidden_states_1], Original ATen: [aten.div, aten.convolution, aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states => add, add_1, clone, convert_element_type, mul, mul_1, rsqrt, sub, var_mean, view, view_1
#   hidden_states_1 => add_2, convert_element_type_5, div_1, exp, neg
#   sample => convolution_1
#   z => div
#   z_1 => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf3]
#   %_frozen_param3 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param3]
#   %getitem_1 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf5 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf5]
#   %_frozen_param140 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param140]
#   %_frozen_param141 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param141]
#   %add_1 : Tensor "f32[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=add_1]
#   %div : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg140_1, 0.18215), kwargs = {})
#   %convolution : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg2_1, %arg3_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %view : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [8, 512, 64, 64]), kwargs = {})
#   %mul_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_2), kwargs = {})
#   %add_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %neg : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_1,), kwargs = {})
#   %exp : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_2 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1, %add_2), kwargs = {})
#   %convert_element_type_5 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.float16), kwargs = {})
#   return %add_1,%convert_element_type_5
triton_poi_fused_clone_convolution_div_native_group_norm_silu_3 = async_compile.triton('triton_poi_fused_clone_convolution_div_native_group_norm_silu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_div_native_group_norm_silu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 100668416}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_div_native_group_norm_silu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 2097152
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = tl.full([1], 65536.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = -tmp18
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.full([1], 1.0, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = (tmp18 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ar\carfaelif52re35x44glrhkreclezgij3s5hymx3pygrek3526kj.py
# Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add => add_6
#   group_norm_2 => clone_3, convert_element_type_12, var_mean_2, view_5
#   hidden_states_4 => add_5, convert_element_type_11, div_2, exp_1, neg_1
#   hidden_states_6 => convolution_3
#   output_tensor => div_3
#   sample => convolution_1
#   view => view_4
#   z => div
#   z_1 => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf3]
#   %_frozen_param3 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param3]
#   %buf15 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf15]
#   %_frozen_param11 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param11]
#   %div : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg140_1, 0.18215), kwargs = {})
#   %convolution : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg2_1, %arg3_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_4,), kwargs = {})
#   %exp_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add_5 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_2 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, %add_5), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float16), kwargs = {})
#   %convolution_3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div_3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_6, 1), kwargs = {})
#   %view_4 : Tensor "f16[8, 512, 4096][2097152, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div_3, [8, 512, 4096]), kwargs = {})
#   %clone_3 : Tensor "f16[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_4,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_12 : Tensor "f32[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_3, torch.float32), kwargs = {})
#   %view_5 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_12, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_5,%buf17
triton_red_fused_add_clone_convolution_div_native_group_norm_silu_view_4 = async_compile.triton('triton_red_fused_add_clone_convolution_div_native_group_norm_silu_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_native_group_norm_silu_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 67110912}}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_native_group_norm_silu_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 65536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 2097152*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 16*x0 + 512*r0_3 + 2097152*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(xmask, tmp11_weight_next, tmp11_weight)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp11_mean, tmp11_m2, tmp11_weight, 1)
    tmp11 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    tl.store(out_ptr0 + (x4), tmp11, xmask)
    tl.store(out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\57\c57glptrzzwx2giytfpka7fvs5knq6iatwmhourfhjmuhowv4ulc.py
# Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add => add_6
#   group_norm_2 => add_7, add_8, clone_3, convert_element_type_12, convert_element_type_13, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2, view_5, view_6
#   hidden_states_4 => add_5, convert_element_type_11, div_2, exp_1, neg_1
#   hidden_states_6 => convolution_3
#   output_tensor => div_3
#   sample => convolution_1
#   view => view_4
#   z => div
#   z_1 => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf3]
#   %_frozen_param3 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param3]
#   %buf15 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf15]
#   %_frozen_param11 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param11]
#   %getitem_5 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf17 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf17]
#   %_frozen_param144 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param144]
#   %_frozen_param145 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param145]
#   %div : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg140_1, 0.18215), kwargs = {})
#   %convolution : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg2_1, %arg3_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_4,), kwargs = {})
#   %exp_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add_5 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_2 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, %add_5), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float16), kwargs = {})
#   %convolution_3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div_3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_6, 1), kwargs = {})
#   %view_4 : Tensor "f16[8, 512, 4096][2097152, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div_3, [8, 512, 4096]), kwargs = {})
#   %clone_3 : Tensor "f16[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_4,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_12 : Tensor "f32[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_3, torch.float32), kwargs = {})
#   %view_5 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_12, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %getitem_5), kwargs = {})
#   %add_7 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_4 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %view_6 : Tensor "f32[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_4, [8, 512, 4096]), kwargs = {})
#   %mul_5 : Tensor "f32[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %unsqueeze_13), kwargs = {})
#   %add_8 : Tensor "f32[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %convert_element_type_13 : Tensor "f16[8, 512, 4096][2097152, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float16), kwargs = {})
#   return %convert_element_type_13
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_view_5 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 67108864, 'x': 67115008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 4096
    y0 = (yindex % 4096)
    tmp0 = tl.load(in_ptr0 + (x2 + 512*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 512*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (32*y1 + (x2 // 16)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (32*y1 + (x2 // 16)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.full([1, 1], 1.0, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = tl.full([1, 1], 65536.0, tl.float32)
    tmp14 = (tmp12 / tmp13)
    tmp15 = tl.full([1, 1], 1e-06, tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4096*x2 + 2097152*y1), tmp25, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\uk\cukwnua4crukuwl23ytbelsvjcz4tviyyw6ba5feqbltgzgsjibf.py
# Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   hidden_states_9 => _scaled_dot_product_efficient_attention
#   key => add_10
#   key_1 => permute_7
#   query => add_9
#   query_1 => permute_6
#   value => add_11
#   value_1 => permute_8
#   view_1 => view_16
#   view_2 => view_17
#   view_3 => view_18
# Graph fragment:
#   %bmm : Tensor "f16[8, 4096, 512][2097152, 512, 1]cuda:0" = PlaceHolder[target=bmm]
#   %_frozen_param15 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param15]
#   %add_9 : Tensor "f16[8, 4096, 512][2097152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm, %arg15_1), kwargs = {})
#   %view_16 : Tensor "f16[8, 4096, 1, 512][2097152, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_9, [8, -1, 1, 512]), kwargs = {})
#   %permute_6 : Tensor "f16[8, 1, 4096, 512][2097152, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_16, [0, 2, 1, 3]), kwargs = {})
#   %add_10 : Tensor "f16[8, 4096, 512][2097152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_1, %arg17_1), kwargs = {})
#   %view_17 : Tensor "f16[8, 4096, 1, 512][2097152, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_10, [8, -1, 1, 512]), kwargs = {})
#   %permute_7 : Tensor "f16[8, 1, 4096, 512][2097152, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_17, [0, 2, 1, 3]), kwargs = {})
#   %add_11 : Tensor "f16[8, 4096, 512][2097152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_2, %arg19_1), kwargs = {})
#   %view_18 : Tensor "f16[8, 4096, 1, 512][2097152, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_11, [8, -1, 1, 512]), kwargs = {})
#   %permute_8 : Tensor "f16[8, 1, 4096, 512][2097152, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_6, %permute_7, %permute_8, None, False), kwargs = {})
#   return %buf23
triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 100664320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\gi\cgilwdwxq4frfaspp2yyld2ufo4uxfmk4jwcipe44aru43d7yi5r.py
# Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_12, transpose_7, hidden_states_14, hidden_states_15, hidden_states_16], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.transpose]
# Source node to ATen node mapping:
#   add => add_6
#   hidden_states_12 => view_21
#   hidden_states_14 => view_22
#   hidden_states_15 => add_12
#   hidden_states_16 => div_4
#   hidden_states_4 => add_5, convert_element_type_11, div_2, exp_1, neg_1
#   hidden_states_6 => convolution_3
#   output_tensor => div_3
#   sample => convolution_1
#   transpose_7 => permute_11
#   z => div
#   z_1 => convolution
# Graph fragment:
#   %addmm : Tensor "f16[32768, 512][512, 1]cuda:0" = PlaceHolder[target=addmm]
#   %buf3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf3]
#   %_frozen_param3 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param3]
#   %buf15 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf15]
#   %_frozen_param11 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param11]
#   %div : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg140_1, 0.18215), kwargs = {})
#   %convolution : Tensor "f16[8, 4, 64, 64][16384, 1, 256, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg2_1, %arg3_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_4,), kwargs = {})
#   %exp_1 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add_5 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_2 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, %add_5), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float16), kwargs = {})
#   %convolution_3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_6 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div_3 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_6, 1), kwargs = {})
#   %view_21 : Tensor "f16[8, 4096, 512][2097152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [8, 4096, 512]), kwargs = {})
#   %permute_11 : Tensor "f16[8, 512, 4096][2097152, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1]), kwargs = {})
#   %view_22 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_11, [8, 512, 64, 64]), kwargs = {})
#   %add_12 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_22, %div_3), kwargs = {})
#   %div_4 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_12, 1), kwargs = {})
#   return %div_4
triton_poi_fused_add_convolution_div_silu_transpose_view_7 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_transpose_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_transpose_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 167774208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_transpose_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 + tmp9
    tmp11 = tmp10 * tmp8
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\rj\crj3it6e3o5e4cgmvz5pxefj2ko6tjvoiqmysewskflfgwxpo4rc.py
# Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_17 => clone_5, convert_element_type_25, var_mean_3, view_23
# Graph fragment:
#   %div_4 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=div_4]
#   %clone_5 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_4,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_25 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.float32), kwargs = {})
#   %view_23 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_25, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_23, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_11,%buf34
triton_red_fused_clone_native_group_norm_8 = async_compile.triton('triton_red_fused_clone_native_group_norm_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 33554432}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_8(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 65536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 2097152*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\53\c532xqxdz2vsfndbaqn2qvyyy2tx2dsxuabxkkjbtvbev7nigsfu.py
# Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_17 => add_13, add_14, clone_5, convert_element_type_25, mul_6, mul_7, rsqrt_3, sub_3, var_mean_3, view_23, view_24
#   hidden_states_18 => add_15, convert_element_type_30, div_5, exp_2, neg_2
# Graph fragment:
#   %div_4 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=div_4]
#   %getitem_11 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_11]
#   %buf34 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf34]
#   %_frozen_param150 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param150]
#   %_frozen_param151 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param151]
#   %add_14 : Tensor "f32[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=add_14]
#   %clone_5 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_4,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_25 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.float32), kwargs = {})
#   %view_23 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_25, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_23, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_23, %getitem_11), kwargs = {})
#   %add_13 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_6 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %view_24 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_6, [8, 512, 64, 64]), kwargs = {})
#   %mul_7 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_24, %unsqueeze_18), kwargs = {})
#   %add_14 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %neg_2 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_14,), kwargs = {})
#   %exp_2 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_15 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_2, 1), kwargs = {})
#   %div_5 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_14, %add_15), kwargs = {})
#   %convert_element_type_30 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_5, torch.float16), kwargs = {})
#   return %add_14,%convert_element_type_30
triton_poi_fused_clone_native_group_norm_silu_9 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 100667392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 2097152
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 65536.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = -tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.full([1], 1.0, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp16 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\di\cdicmdos5fua5tvuy72n2dq5tiag3roiom3jstky4y3r37er432u.py
# Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_2 => add_19
#   hidden_states_21 => add_18, convert_element_type_36, div_6, exp_3, neg_3
#   hidden_states_23 => convolution_5
#   hidden_states_24 => clone_8, convert_element_type_37, var_mean_5, view_27
#   output_tensor_1 => div_7
# Graph fragment:
#   %div_4 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=div_4]
#   %buf44 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf44]
#   %_frozen_param29 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param29]
#   %neg_3 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_17,), kwargs = {})
#   %exp_3 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_3,), kwargs = {})
#   %add_18 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_3, 1), kwargs = {})
#   %div_6 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_17, %add_18), kwargs = {})
#   %convert_element_type_36 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.float16), kwargs = {})
#   %convolution_5 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_36, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_19 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_5), kwargs = {})
#   %div_7 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_19, 1), kwargs = {})
#   %clone_8 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_7,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_37 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_8, torch.float32), kwargs = {})
#   %view_27 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_37, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_27, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_15,%buf46
triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10 = async_compile.triton('triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 67109888}}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 65536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 2097152*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 16*x0 + 512*r0_3 + 2097152*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
    tl.store(out_ptr1 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\5t\c5tjg3yfwcq2ihqwd5kkvl76uhq3duc66gjnyg57lafzirpq6l7y.py
# Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_24, hidden_states_25], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_2 => add_19
#   hidden_states_21 => add_18, convert_element_type_36, div_6, exp_3, neg_3
#   hidden_states_23 => convolution_5
#   hidden_states_24 => add_20, add_21, clone_8, convert_element_type_37, mul_10, mul_11, rsqrt_5, sub_5, var_mean_5, view_27, view_28
#   hidden_states_25 => add_22, convert_element_type_42, div_8, exp_4, neg_4
#   output_tensor_1 => div_7
# Graph fragment:
#   %div_4 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=div_4]
#   %buf44 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf44]
#   %_frozen_param29 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param29]
#   %getitem_15 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf46 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf46]
#   %_frozen_param154 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param154]
#   %_frozen_param155 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param155]
#   %add_21 : Tensor "f32[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=add_21]
#   %neg_3 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_17,), kwargs = {})
#   %exp_3 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_3,), kwargs = {})
#   %add_18 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_3, 1), kwargs = {})
#   %div_6 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_17, %add_18), kwargs = {})
#   %convert_element_type_36 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.float16), kwargs = {})
#   %convolution_5 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_36, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_19 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_5), kwargs = {})
#   %div_7 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_19, 1), kwargs = {})
#   %clone_8 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_7,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_37 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_8, torch.float32), kwargs = {})
#   %view_27 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_37, [8, 32, 16, 4096]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_27, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_27, %getitem_15), kwargs = {})
#   %add_20 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
#   %mul_10 : Tensor "f32[8, 32, 16, 4096][2097152, 65536, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %view_28 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_10, [8, 512, 64, 64]), kwargs = {})
#   %mul_11 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %unsqueeze_30), kwargs = {})
#   %add_21 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_33), kwargs = {})
#   %neg_4 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_21,), kwargs = {})
#   %exp_4 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_4,), kwargs = {})
#   %add_22 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_4, 1), kwargs = {})
#   %div_8 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_21, %add_22), kwargs = {})
#   %convert_element_type_42 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.float16), kwargs = {})
#   return %add_21,%convert_element_type_42
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 134222848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 2097152
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = tl.full([1], 65536.0, tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tl.full([1], 1e-06, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = -tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tmp24 + tmp5
    tmp26 = (tmp22 / tmp25)
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ju\cjukthq2g4ax7hv4npu57ucfntyy5zncopvp3rn4xpm6pemd4ggw.py
# Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_28, hidden_states_30, add_3, output_tensor_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_2 => add_19
#   add_3 => add_26
#   hidden_states_21 => add_18, convert_element_type_36, div_6, exp_3, neg_3
#   hidden_states_23 => convolution_5
#   hidden_states_28 => add_25, convert_element_type_48, div_9, exp_5, neg_5
#   hidden_states_30 => convolution_7
#   output_tensor_1 => div_7
#   output_tensor_2 => div_10
# Graph fragment:
#   %div_4 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=div_4]
#   %buf44 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf44]
#   %_frozen_param29 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param29]
#   %buf56 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf56]
#   %_frozen_param37 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param37]
#   %neg_3 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_17,), kwargs = {})
#   %exp_3 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_3,), kwargs = {})
#   %add_18 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_3, 1), kwargs = {})
#   %div_6 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_17, %add_18), kwargs = {})
#   %convert_element_type_36 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.float16), kwargs = {})
#   %convolution_5 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_36, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_19 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_5), kwargs = {})
#   %div_7 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_19, 1), kwargs = {})
#   %neg_5 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_24,), kwargs = {})
#   %exp_5 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_5,), kwargs = {})
#   %add_25 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_5, 1), kwargs = {})
#   %div_9 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_24, %add_25), kwargs = {})
#   %convert_element_type_48 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_9, torch.float16), kwargs = {})
#   %convolution_7 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_48, %arg36_1, %arg37_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_26 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_7), kwargs = {})
#   %div_10 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_26, 1.0), kwargs = {})
#   return %div_10
triton_poi_fused_add_convolution_div_silu_12 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 167774208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\m6\cm6oirlen53ltrdgpvcui6vxhpo7lboired7f6lhsw2mdedjfvty.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_4 => add_33
#   add_5 => add_40
#   hidden_states_35 => add_32, convert_element_type_60, div_12, exp_7, neg_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => add_39, convert_element_type_72, div_15, exp_9, neg_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_41, add_42, clone_17, convert_element_type_73, convert_element_type_74, convert_element_type_75, convert_element_type_78, iota, mul_22, mul_23, unsqueeze_64
#   output_tensor_3 => div_13
#   output_tensor_4 => div_16
# Graph fragment:
#   %div_10 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=div_10]
#   %buf69 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf69]
#   %_frozen_param45 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param45]
#   %buf81 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0" = PlaceHolder[target=buf81]
#   %_frozen_param53 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param53]
#   %_unsafe_index : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=_unsafe_index]
#   %neg_7 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_31,), kwargs = {})
#   %exp_7 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_7,), kwargs = {})
#   %add_32 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_7, 1), kwargs = {})
#   %div_12 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_31, %add_32), kwargs = {})
#   %convert_element_type_60 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_12, torch.float16), kwargs = {})
#   %convolution_9 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_60, %arg44_1, %arg45_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_33 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_10, %convolution_9), kwargs = {})
#   %div_13 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_33, 1.0), kwargs = {})
#   %neg_9 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_38,), kwargs = {})
#   %exp_9 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_9,), kwargs = {})
#   %add_39 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_9, 1), kwargs = {})
#   %div_15 : Tensor "f32[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_38, %add_39), kwargs = {})
#   %convert_element_type_72 : Tensor "f16[8, 512, 64, 64][2097152, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_15, torch.float16), kwargs = {})
#   %convolution_11 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_72, %arg52_1, %arg53_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_40 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %convolution_11), kwargs = {})
#   %div_16 : Tensor "f16[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_40, 1.0), kwargs = {})
#   %convert_element_type_73 : Tensor "f32[8, 512, 64, 64][2097152, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_16, torch.float32), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_22 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_41 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, 0), kwargs = {})
#   %convert_element_type_74 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_41, torch.float32), kwargs = {})
#   %add_42 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_74, 0.0), kwargs = {})
#   %mul_23 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, 0.5), kwargs = {})
#   %convert_element_type_75 : Tensor "i64[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_75, -1), kwargs = {})
#   %_unsafe_index : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_73, [None, None, %unsqueeze_64, %convert_element_type_75]), kwargs = {})
#   %clone_17 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.float16), kwargs = {})
#   return %_unsafe_index,%convert_element_type_78
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_13 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 268437504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 65536) % 128)
    x1 = ((xindex // 512) % 128)
    x0 = (xindex % 512)
    x3 = xindex // 8388608
    x4 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 512*tmp8 + 32768*tmp4 + 2097152*x3), None).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x0 + 512*tmp8 + 32768*tmp4 + 2097152*x3), None).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full([1], 1.0, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.load(in_ptr3 + (x0 + 512*tmp8 + 32768*tmp4 + 2097152*x3), None).to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp19 * tmp14
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp22, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ja\cjabhbn7bubuossqmfgcv6novhwzgeakzarq5fpa4hfahrhp7aa7.py
# Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_45 => clone_17, convert_element_type_78
#   hidden_states_46 => convolution_12
#   hidden_states_47 => clone_18, convert_element_type_79, var_mean_11, view_39
# Graph fragment:
#   %buf84 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf84]
#   %_frozen_param55 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param55]
#   %clone_17 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.float16), kwargs = {})
#   %convolution_12 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_78, %arg54_1, %arg55_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_18 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_12,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_79 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_18, torch.float32), kwargs = {})
#   %view_39 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_79, [8, 32, 16, 16384]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_39, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_27,%buf86
triton_red_fused__to_copy_clone_convolution_native_group_norm_14 = async_compile.triton('triton_red_fused__to_copy_clone_convolution_native_group_norm_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_convolution_native_group_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 134218752}}
)
@triton.jit
def triton_red_fused__to_copy_clone_convolution_native_group_norm_14(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 8388608*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\qm\cqmwercldk75igdq5n2y2bk2rm5sye2o4mulnj5snbvbzp3mahgx.py
# Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_47, hidden_states_48], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_45 => clone_17, convert_element_type_78
#   hidden_states_46 => convolution_12
#   hidden_states_47 => add_45, add_46, clone_18, convert_element_type_79, mul_26, mul_27, rsqrt_11, sub_11, var_mean_11, view_39, view_40
#   hidden_states_48 => add_47, convert_element_type_84, div_17, exp_10, neg_10
# Graph fragment:
#   %buf84 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf84]
#   %_frozen_param55 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param55]
#   %getitem_27 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf86 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf86]
#   %_frozen_param166 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param166]
#   %_frozen_param167 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param167]
#   %add_46 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_46]
#   %clone_17 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.float16), kwargs = {})
#   %convolution_12 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_78, %arg54_1, %arg55_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_18 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_12,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_79 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_18, torch.float32), kwargs = {})
#   %view_39 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_79, [8, 32, 16, 16384]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_39, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_39, %getitem_27), kwargs = {})
#   %add_45 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_11 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_26 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %view_40 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_26, [8, 512, 128, 128]), kwargs = {})
#   %mul_27 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_40, %unsqueeze_67), kwargs = {})
#   %add_46 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_70), kwargs = {})
#   %neg_10 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_46,), kwargs = {})
#   %exp_10 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_10,), kwargs = {})
#   %add_47 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_10, 1), kwargs = {})
#   %div_17 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_46, %add_47), kwargs = {})
#   %convert_element_type_84 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_17, torch.float16), kwargs = {})
#   return %add_46,%convert_element_type_84
triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15 = async_compile.triton('triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 402658304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 8388608
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = tl.full([1], 262144.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = -tmp18
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.full([1], 1.0, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = (tmp18 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\g5\cg5gaqpodnnyoff7if5wjdqaowdjkdscuq2gzqzhljovbwooc5sr.py
# Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_6 => add_51
#   hidden_states_45 => clone_17, convert_element_type_78
#   hidden_states_46 => convolution_12
#   hidden_states_51 => add_50, convert_element_type_90, div_18, exp_11, neg_11
#   hidden_states_53 => convolution_14
#   hidden_states_54 => clone_21, convert_element_type_91, var_mean_13, view_43
#   output_tensor_5 => div_19
# Graph fragment:
#   %buf84 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf84]
#   %_frozen_param55 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param55]
#   %buf96 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf96]
#   %_frozen_param63 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param63]
#   %clone_17 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.float16), kwargs = {})
#   %convolution_12 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_78, %arg54_1, %arg55_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_11 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_49,), kwargs = {})
#   %exp_11 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_11,), kwargs = {})
#   %add_50 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_11, 1), kwargs = {})
#   %div_18 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, %add_50), kwargs = {})
#   %convert_element_type_90 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_18, torch.float16), kwargs = {})
#   %convolution_14 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %arg62_1, %arg63_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_51 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_14), kwargs = {})
#   %div_19 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_51, 1.0), kwargs = {})
#   %clone_21 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_91 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_21, torch.float32), kwargs = {})
#   %view_43 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_91, [8, 32, 16, 16384]), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_43, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_31,%buf98
triton_red_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16 = async_compile.triton('triton_red_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 268437504}}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 8388608*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 16*x0 + 512*r0_3 + 8388608*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 16*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(xmask, tmp11_weight_next, tmp11_weight)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp11_mean, tmp11_m2, tmp11_weight, 1)
    tmp11 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    tl.store(out_ptr0 + (x4), tmp11, xmask)
    tl.store(out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\22\c22p3qac3cesmp5oanxunuirfbd6lb7nf527cnwpvtwmwo3wzmef.py
# Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54, hidden_states_55], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_6 => add_51
#   hidden_states_45 => clone_17, convert_element_type_78
#   hidden_states_46 => convolution_12
#   hidden_states_51 => add_50, convert_element_type_90, div_18, exp_11, neg_11
#   hidden_states_53 => convolution_14
#   hidden_states_54 => add_52, add_53, clone_21, convert_element_type_91, mul_30, mul_31, rsqrt_13, sub_13, var_mean_13, view_43, view_44
#   hidden_states_55 => add_54, convert_element_type_96, div_20, exp_12, neg_12
#   output_tensor_5 => div_19
# Graph fragment:
#   %buf84 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf84]
#   %_frozen_param55 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param55]
#   %buf96 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf96]
#   %_frozen_param63 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param63]
#   %getitem_31 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_31]
#   %buf98 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf98]
#   %_frozen_param170 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param170]
#   %_frozen_param171 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param171]
#   %add_53 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_53]
#   %clone_17 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.float16), kwargs = {})
#   %convolution_12 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_78, %arg54_1, %arg55_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_11 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_49,), kwargs = {})
#   %exp_11 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_11,), kwargs = {})
#   %add_50 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_11, 1), kwargs = {})
#   %div_18 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, %add_50), kwargs = {})
#   %convert_element_type_90 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_18, torch.float16), kwargs = {})
#   %convolution_14 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %arg62_1, %arg63_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_51 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_14), kwargs = {})
#   %div_19 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_51, 1.0), kwargs = {})
#   %clone_21 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_91 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_21, torch.float32), kwargs = {})
#   %view_43 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_91, [8, 32, 16, 16384]), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_43, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_13 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_43, %getitem_31), kwargs = {})
#   %add_52 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-06), kwargs = {})
#   %rsqrt_13 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_52,), kwargs = {})
#   %mul_30 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_13), kwargs = {})
#   %view_44 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_30, [8, 512, 128, 128]), kwargs = {})
#   %mul_31 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_44, %unsqueeze_79), kwargs = {})
#   %add_53 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_82), kwargs = {})
#   %neg_12 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_53,), kwargs = {})
#   %exp_12 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_12,), kwargs = {})
#   %add_54 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1), kwargs = {})
#   %div_20 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_53, %add_54), kwargs = {})
#   %convert_element_type_96 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_20, torch.float16), kwargs = {})
#   return %add_53,%convert_element_type_96
triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 536877056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 8388608
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = tl.full([1], 262144.0, tl.float32)
    tmp14 = (tmp12 / tmp13)
    tmp15 = tl.full([1], 1e-06, tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = -tmp24
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tmp26 + tmp7
    tmp28 = (tmp24 / tmp27)
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\7d\c7duwbeqwbmxwlmjqqfs3kk7arg6ss47xdbzzc2doq6tmsuuvnue.py
# Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_58, hidden_states_60, add_7, output_tensor_6], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_6 => add_51
#   add_7 => add_58
#   hidden_states_45 => clone_17, convert_element_type_78
#   hidden_states_46 => convolution_12
#   hidden_states_51 => add_50, convert_element_type_90, div_18, exp_11, neg_11
#   hidden_states_53 => convolution_14
#   hidden_states_58 => add_57, convert_element_type_102, div_21, exp_13, neg_13
#   hidden_states_60 => convolution_16
#   output_tensor_5 => div_19
#   output_tensor_6 => div_22
# Graph fragment:
#   %buf84 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf84]
#   %_frozen_param55 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param55]
#   %buf96 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf96]
#   %_frozen_param63 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param63]
#   %buf108 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf108]
#   %_frozen_param71 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param71]
#   %clone_17 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_78 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_17, torch.float16), kwargs = {})
#   %convolution_12 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_78, %arg54_1, %arg55_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_11 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_49,), kwargs = {})
#   %exp_11 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_11,), kwargs = {})
#   %add_50 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_11, 1), kwargs = {})
#   %div_18 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, %add_50), kwargs = {})
#   %convert_element_type_90 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_18, torch.float16), kwargs = {})
#   %convolution_14 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_90, %arg62_1, %arg63_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_51 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_14), kwargs = {})
#   %div_19 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_51, 1.0), kwargs = {})
#   %neg_13 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_56,), kwargs = {})
#   %exp_13 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_13,), kwargs = {})
#   %add_57 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_13, 1), kwargs = {})
#   %div_21 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_56, %add_57), kwargs = {})
#   %convert_element_type_102 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_21, torch.float16), kwargs = {})
#   %convolution_16 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_102, %arg70_1, %arg71_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_19, %convolution_16), kwargs = {})
#   %div_22 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_58, 1.0), kwargs = {})
#   return %div_22
triton_poi_fused__to_copy_add_clone_convolution_div_silu_18 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_div_silu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_div_silu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 671091712}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_div_silu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp12 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\w5\cw5a2ucho6mdhrysxh7hbjdnlkmty2ykd4rtqudiqmgrs6zvkvni.py
# Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_61 => clone_24, convert_element_type_103, var_mean_15, view_47
# Graph fragment:
#   %div_22 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_22]
#   %clone_24 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_22,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_103 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_24, torch.float32), kwargs = {})
#   %view_47 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_103, [8, 32, 16, 16384]), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_47, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_35,%buf111
triton_red_fused_clone_native_group_norm_19 = async_compile.triton('triton_red_fused_clone_native_group_norm_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 134217728}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_19(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 8388608*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ct\cctvpeknim4ntvhgnfbq3xjxlvck4dihfvksf67lyjkmedl53yzt.py
# Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_61 => add_59, add_60, clone_24, convert_element_type_103, mul_34, mul_35, rsqrt_15, sub_15, var_mean_15, view_47, view_48
#   hidden_states_62 => add_61, convert_element_type_108, div_23, exp_14, neg_14
# Graph fragment:
#   %div_22 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_22]
#   %getitem_35 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_35]
#   %buf111 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf111]
#   %_frozen_param174 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param174]
#   %_frozen_param175 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param175]
#   %add_60 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_60]
#   %clone_24 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_22,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_103 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_24, torch.float32), kwargs = {})
#   %view_47 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_103, [8, 32, 16, 16384]), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_47, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_15 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_47, %getitem_35), kwargs = {})
#   %add_59 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_34, 1e-06), kwargs = {})
#   %rsqrt_15 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_59,), kwargs = {})
#   %mul_34 : Tensor "f32[8, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %rsqrt_15), kwargs = {})
#   %view_48 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_34, [8, 512, 128, 128]), kwargs = {})
#   %mul_35 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_48, %unsqueeze_91), kwargs = {})
#   %add_60 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_94), kwargs = {})
#   %neg_14 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_60,), kwargs = {})
#   %exp_14 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_14,), kwargs = {})
#   %add_61 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_14, 1), kwargs = {})
#   %div_23 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_60, %add_61), kwargs = {})
#   %convert_element_type_108 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_23, torch.float16), kwargs = {})
#   return %add_60,%convert_element_type_108
triton_poi_fused_clone_native_group_norm_silu_20 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 402657280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 8388608
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 262144.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = -tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.full([1], 1.0, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp16 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\75\c75f2nmtwvkgs5qdax7bvrrecvutq66obyk5qb7i7szhfwtjpado.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_8 => add_65
#   hidden_states_65 => add_64, convert_element_type_114, div_24, exp_15, neg_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_66, add_67, clone_27, convert_element_type_115, convert_element_type_116, convert_element_type_117, convert_element_type_120, iota_2, mul_38, mul_39, unsqueeze_101
#   output_tensor_7 => div_25
# Graph fragment:
#   %div_22 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_22]
#   %buf121 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf121]
#   %_frozen_param79 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param79]
#   %neg_15 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_63,), kwargs = {})
#   %exp_15 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_15,), kwargs = {})
#   %add_64 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_15, 1), kwargs = {})
#   %div_24 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_63, %add_64), kwargs = {})
#   %convert_element_type_114 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_24, torch.float16), kwargs = {})
#   %convolution_18 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_114, %arg78_1, %arg79_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_65 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_22, %convolution_18), kwargs = {})
#   %div_25 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_65, 1.0), kwargs = {})
#   %convert_element_type_115 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_25, torch.float32), kwargs = {})
#   %iota_2 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_38 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_66 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, 0), kwargs = {})
#   %convert_element_type_116 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_66, torch.float32), kwargs = {})
#   %add_67 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_116, 0.0), kwargs = {})
#   %mul_39 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, 0.5), kwargs = {})
#   %convert_element_type_117 : Tensor "i64[256][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_117, -1), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_115, [None, None, %unsqueeze_101, %convert_element_type_117]), kwargs = {})
#   %clone_27 : Tensor "f32[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_120 : Tensor "f16[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_27, torch.float16), kwargs = {})
#   return %convert_element_type_120
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_21 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1073742848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 131072) % 256)
    x1 = ((xindex // 512) % 256)
    x0 = (xindex % 512)
    x3 = xindex // 33554432
    x5 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 512*tmp8 + 65536*tmp4 + 8388608*x3), None).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x0 + 512*tmp8 + 65536*tmp4 + 8388608*x3), None).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full([1], 1.0, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp17, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\qo\cqoofmqpttllpboxptlr2i3c4qmaciz6uuvgucbc5o3p7uim3onm.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_8 => add_65
#   hidden_states_65 => add_64, convert_element_type_114, div_24, exp_15, neg_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_66, add_67, clone_27, convert_element_type_115, convert_element_type_116, convert_element_type_117, convert_element_type_120, iota_2, mul_38, mul_39, unsqueeze_101
#   hidden_states_69 => convolution_19
#   output_tensor_7 => div_25
# Graph fragment:
#   %buf123 : Tensor "f16[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf123]
#   %_frozen_param81 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=_frozen_param81]
#   %neg_15 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_63,), kwargs = {})
#   %exp_15 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_15,), kwargs = {})
#   %add_64 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_15, 1), kwargs = {})
#   %div_24 : Tensor "f32[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_63, %add_64), kwargs = {})
#   %convert_element_type_114 : Tensor "f16[8, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_24, torch.float16), kwargs = {})
#   %convolution_18 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_114, %arg78_1, %arg79_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_65 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_22, %convolution_18), kwargs = {})
#   %div_25 : Tensor "f16[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_65, 1.0), kwargs = {})
#   %convert_element_type_115 : Tensor "f32[8, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_25, torch.float32), kwargs = {})
#   %iota_2 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_38 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_66 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, 0), kwargs = {})
#   %convert_element_type_116 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_66, torch.float32), kwargs = {})
#   %add_67 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_116, 0.0), kwargs = {})
#   %mul_39 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, 0.5), kwargs = {})
#   %convert_element_type_117 : Tensor "i64[256][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_117, -1), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_115, [None, None, %unsqueeze_101, %convert_element_type_117]), kwargs = {})
#   %clone_27 : Tensor "f32[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_120 : Tensor "f16[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_27, torch.float16), kwargs = {})
#   %convolution_19 : Tensor "f16[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_120, %arg80_1, %arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_19
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_22 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1610613760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\t4\ct46yeb2o64ranhv7dchkmqj2gwj6r426wqdycqa3jgrcnoovtsp.py
# Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_70 => clone_28, convert_element_type_121, var_mean_17, view_51
# Graph fragment:
#   %convolution_19 : Tensor "f16[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=convolution_19]
#   %clone_28 : Tensor "f16[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_121 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_28, torch.float32), kwargs = {})
#   %view_51 : Tensor "f32[8, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_121, [8, 32, 16, 65536]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_51, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_39,%buf126
triton_red_fused_clone_native_group_norm_23 = async_compile.triton('triton_red_fused_clone_native_group_norm_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 1048576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 536870912}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_23(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1048576
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 512*r0_3 + 33554432*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\xq\cxqlz6nxb6exytnyg37htgorrlb433gudi62opkfl3taoio44ckh.py
# Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_70 => add_70, add_71, clone_28, convert_element_type_121, mul_42, mul_43, rsqrt_17, sub_17, var_mean_17, view_51, view_52
#   hidden_states_71 => add_72, convert_element_type_126, div_26, exp_16, neg_16
# Graph fragment:
#   %convolution_19 : Tensor "f16[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=convolution_19]
#   %getitem_39 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf126 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf126]
#   %_frozen_param178 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param178]
#   %_frozen_param179 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param179]
#   %add_71 : Tensor "f32[8, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=add_71]
#   %clone_28 : Tensor "f16[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_121 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_28, torch.float32), kwargs = {})
#   %view_51 : Tensor "f32[8, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_121, [8, 32, 16, 65536]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_51, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_17 : Tensor "f32[8, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_51, %getitem_39), kwargs = {})
#   %add_70 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_70,), kwargs = {})
#   %mul_42 : Tensor "f32[8, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %rsqrt_17), kwargs = {})
#   %view_52 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_42, [8, 512, 256, 256]), kwargs = {})
#   %mul_43 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_52, %unsqueeze_104), kwargs = {})
#   %add_71 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_107), kwargs = {})
#   %neg_16 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_71,), kwargs = {})
#   %exp_16 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_16,), kwargs = {})
#   %add_72 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_16, 1), kwargs = {})
#   %div_26 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_71, %add_72), kwargs = {})
#   %convert_element_type_126 : Tensor "f16[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_26, torch.float16), kwargs = {})
#   return %add_71,%convert_element_type_126
triton_poi_fused_clone_native_group_norm_silu_24 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1610616832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 33554432
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 16)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 1048576.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = -tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.full([1], 1.0, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp16 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\qe\cqe6pdg3sikyvamwkaxy57eu53sop4hkr2l7iz3zi3ckb3xfehbz.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => add_72, convert_element_type_126, div_26, exp_16, neg_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => clone_29, convert_element_type_127, var_mean_18, view_53
# Graph fragment:
#   %buf130 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=buf130]
#   %_frozen_param85 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param85]
#   %neg_16 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_71,), kwargs = {})
#   %exp_16 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_16,), kwargs = {})
#   %add_72 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_16, 1), kwargs = {})
#   %div_26 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_71, %add_72), kwargs = {})
#   %convert_element_type_126 : Tensor "f16[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_26, torch.float16), kwargs = {})
#   %convolution_20 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_126, %arg84_1, %arg85_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_29 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_20,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_127 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_29, torch.float32), kwargs = {})
#   %view_53 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_127, [8, 32, 8, 65536]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_41,%buf132
triton_red_fused_clone_convolution_native_group_norm_silu_25 = async_compile.triton('triton_red_fused_clone_convolution_native_group_norm_silu_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 524288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_convolution_native_group_norm_silu_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 268435968}}
)
@triton.jit
def triton_red_fused_clone_convolution_native_group_norm_silu_25(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 524288
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 256*r0_3 + 16777216*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 8*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\cc\cccy273qrefitb5navem6x2vowu2guyor44jevqjxebpj3qe2gak.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73, hidden_states_74], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => add_72, convert_element_type_126, div_26, exp_16, neg_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => add_73, add_74, clone_29, convert_element_type_127, mul_44, mul_45, rsqrt_18, sub_18, var_mean_18, view_53, view_54
#   hidden_states_74 => add_75, convert_element_type_132, div_27, exp_17, neg_17
# Graph fragment:
#   %buf130 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=buf130]
#   %_frozen_param85 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param85]
#   %getitem_41 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf132 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf132]
#   %_frozen_param180 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param180]
#   %_frozen_param181 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param181]
#   %add_74 : Tensor "f32[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=add_74]
#   %neg_16 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_71,), kwargs = {})
#   %exp_16 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_16,), kwargs = {})
#   %add_72 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_16, 1), kwargs = {})
#   %div_26 : Tensor "f32[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_71, %add_72), kwargs = {})
#   %convert_element_type_126 : Tensor "f16[8, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_26, torch.float16), kwargs = {})
#   %convolution_20 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_126, %arg84_1, %arg85_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_29 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_20,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_127 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_29, torch.float32), kwargs = {})
#   %view_53 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_127, [8, 32, 8, 65536]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_53, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_18 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_53, %getitem_41), kwargs = {})
#   %add_73 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-06), kwargs = {})
#   %rsqrt_18 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_73,), kwargs = {})
#   %mul_44 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_18), kwargs = {})
#   %view_54 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_44, [8, 256, 256, 256]), kwargs = {})
#   %mul_45 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_54, %unsqueeze_110), kwargs = {})
#   %add_74 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %unsqueeze_113), kwargs = {})
#   %neg_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_74,), kwargs = {})
#   %exp_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_17,), kwargs = {})
#   %add_75 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_17, 1), kwargs = {})
#   %div_27 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_74, %add_75), kwargs = {})
#   %convert_element_type_132 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_27, torch.float16), kwargs = {})
#   return %add_74,%convert_element_type_132
triton_poi_fused_clone_convolution_native_group_norm_silu_26 = async_compile.triton('triton_poi_fused_clone_convolution_native_group_norm_silu_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_native_group_norm_silu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 805309952}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_native_group_norm_silu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 16777216
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = tl.full([1], 524288.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = -tmp18
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.full([1], 1.0, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = (tmp18 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\qc\cqcxraftiztvjub6rphmfvk4drhmmqqigx6xq7q5h34un75t6r6p.py
# Topologically Sorted Source Nodes: [input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_9 => add_76
#   hidden_states_74 => add_75, convert_element_type_132, div_27, exp_17, neg_17
#   hidden_states_76 => convolution_21
#   hidden_states_77 => clone_31, convert_element_type_133, var_mean_19, view_55
#   input_tensor => convolution_22
#   output_tensor_8 => div_28
# Graph fragment:
#   %buf134 : Tensor "f16[524288, 256][256, 1]cuda:0" = PlaceHolder[target=buf134]
#   %buf137 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=buf137]
#   %_frozen_param89 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param89]
#   %convolution_22 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_19, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_74,), kwargs = {})
#   %exp_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_17,), kwargs = {})
#   %add_75 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_17, 1), kwargs = {})
#   %div_27 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_74, %add_75), kwargs = {})
#   %convert_element_type_132 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_27, torch.float16), kwargs = {})
#   %convolution_21 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_132, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_76 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %div_28 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_76, 1.0), kwargs = {})
#   %clone_31 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_28,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_133 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_31, torch.float32), kwargs = {})
#   %view_55 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_133, [8, 32, 8, 65536]), kwargs = {})
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_55, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_43,%buf139
triton_red_fused_add_clone_convolution_div_native_group_norm_silu_27 = async_compile.triton('triton_red_fused_add_clone_convolution_div_native_group_norm_silu_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 524288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_native_group_norm_silu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 536871424}}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_native_group_norm_silu_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 524288
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 256*r0_3 + 16777216*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 8*x0 + 256*r0_3 + 16777216*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 8*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
    tl.store(out_ptr1 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\iq\ciqplqnw6xlmeqlrjot5pigefd3wn24bp74sztbexnurtc5xlz26.py
# Topologically Sorted Source Nodes: [input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77, hidden_states_78], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_9 => add_76
#   hidden_states_74 => add_75, convert_element_type_132, div_27, exp_17, neg_17
#   hidden_states_76 => convolution_21
#   hidden_states_77 => add_77, add_78, clone_31, convert_element_type_133, mul_46, mul_47, rsqrt_19, sub_19, var_mean_19, view_55, view_56
#   hidden_states_78 => add_79, convert_element_type_138, div_29, exp_18, neg_18
#   input_tensor => convolution_22
#   output_tensor_8 => div_28
# Graph fragment:
#   %buf134 : Tensor "f16[524288, 256][256, 1]cuda:0" = PlaceHolder[target=buf134]
#   %buf137 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=buf137]
#   %_frozen_param89 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param89]
#   %getitem_43 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_43]
#   %buf139 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf139]
#   %_frozen_param182 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param182]
#   %_frozen_param183 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param183]
#   %add_78 : Tensor "f32[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=add_78]
#   %convolution_22 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_19, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_74,), kwargs = {})
#   %exp_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_17,), kwargs = {})
#   %add_75 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_17, 1), kwargs = {})
#   %div_27 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_74, %add_75), kwargs = {})
#   %convert_element_type_132 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_27, torch.float16), kwargs = {})
#   %convolution_21 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_132, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_76 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %div_28 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_76, 1.0), kwargs = {})
#   %clone_31 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_28,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_133 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_31, torch.float32), kwargs = {})
#   %view_55 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_133, [8, 32, 8, 65536]), kwargs = {})
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_55, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_19 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_55, %getitem_43), kwargs = {})
#   %add_77 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-06), kwargs = {})
#   %rsqrt_19 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_77,), kwargs = {})
#   %mul_46 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_19), kwargs = {})
#   %view_56 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_46, [8, 256, 256, 256]), kwargs = {})
#   %mul_47 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_56, %unsqueeze_116), kwargs = {})
#   %add_78 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_119), kwargs = {})
#   %neg_18 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_78,), kwargs = {})
#   %exp_18 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_18,), kwargs = {})
#   %add_79 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_18, 1), kwargs = {})
#   %div_29 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_78, %add_79), kwargs = {})
#   %convert_element_type_138 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_29, torch.float16), kwargs = {})
#   return %add_78,%convert_element_type_138
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_28 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1073745408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 16777216
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = tl.full([1], 524288.0, tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tl.full([1], 1e-06, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = -tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tmp24 + tmp5
    tmp26 = (tmp22 / tmp25)
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\hb\chb7enm2z4aq6tosh7fncxntgu547ku3zbeg5rvrnly25rhxqwbh.py
# Topologically Sorted Source Nodes: [input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_81, hidden_states_83, add_10, output_tensor_9], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_10 => add_83
#   add_9 => add_76
#   hidden_states_74 => add_75, convert_element_type_132, div_27, exp_17, neg_17
#   hidden_states_76 => convolution_21
#   hidden_states_81 => add_82, convert_element_type_144, div_30, exp_19, neg_19
#   hidden_states_83 => convolution_24
#   input_tensor => convolution_22
#   output_tensor_8 => div_28
#   output_tensor_9 => div_31
# Graph fragment:
#   %buf134 : Tensor "f16[524288, 256][256, 1]cuda:0" = PlaceHolder[target=buf134]
#   %buf137 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=buf137]
#   %_frozen_param89 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param89]
#   %buf149 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=buf149]
#   %_frozen_param99 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param99]
#   %convolution_22 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_19, %arg90_1, %arg91_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_74,), kwargs = {})
#   %exp_17 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_17,), kwargs = {})
#   %add_75 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_17, 1), kwargs = {})
#   %div_27 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_74, %add_75), kwargs = {})
#   %convert_element_type_132 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_27, torch.float16), kwargs = {})
#   %convolution_21 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_132, %arg88_1, %arg89_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_76 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %div_28 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_76, 1.0), kwargs = {})
#   %neg_19 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_81,), kwargs = {})
#   %exp_19 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_19,), kwargs = {})
#   %add_82 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_19, 1), kwargs = {})
#   %div_30 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_81, %add_82), kwargs = {})
#   %convert_element_type_144 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_30, torch.float16), kwargs = {})
#   %convolution_24 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_144, %arg98_1, %arg99_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_83 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_28, %convolution_24), kwargs = {})
#   %div_31 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_83, 1.0), kwargs = {})
#   return %div_31
triton_poi_fused_add_convolution_div_silu_29 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1342178304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\fj\cfjzjctwavkw4y3cjuf2vbijxgyhozk5brcuhji7ombvdcf5bnne.py
# Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_84 => clone_34, convert_element_type_145, var_mean_21, view_59
# Graph fragment:
#   %div_31 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=div_31]
#   %clone_34 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_31,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_145 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_34, torch.float32), kwargs = {})
#   %view_59 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_145, [8, 32, 8, 65536]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_59, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_47,%buf152
triton_red_fused_clone_native_group_norm_30 = async_compile.triton('triton_red_fused_clone_native_group_norm_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 524288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 268435456}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_30(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 524288
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 256*r0_3 + 16777216*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ql\cql6nugh54ebsuhpfk3gfaqtj5kzi2qamds777v7jir36ptux53v.py
# Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_84 => add_84, add_85, clone_34, convert_element_type_145, mul_50, mul_51, rsqrt_21, sub_21, var_mean_21, view_59, view_60
#   hidden_states_85 => add_86, convert_element_type_150, div_32, exp_20, neg_20
# Graph fragment:
#   %div_31 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=div_31]
#   %getitem_47 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_47]
#   %buf152 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf152]
#   %_frozen_param186 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param186]
#   %_frozen_param187 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param187]
#   %add_85 : Tensor "f32[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=add_85]
#   %clone_34 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_31,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_145 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_34, torch.float32), kwargs = {})
#   %view_59 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_145, [8, 32, 8, 65536]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_59, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_21 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_59, %getitem_47), kwargs = {})
#   %add_84 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-06), kwargs = {})
#   %rsqrt_21 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_84,), kwargs = {})
#   %mul_50 : Tensor "f32[8, 32, 8, 65536][16777216, 524288, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %rsqrt_21), kwargs = {})
#   %view_60 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_50, [8, 256, 256, 256]), kwargs = {})
#   %mul_51 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_60, %unsqueeze_128), kwargs = {})
#   %add_85 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %unsqueeze_131), kwargs = {})
#   %neg_20 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_85,), kwargs = {})
#   %exp_20 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_20,), kwargs = {})
#   %add_86 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_20, 1), kwargs = {})
#   %div_32 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_85, %add_86), kwargs = {})
#   %convert_element_type_150 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_32, torch.float16), kwargs = {})
#   return %add_85,%convert_element_type_150
triton_poi_fused_clone_native_group_norm_silu_31 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 805309440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 16777216
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 524288.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = -tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.full([1], 1.0, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp16 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\2d\c2d4m7k55cgght6xeige575uydy6xdfgu3cb5fzoxoizu5ngiaac.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_11 => add_90
#   hidden_states_88 => add_89, convert_element_type_156, div_33, exp_21, neg_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, add_91, add_92, clone_37, convert_element_type_157, convert_element_type_158, convert_element_type_159, convert_element_type_162, iota_4, mul_54, mul_55, unsqueeze_138
#   output_tensor_10 => div_34
# Graph fragment:
#   %div_31 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=div_31]
#   %buf162 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0" = PlaceHolder[target=buf162]
#   %_frozen_param107 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param107]
#   %neg_21 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_88,), kwargs = {})
#   %exp_21 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_21,), kwargs = {})
#   %add_89 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_21, 1), kwargs = {})
#   %div_33 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_88, %add_89), kwargs = {})
#   %convert_element_type_156 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_33, torch.float16), kwargs = {})
#   %convolution_26 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_156, %arg106_1, %arg107_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_90 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_31, %convolution_26), kwargs = {})
#   %div_34 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_90, 1.0), kwargs = {})
#   %convert_element_type_157 : Tensor "f32[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_34, torch.float32), kwargs = {})
#   %iota_4 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_91 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_158 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_91, torch.float32), kwargs = {})
#   %add_92 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_158, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, 0.5), kwargs = {})
#   %convert_element_type_159 : Tensor "i64[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_138 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_159, -1), kwargs = {})
#   %_unsafe_index_2 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_157, [None, None, %unsqueeze_138, %convert_element_type_159]), kwargs = {})
#   %clone_37 : Tensor "f32[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_2,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_162 : Tensor "f16[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_37, torch.float16), kwargs = {})
#   return %convert_element_type_162
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_32 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2147484160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 131072) % 512)
    x1 = ((xindex // 256) % 512)
    x0 = (xindex % 256)
    x3 = xindex // 67108864
    x5 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 256*tmp8 + 65536*tmp4 + 16777216*x3), None).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x0 + 256*tmp8 + 65536*tmp4 + 16777216*x3), None).to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full([1], 1.0, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp17, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\r4\cr4t3jpctot3dfet7jzxj5ubwnjhumvboisqplqaiqfmh7mxjqcs.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_11 => add_90
#   hidden_states_88 => add_89, convert_element_type_156, div_33, exp_21, neg_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, add_91, add_92, clone_37, convert_element_type_157, convert_element_type_158, convert_element_type_159, convert_element_type_162, iota_4, mul_54, mul_55, unsqueeze_138
#   hidden_states_92 => convolution_27
#   output_tensor_10 => div_34
# Graph fragment:
#   %buf164 : Tensor "f16[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf164]
#   %_frozen_param109 : Tensor "f16[256][1]cuda:0" = PlaceHolder[target=_frozen_param109]
#   %neg_21 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_88,), kwargs = {})
#   %exp_21 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_21,), kwargs = {})
#   %add_89 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_21, 1), kwargs = {})
#   %div_33 : Tensor "f32[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_88, %add_89), kwargs = {})
#   %convert_element_type_156 : Tensor "f16[8, 256, 256, 256][16777216, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_33, torch.float16), kwargs = {})
#   %convolution_26 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_156, %arg106_1, %arg107_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_90 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_31, %convolution_26), kwargs = {})
#   %div_34 : Tensor "f16[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_90, 1.0), kwargs = {})
#   %convert_element_type_157 : Tensor "f32[8, 256, 256, 256][16777216, 1, 65536, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_34, torch.float32), kwargs = {})
#   %iota_4 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_91 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_158 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_91, torch.float32), kwargs = {})
#   %add_92 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_158, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, 0.5), kwargs = {})
#   %convert_element_type_159 : Tensor "i64[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_138 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_159, -1), kwargs = {})
#   %_unsafe_index_2 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_157, [None, None, %unsqueeze_138, %convert_element_type_159]), kwargs = {})
#   %clone_37 : Tensor "f32[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_2,), kwargs = {memory_format: torch.channels_last})
#   %convert_element_type_162 : Tensor "f16[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_37, torch.float16), kwargs = {})
#   %convolution_27 : Tensor "f16[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_162, %arg108_1, %arg109_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_27
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3221225984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\xq\cxquja5myfyllmf2eshpjnkdqucy375dw4whsetr4vgsg2ebksou.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => clone_38, convert_element_type_163, var_mean_23, view_63
# Graph fragment:
#   %convolution_27 : Tensor "f16[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=convolution_27]
#   %clone_38 : Tensor "f16[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_27,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_163 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_38, torch.float32), kwargs = {})
#   %view_63 : Tensor "f32[8, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_163, [8, 32, 8, 262144]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_63, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_51,%buf167
triton_red_fused_clone_native_group_norm_34 = async_compile.triton('triton_red_fused_clone_native_group_norm_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2097152},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 1073741824}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_34(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 2097152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 256*r0_3 + 67108864*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\a5\ca53mtsqraw225lduollehtdpam2hwtpfzyzxg5waixq7yjquz2l.py
# Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_93 => add_95, add_96, clone_38, convert_element_type_163, mul_58, mul_59, rsqrt_23, sub_23, var_mean_23, view_63, view_64
#   hidden_states_94 => add_97, convert_element_type_168, div_35, exp_22, neg_22
# Graph fragment:
#   %convolution_27 : Tensor "f16[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=convolution_27]
#   %getitem_51 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_51]
#   %buf167 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf167]
#   %_frozen_param190 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param190]
#   %_frozen_param191 : Tensor "f16[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param191]
#   %add_96 : Tensor "f32[8, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=add_96]
#   %clone_38 : Tensor "f16[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_27,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_163 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_38, torch.float32), kwargs = {})
#   %view_63 : Tensor "f32[8, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_163, [8, 32, 8, 262144]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_63, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_23 : Tensor "f32[8, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_63, %getitem_51), kwargs = {})
#   %add_95 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-06), kwargs = {})
#   %rsqrt_23 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_95,), kwargs = {})
#   %mul_58 : Tensor "f32[8, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_23), kwargs = {})
#   %view_64 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_58, [8, 256, 512, 512]), kwargs = {})
#   %mul_59 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_64, %unsqueeze_141), kwargs = {})
#   %add_96 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_144), kwargs = {})
#   %neg_22 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_96,), kwargs = {})
#   %exp_22 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_22,), kwargs = {})
#   %add_97 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_22, 1), kwargs = {})
#   %div_35 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_96, %add_97), kwargs = {})
#   %convert_element_type_168 : Tensor "f16[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_35, torch.float16), kwargs = {})
#   return %add_96,%convert_element_type_168
triton_poi_fused_clone_native_group_norm_silu_35 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3221228544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 67108864
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 2097152.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = -tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.full([1], 1.0, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp16 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\oo\coox6nmahjzqqpmfqsjpn4mi3tmy2lf2hcl55xjj2awqelqrpxk7.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_94 => add_97, convert_element_type_168, div_35, exp_22, neg_22
#   hidden_states_95 => convolution_28
#   hidden_states_96 => clone_39, convert_element_type_169, var_mean_24, view_65
# Graph fragment:
#   %buf171 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=buf171]
#   %_frozen_param113 : Tensor "f16[128][1]cuda:0" = PlaceHolder[target=_frozen_param113]
#   %neg_22 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_96,), kwargs = {})
#   %exp_22 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_22,), kwargs = {})
#   %add_97 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_22, 1), kwargs = {})
#   %div_35 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_96, %add_97), kwargs = {})
#   %convert_element_type_168 : Tensor "f16[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_35, torch.float16), kwargs = {})
#   %convolution_28 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_168, %arg112_1, %arg113_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_39 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_28,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_169 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_39, torch.float32), kwargs = {})
#   %view_65 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_169, [8, 32, 4, 262144]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_65, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_53,%buf173
triton_red_fused_clone_convolution_native_group_norm_silu_36 = async_compile.triton('triton_red_fused_clone_convolution_native_group_norm_silu_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 1048576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_convolution_native_group_norm_silu_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 536871168}}
)
@triton.jit
def triton_red_fused_clone_convolution_native_group_norm_silu_36(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1048576
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 33554432*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\o3\co3iimjqhdso73a3flgpebgxlhzmcq5ugwtmc54t572dxqkpaghn.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96, hidden_states_97], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_94 => add_97, convert_element_type_168, div_35, exp_22, neg_22
#   hidden_states_95 => convolution_28
#   hidden_states_96 => add_98, add_99, clone_39, convert_element_type_169, mul_60, mul_61, rsqrt_24, sub_24, var_mean_24, view_65, view_66
#   hidden_states_97 => add_100, convert_element_type_174, div_36, exp_23, neg_23
# Graph fragment:
#   %buf171 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=buf171]
#   %_frozen_param113 : Tensor "f16[128][1]cuda:0" = PlaceHolder[target=_frozen_param113]
#   %getitem_53 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_53]
#   %buf173 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf173]
#   %_frozen_param192 : Tensor "f16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param192]
#   %_frozen_param193 : Tensor "f16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param193]
#   %add_99 : Tensor "f32[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=add_99]
#   %neg_22 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_96,), kwargs = {})
#   %exp_22 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_22,), kwargs = {})
#   %add_97 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_22, 1), kwargs = {})
#   %div_35 : Tensor "f32[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_96, %add_97), kwargs = {})
#   %convert_element_type_168 : Tensor "f16[8, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_35, torch.float16), kwargs = {})
#   %convolution_28 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_168, %arg112_1, %arg113_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_39 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_28,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_169 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_39, torch.float32), kwargs = {})
#   %view_65 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_169, [8, 32, 4, 262144]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_65, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_65, %getitem_53), kwargs = {})
#   %add_98 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-06), kwargs = {})
#   %rsqrt_24 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_98,), kwargs = {})
#   %mul_60 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %view_66 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_60, [8, 128, 512, 512]), kwargs = {})
#   %mul_61 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_66, %unsqueeze_147), kwargs = {})
#   %add_99 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %unsqueeze_150), kwargs = {})
#   %neg_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %exp_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_23,), kwargs = {})
#   %add_100 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_23, 1), kwargs = {})
#   %div_36 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_99, %add_100), kwargs = {})
#   %convert_element_type_174 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_36, torch.float16), kwargs = {})
#   return %add_99,%convert_element_type_174
triton_poi_fused_clone_convolution_native_group_norm_silu_37 = async_compile.triton('triton_poi_fused_clone_convolution_native_group_norm_silu_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_native_group_norm_silu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1610615552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_native_group_norm_silu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 33554432
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = tl.full([1], 1048576.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = -tmp18
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.full([1], 1.0, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = (tmp18 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ip\cipk27tsmjom5tphqeddd7zxgw2w3stvwp5huymdebjb6jkherk2.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_12 => add_101
#   hidden_states_100 => clone_41, convert_element_type_175, var_mean_25, view_67
#   hidden_states_97 => add_100, convert_element_type_174, div_36, exp_23, neg_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_37
# Graph fragment:
#   %buf175 : Tensor "f16[2097152, 128][128, 1]cuda:0" = PlaceHolder[target=buf175]
#   %buf178 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=buf178]
#   %_frozen_param117 : Tensor "f16[128][1]cuda:0" = PlaceHolder[target=_frozen_param117]
#   %convolution_30 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg118_1, %arg119_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %exp_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_23,), kwargs = {})
#   %add_100 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_23, 1), kwargs = {})
#   %div_36 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_99, %add_100), kwargs = {})
#   %convert_element_type_174 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_36, torch.float16), kwargs = {})
#   %convolution_29 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_174, %arg116_1, %arg117_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_101 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_37 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_101, 1.0), kwargs = {})
#   %clone_41 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_37,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_175 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_41, torch.float32), kwargs = {})
#   %view_67 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_175, [8, 32, 4, 262144]), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_67, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_55,%buf180
triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38 = async_compile.triton('triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 1048576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 1073742080}}
)
@triton.jit
def triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1048576
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 33554432*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 4*x0 + 128*r0_3 + 33554432*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 4*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
    tl.store(out_ptr1 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ug\cug6bebd3fjm2y2h4ksea5urzfwgxi56hwrkldx4qkhzvlkkf6ia.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100, hidden_states_101], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_12 => add_101
#   hidden_states_100 => add_102, add_103, clone_41, convert_element_type_175, mul_62, mul_63, rsqrt_25, sub_25, var_mean_25, view_67, view_68
#   hidden_states_101 => add_104, convert_element_type_180, div_38, exp_24, neg_24
#   hidden_states_97 => add_100, convert_element_type_174, div_36, exp_23, neg_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_37
# Graph fragment:
#   %buf175 : Tensor "f16[2097152, 128][128, 1]cuda:0" = PlaceHolder[target=buf175]
#   %buf178 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=buf178]
#   %_frozen_param117 : Tensor "f16[128][1]cuda:0" = PlaceHolder[target=_frozen_param117]
#   %getitem_55 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_55]
#   %buf180 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf180]
#   %_frozen_param194 : Tensor "f16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param194]
#   %_frozen_param195 : Tensor "f16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param195]
#   %add_103 : Tensor "f32[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=add_103]
#   %convolution_30 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg118_1, %arg119_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %exp_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_23,), kwargs = {})
#   %add_100 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_23, 1), kwargs = {})
#   %div_36 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_99, %add_100), kwargs = {})
#   %convert_element_type_174 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_36, torch.float16), kwargs = {})
#   %convolution_29 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_174, %arg116_1, %arg117_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_101 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_37 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_101, 1.0), kwargs = {})
#   %clone_41 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_37,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_175 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_41, torch.float32), kwargs = {})
#   %view_67 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_175, [8, 32, 4, 262144]), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_67, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_67, %getitem_55), kwargs = {})
#   %add_102 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-06), kwargs = {})
#   %rsqrt_25 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_102,), kwargs = {})
#   %mul_62 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %view_68 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_62, [8, 128, 512, 512]), kwargs = {})
#   %mul_63 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_68, %unsqueeze_153), kwargs = {})
#   %add_103 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %unsqueeze_156), kwargs = {})
#   %neg_24 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_103,), kwargs = {})
#   %exp_24 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_24,), kwargs = {})
#   %add_104 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_24, 1), kwargs = {})
#   %div_38 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_103, %add_104), kwargs = {})
#   %convert_element_type_180 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_38, torch.float16), kwargs = {})
#   return %add_103,%convert_element_type_180
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2147486464}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 33554432
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = tl.full([1], 1048576.0, tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tl.full([1], 1e-06, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = -tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tmp24 + tmp5
    tmp26 = (tmp22 / tmp25)
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\lv\clvfqrgx4642zmwcs55tai4fzmjvsq2jt4qscaaunnfq66rwsoku.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_104, hidden_states_106, add_13, output_tensor_12], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_12 => add_101
#   add_13 => add_108
#   hidden_states_104 => add_107, convert_element_type_186, div_39, exp_25, neg_25
#   hidden_states_106 => convolution_32
#   hidden_states_97 => add_100, convert_element_type_174, div_36, exp_23, neg_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_37
#   output_tensor_12 => div_40
# Graph fragment:
#   %buf175 : Tensor "f16[2097152, 128][128, 1]cuda:0" = PlaceHolder[target=buf175]
#   %buf178 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=buf178]
#   %_frozen_param117 : Tensor "f16[128][1]cuda:0" = PlaceHolder[target=_frozen_param117]
#   %buf190 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=buf190]
#   %_frozen_param127 : Tensor "f16[128][1]cuda:0" = PlaceHolder[target=_frozen_param127]
#   %convolution_30 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg118_1, %arg119_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_99,), kwargs = {})
#   %exp_23 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_23,), kwargs = {})
#   %add_100 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_23, 1), kwargs = {})
#   %div_36 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_99, %add_100), kwargs = {})
#   %convert_element_type_174 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_36, torch.float16), kwargs = {})
#   %convolution_29 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_174, %arg116_1, %arg117_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_101 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_37 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_101, 1.0), kwargs = {})
#   %neg_25 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_106,), kwargs = {})
#   %exp_25 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_25,), kwargs = {})
#   %add_107 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_25, 1), kwargs = {})
#   %div_39 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_106, %add_107), kwargs = {})
#   %convert_element_type_186 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_39, torch.float16), kwargs = {})
#   %convolution_32 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_186, %arg126_1, %arg127_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_108 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_37, %convolution_32), kwargs = {})
#   %div_40 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_108, 1.0), kwargs = {})
#   return %div_40
triton_poi_fused_add_convolution_div_silu_40 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2684355072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\bf\cbfr6g2t55qzgcfb5n2oe3wafcoen2it2gjqnplfwshmmilyn7h6.py
# Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_107 => clone_44, convert_element_type_187, var_mean_27, view_71
# Graph fragment:
#   %div_40 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=div_40]
#   %clone_44 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_40,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_187 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_44, torch.float32), kwargs = {})
#   %view_71 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_187, [8, 32, 4, 262144]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_71, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_59,%buf193
triton_red_fused_clone_native_group_norm_41 = async_compile.triton('triton_red_fused_clone_native_group_norm_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 1048576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096, 'r0_': 536870912}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_41(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1048576
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 4)
        r0_3 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2 + 4*x0 + 128*r0_3 + 33554432*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\ln\clnjd2banfg5v56ndfi2f3gtddncttewj3rtobwoearazpffw52t.py
# Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_107 => add_109, add_110, clone_44, convert_element_type_187, mul_66, mul_67, rsqrt_27, sub_27, var_mean_27, view_71, view_72
#   hidden_states_108 => add_111, convert_element_type_192, div_41, exp_26, neg_26
# Graph fragment:
#   %div_40 : Tensor "f16[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=div_40]
#   %getitem_59 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_59]
#   %buf193 : Tensor "f32[8, 32, 1, 1][32, 1, 256, 256]cuda:0" = PlaceHolder[target=buf193]
#   %_frozen_param198 : Tensor "f16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param198]
#   %_frozen_param199 : Tensor "f16[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=_frozen_param199]
#   %add_110 : Tensor "f32[8, 128, 512, 512][33554432, 1, 65536, 128]cuda:0" = PlaceHolder[target=add_110]
#   %clone_44 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_40,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_187 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_44, torch.float32), kwargs = {})
#   %view_71 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_187, [8, 32, 4, 262144]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_71, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_71, %getitem_59), kwargs = {})
#   %add_109 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_27 : Tensor "f32[8, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_109,), kwargs = {})
#   %mul_66 : Tensor "f32[8, 32, 4, 262144][33554432, 1048576, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %view_72 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_66, [8, 128, 512, 512]), kwargs = {})
#   %mul_67 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_72, %unsqueeze_165), kwargs = {})
#   %add_110 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, %unsqueeze_168), kwargs = {})
#   %neg_26 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_110,), kwargs = {})
#   %exp_26 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_26,), kwargs = {})
#   %add_111 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_26, 1), kwargs = {})
#   %div_41 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_110, %add_111), kwargs = {})
#   %convert_element_type_192 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_41, torch.float16), kwargs = {})
#   return %add_110,%convert_element_type_192
triton_poi_fused_clone_native_group_norm_silu_42 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1610615296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 33554432
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 1048576.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = -tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.full([1], 1.0, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp16 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: C:\Users\Administrator\AppData\Local\Temp\torchinductor_administrator\rk\crkqrzyjttqbk6vbkx4jqkzaslet4k42cqcrotlondxz33zlndxl.py
# Topologically Sorted Source Nodes: [sample_2, sample_3, clamp], Original ATen: [aten.silu, aten.convolution, aten.clamp]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min, convert_element_type_205, convert_element_type_206
#   sample_2 => add_118, convert_element_type_204, div_44, exp_28, neg_28
#   sample_3 => convolution_default_1
# Graph fragment:
#   %buf209 : Tensor "f16[8, 3, 512, 512][786432, 1, 1536, 3]cuda:0" = PlaceHolder[target=buf209]
#   %neg_28 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_117,), kwargs = {})
#   %exp_28 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_28,), kwargs = {})
#   %add_118 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_28, 1), kwargs = {})
#   %div_44 : Tensor "f32[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_117, %add_118), kwargs = {})
#   %convert_element_type_204 : Tensor "f16[8, 128, 512, 512][33554432, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_44, torch.float16), kwargs = {})
#   %convolution_default_1 : Tensor "f16[8, 3, 512, 512][786432, 1, 1536, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_204, %div_tensor, %div_tensor_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type_205 : Tensor "f32[8, 3, 512, 512][786432, 1, 1536, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_default_1, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[8, 3, 512, 512][786432, 1, 1536, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_205, 0.0), kwargs = {})
#   %clamp_max : Tensor "f32[8, 3, 512, 512][786432, 1, 1536, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %convert_element_type_206 : Tensor "f16[8, 3, 512, 512][786432, 1, 1536, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float16), kwargs = {})
#   %inductor_force_stride_order_default : Tensor "f16[8, 3, 512, 512][786432, 1, 1536, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_force_stride_order.default](args = (%convert_element_type_206, (786432, 1, 1536, 3)), kwargs = {})
#   return %inductor_force_stride_order_default
triton_poi_fused_clamp_convolution_silu_43 = async_compile.triton('triton_poi_fused_clamp_convolution_silu_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=30, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_convolution_silu_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3976EC8C2B69FACD4D0F1FAEDC3E9A15AF0909F3EEA23EFAAD91786DC96FE283', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_convolution_silu_43(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 3)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = x0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.full([1], 0.489013671875, tl.float32)
    tmp7 = tl.full([1], 0.480224609375, tl.float32)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full([1], 0.5068359375, tl.float32)
    tmp10 = tl.where(tmp3, tmp9, tmp8)
    tmp11 = tmp0 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.full([1], 0.0, tl.float32)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tl.full([1], 1.0, tl.float32)
    tmp16 = triton_helpers.minimum(tmp14, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')

def partition_0(args):
    arg140_1, = args
    args.clear()
    assert_size_stride(arg140_1, (8, 4, 64, 64), (16384, 1, 256, 4))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 4, 64, 64), (16384, 1, 256, 4), torch.float16)
        # Topologically Sorted Source Nodes: [z, z_1], Original ATen: [aten.div, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_div_0:1
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_div_0.run(arg140_1, buf0, 131072, stream=stream0)
        del arg140_1
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float16)
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] triton_poi_fused_1:2
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf1, 4, stream=stream0)
        buf2 = empty_strided_cuda((32768, 4), (4, 1), torch.float16)
        # Topologically Sorted Source Nodes: [z, z_1], Original ATen: [aten.div, aten.convolution]
        # [Provenance debug handles] extern_kernels.bias_addmm:3
        extern_kernels.bias_addmm(reinterpret_tensor(buf1, (32768, 4), (0, 1), 0), reinterpret_tensor(buf0, (32768, 4), (4, 1), 0), reinterpret_tensor(_frozen_param0, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf2)
        del buf0
        del buf1
        # Topologically Sorted Source Nodes: [z, z_1, sample], Original ATen: [aten.div, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:4
        buf3 = extern_kernels.convolution(reinterpret_tensor(buf2, (8, 4, 64, 64), (16384, 1, 256, 4), 0), _frozen_param2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf2
        buf4 = empty_strided_cuda((8, 32, 1, 1), (32, 1, 256, 256), torch.float32)
        buf5 = empty_strided_cuda((8, 32, 1, 1), (32, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states], Original ATen: [aten.div, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_div_native_group_norm_2:5
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_div_native_group_norm_2.run(buf3, _frozen_param3, buf4, buf5, 256, 65536, stream=stream0)
        buf8 = empty_strided_cuda((8, 512, 64, 64), (2097152, 1, 32768, 512), torch.float16)
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states, hidden_states_1], Original ATen: [aten.div, aten.convolution, aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_div_native_group_norm_silu_3:6
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_div_native_group_norm_silu_3.run(buf3, _frozen_param3, buf4, buf5, _frozen_param140, _frozen_param141, buf8, 16777216, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:7
        buf9 = extern_kernels.convolution(buf8, _frozen_param6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf10 = buf5; del buf5  # reuse
        buf11 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_div_native_group_norm_2:8
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_div_native_group_norm_2.run(buf9, _frozen_param7, buf10, buf11, 256, 65536, stream=stream0)
        buf14 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3, hidden_states_4], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_div_native_group_norm_silu_3:9
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_div_native_group_norm_silu_3.run(buf9, _frozen_param7, buf10, buf11, _frozen_param142, _frozen_param143, buf14, 16777216, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:10
        buf15 = extern_kernels.convolution(buf14, _frozen_param10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf16 = buf11; del buf11  # reuse
        buf17 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_clone_convolution_div_native_group_norm_silu_view_4:11
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_native_group_norm_silu_view_4.run(buf3, _frozen_param3, buf15, _frozen_param11, buf16, buf17, 256, 65536, stream=stream0)
        buf19 = reinterpret_tensor(buf14, (8, 512, 4096), (2097152, 4096, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_view_5:12
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_view_5.run(buf3, _frozen_param3, buf15, _frozen_param11, buf16, buf17, _frozen_param144, _frozen_param145, buf19, 32768, 512, stream=stream0)
        buf20 = reinterpret_tensor(buf9, (8, 4096, 512), (2097152, 512, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.clone, aten.native_group_norm, aten.transpose, aten.bmm]
        # [Provenance debug handles] extern_kernels.bmm:13
        extern_kernels.bmm(reinterpret_tensor(buf19, (8, 4096, 512), (2097152, 1, 4096), 0), _frozen_param146, out=buf20)
        buf21 = empty_strided_cuda((8, 4096, 512), (2097152, 512, 1), torch.float16)
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, key], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.clone, aten.native_group_norm, aten.transpose, aten.bmm]
        # [Provenance debug handles] extern_kernels.bmm:14
        extern_kernels.bmm(reinterpret_tensor(buf19, (8, 4096, 512), (2097152, 1, 4096), 0), _frozen_param147, out=buf21)
        buf22 = empty_strided_cuda((8, 4096, 512), (2097152, 512, 1), torch.float16)
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, value], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.clone, aten.native_group_norm, aten.transpose, aten.bmm]
        # [Provenance debug handles] extern_kernels.bmm:15
        extern_kernels.bmm(reinterpret_tensor(buf19, (8, 4096, 512), (2097152, 1, 4096), 0), _frozen_param148, out=buf22)
        del buf19
        buf23 = reinterpret_tensor(buf20, (8, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6:16
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6.run(buf23, _frozen_param15, 16777216, stream=stream0)
        buf24 = reinterpret_tensor(buf21, (8, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6:17
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6.run(buf24, _frozen_param17, 16777216, stream=stream0)
        buf25 = reinterpret_tensor(buf22, (8, 1, 4096, 512), (2097152, 2097152, 512, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6:18
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_transpose_view_6.run(buf25, _frozen_param19, 16777216, stream=stream0)
        # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:19
        buf26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf23, buf24, buf25, None, False)
        del buf23
        del buf24
        buf27 = buf26[0]
        assert_size_stride(buf27, (8, 1, 4096, 512), (2097152, 512, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf27, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf26
        buf31 = reinterpret_tensor(buf25, (32768, 512), (512, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [transpose_6, hidden_states_10, hidden_states_12], Original ATen: [aten.transpose, aten.view, aten.addmm]
        # [Provenance debug handles] extern_kernels.bias_addmm:20
        extern_kernels.bias_addmm(reinterpret_tensor(_frozen_param21, (32768, 512), (0, 1), 0), reinterpret_tensor(buf27, (32768, 512), (512, 1), 0), _frozen_param149, alpha=1, beta=1, out=buf31)
        del buf27
        buf32 = reinterpret_tensor(buf31, (8, 512, 64, 64), (2097152, 1, 32768, 512), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [z, z_1, sample, hidden_states_4, hidden_states_6, add, output_tensor, hidden_states_12, transpose_7, hidden_states_14, hidden_states_15, hidden_states_16], Original ATen: [aten.div, aten.convolution, aten.silu, aten.add, aten.view, aten.transpose]
        # [Provenance debug handles] triton_poi_fused_add_convolution_div_silu_transpose_view_7:21
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_transpose_view_7.run(buf32, buf3, _frozen_param3, buf15, _frozen_param11, 16777216, stream=stream0)
        del buf15
        buf33 = buf17; del buf17  # reuse
        buf34 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_8:22
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_8.run(buf32, buf33, buf34, 256, 65536, stream=stream0)
        buf37 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_9:23
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_9.run(buf32, buf33, buf34, _frozen_param150, _frozen_param151, buf37, 16777216, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:24
        buf38 = extern_kernels.convolution(buf37, _frozen_param24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf39 = buf34; del buf34  # reuse
        buf40 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_div_native_group_norm_2:25
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_div_native_group_norm_2.run(buf38, _frozen_param25, buf39, buf40, 256, 65536, stream=stream0)
        buf43 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_div_native_group_norm_silu_3:26
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_div_native_group_norm_silu_3.run(buf38, _frozen_param25, buf39, buf40, _frozen_param152, _frozen_param153, buf43, 16777216, stream=stream0)
        del buf38
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:27
        buf44 = extern_kernels.convolution(buf43, _frozen_param28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf45 = buf40; del buf40  # reuse
        buf46 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10:28
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10.run(buf32, buf44, _frozen_param29, buf45, buf46, 256, 65536, stream=stream0)
        buf49 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_24, hidden_states_25], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11:29
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11.run(buf32, buf44, _frozen_param29, buf45, buf46, _frozen_param154, _frozen_param155, buf49, 16777216, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:30
        buf50 = extern_kernels.convolution(buf49, _frozen_param32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf51 = buf46; del buf46  # reuse
        buf52 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_div_native_group_norm_2:31
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_div_native_group_norm_2.run(buf50, _frozen_param33, buf51, buf52, 256, 65536, stream=stream0)
        buf55 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_div_native_group_norm_silu_3:32
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_div_native_group_norm_silu_3.run(buf50, _frozen_param33, buf51, buf52, _frozen_param156, _frozen_param157, buf55, 16777216, stream=stream0)
        del buf50
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:33
        buf56 = extern_kernels.convolution(buf55, _frozen_param36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf55
        buf57 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, hidden_states_28, hidden_states_30, add_3, output_tensor_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div]
        # [Provenance debug handles] triton_poi_fused_add_convolution_div_silu_12:34
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_12.run(buf57, buf44, _frozen_param29, buf56, _frozen_param37, 16777216, stream=stream0)
        del buf44
        buf58 = buf52; del buf52  # reuse
        buf59 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_8:35
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_8.run(buf57, buf58, buf59, 256, 65536, stream=stream0)
        buf62 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_9:36
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_9.run(buf57, buf58, buf59, _frozen_param158, _frozen_param159, buf62, 16777216, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:37
        buf63 = extern_kernels.convolution(buf62, _frozen_param40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf64 = buf59; del buf59  # reuse
        buf65 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_div_native_group_norm_2:38
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_div_native_group_norm_2.run(buf63, _frozen_param41, buf64, buf65, 256, 65536, stream=stream0)
        buf68 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_div_native_group_norm_silu_3:39
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_div_native_group_norm_silu_3.run(buf63, _frozen_param41, buf64, buf65, _frozen_param160, _frozen_param161, buf68, 16777216, stream=stream0)
        del buf63
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:40
        buf69 = extern_kernels.convolution(buf68, _frozen_param44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf70 = buf65; del buf65  # reuse
        buf71 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10:41
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_native_group_norm_silu_10.run(buf57, buf69, _frozen_param45, buf70, buf71, 256, 65536, stream=stream0)
        buf74 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38, hidden_states_39], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11:42
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_11.run(buf57, buf69, _frozen_param45, buf70, buf71, _frozen_param162, _frozen_param163, buf74, 16777216, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:43
        buf75 = extern_kernels.convolution(buf74, _frozen_param48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        buf76 = buf71; del buf71  # reuse
        buf77 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_div_native_group_norm_2:44
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_div_native_group_norm_2.run(buf75, _frozen_param49, buf76, buf77, 256, 65536, stream=stream0)
        buf80 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_div_native_group_norm_silu_3:45
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_div_native_group_norm_silu_3.run(buf75, _frozen_param49, buf76, buf77, _frozen_param164, _frozen_param165, buf80, 16777216, stream=stream0)
        del buf75
        # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_44], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:46
        buf81 = extern_kernels.convolution(buf80, _frozen_param52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 512, 64, 64), (2097152, 1, 32768, 512), 'torch.ops.aten.convolution.default')
        del buf80
        buf83 = empty_strided_cuda((8, 512, 128, 128), (8388608, 1, 65536, 512), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_13:47
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_13.run(buf57, buf69, _frozen_param45, buf81, _frozen_param53, buf83, 67108864, stream=stream0)
        del buf57
        del buf69
        del buf81
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:48
        buf84 = extern_kernels.convolution(buf83, _frozen_param54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf85 = buf77; del buf77  # reuse
        buf86 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy_clone_convolution_native_group_norm_14:49
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_clone_convolution_native_group_norm_14.run(buf84, _frozen_param55, buf85, buf86, 256, 262144, stream=stream0)
        buf89 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_47, hidden_states_48], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15:50
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15.run(buf84, _frozen_param55, buf85, buf86, _frozen_param166, _frozen_param167, buf89, 67108864, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:51
        buf90 = extern_kernels.convolution(buf89, _frozen_param58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf91 = buf86; del buf86  # reuse
        buf92 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy_clone_convolution_native_group_norm_14:52
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_clone_convolution_native_group_norm_14.run(buf90, _frozen_param59, buf91, buf92, 256, 262144, stream=stream0)
        buf95 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50, hidden_states_51], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15:53
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15.run(buf90, _frozen_param59, buf91, buf92, _frozen_param168, _frozen_param169, buf95, 67108864, stream=stream0)
        del buf90
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:54
        buf96 = extern_kernels.convolution(buf95, _frozen_param62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf97 = buf92; del buf92  # reuse
        buf98 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16:55
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16.run(buf84, _frozen_param55, buf96, _frozen_param63, buf97, buf98, 256, 262144, stream=stream0)
        buf101 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54, hidden_states_55], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17:56
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17.run(buf84, _frozen_param55, buf96, _frozen_param63, buf97, buf98, _frozen_param170, _frozen_param171, buf101, 67108864, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:57
        buf102 = extern_kernels.convolution(buf101, _frozen_param66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf103 = buf98; del buf98  # reuse
        buf104 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy_clone_convolution_native_group_norm_14:58
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_clone_convolution_native_group_norm_14.run(buf102, _frozen_param67, buf103, buf104, 256, 262144, stream=stream0)
        buf107 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57, hidden_states_58], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15:59
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15.run(buf102, _frozen_param67, buf103, buf104, _frozen_param172, _frozen_param173, buf107, 67108864, stream=stream0)
        del buf102
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_60], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:60
        buf108 = extern_kernels.convolution(buf107, _frozen_param70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        del buf107
        buf109 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_58, hidden_states_60, add_7, output_tensor_6], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.silu, aten.add, aten.div]
        # [Provenance debug handles] triton_poi_fused__to_copy_add_clone_convolution_div_silu_18:61
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_convolution_div_silu_18.run(buf109, _frozen_param55, buf96, _frozen_param63, buf108, _frozen_param71, 67108864, stream=stream0)
        del buf108
        buf110 = buf104; del buf104  # reuse
        buf111 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_19:62
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_19.run(buf109, buf110, buf111, 256, 262144, stream=stream0)
        buf114 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_20:63
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_20.run(buf109, buf110, buf111, _frozen_param174, _frozen_param175, buf114, 67108864, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:64
        buf115 = extern_kernels.convolution(buf114, _frozen_param74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        buf116 = buf111; del buf111  # reuse
        buf117 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy_clone_convolution_native_group_norm_14:65
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_clone_convolution_native_group_norm_14.run(buf115, _frozen_param75, buf116, buf117, 256, 262144, stream=stream0)
        buf120 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64, hidden_states_65], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15:66
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_convolution_native_group_norm_silu_15.run(buf115, _frozen_param75, buf116, buf117, _frozen_param176, _frozen_param177, buf120, 67108864, stream=stream0)
        del buf115
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:67
        buf121 = extern_kernels.convolution(buf120, _frozen_param78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
        del buf120
        buf122 = empty_strided_cuda((8, 512, 256, 256), (33554432, 1, 131072, 512), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_21:68
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_21.run(buf109, buf121, _frozen_param79, buf122, 268435456, stream=stream0)
        del buf109
        del buf121
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] extern_kernels.convolution:69
        buf123 = extern_kernels.convolution(buf122, _frozen_param80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_22:70
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_22.run(buf124, _frozen_param81, 268435456, stream=stream0)
        buf125 = buf117; del buf117  # reuse
        buf126 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_23:71
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_23.run(buf124, buf125, buf126, 256, 1048576, stream=stream0)
        buf129 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_24:72
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_24.run(buf124, buf125, buf126, _frozen_param178, _frozen_param179, buf129, 268435456, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:73
        buf130 = extern_kernels.convolution(buf129, _frozen_param84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf129
        buf131 = buf126; del buf126  # reuse
        buf132 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_native_group_norm_silu_25:74
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_native_group_norm_silu_25.run(buf130, _frozen_param85, buf131, buf132, 256, 524288, stream=stream0)
        buf134 = empty_strided_cuda((524288, 256), (256, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.bias_addmm:75
        extern_kernels.bias_addmm(reinterpret_tensor(_frozen_param91, (524288, 256), (0, 1), 0), reinterpret_tensor(buf124, (524288, 512), (512, 1), 0), reinterpret_tensor(_frozen_param90, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf134)
        del buf124
        buf136 = empty_strided_cuda((8, 256, 256, 256), (16777216, 1, 65536, 256), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73, hidden_states_74], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_native_group_norm_silu_26:76
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_native_group_norm_silu_26.run(buf130, _frozen_param85, buf131, buf132, _frozen_param180, _frozen_param181, buf136, 134217728, stream=stream0)
        del buf130
        # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:77
        buf137 = extern_kernels.convolution(buf136, _frozen_param88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf138 = buf132; del buf132  # reuse
        buf139 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_clone_convolution_div_native_group_norm_silu_27:78
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_native_group_norm_silu_27.run(buf134, buf137, _frozen_param89, buf138, buf139, 256, 524288, stream=stream0)
        buf142 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77, hidden_states_78], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_28:79
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_28.run(buf134, buf137, _frozen_param89, buf138, buf139, _frozen_param182, _frozen_param183, buf142, 134217728, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:80
        buf143 = extern_kernels.convolution(buf142, _frozen_param94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf144 = buf139; del buf139  # reuse
        buf145 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_native_group_norm_silu_25:81
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_native_group_norm_silu_25.run(buf143, _frozen_param95, buf144, buf145, 256, 524288, stream=stream0)
        buf148 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80, hidden_states_81], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_native_group_norm_silu_26:82
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_native_group_norm_silu_26.run(buf143, _frozen_param95, buf144, buf145, _frozen_param184, _frozen_param185, buf148, 134217728, stream=stream0)
        del buf143
        # Topologically Sorted Source Nodes: [hidden_states_81, hidden_states_83], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:83
        buf149 = extern_kernels.convolution(buf148, _frozen_param98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf148
        buf150 = reinterpret_tensor(buf134, (8, 256, 256, 256), (16777216, 1, 65536, 256), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_81, hidden_states_83, add_10, output_tensor_9], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
        # [Provenance debug handles] triton_poi_fused_add_convolution_div_silu_29:84
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_29.run(buf150, buf137, _frozen_param89, buf149, _frozen_param99, 134217728, stream=stream0)
        del buf137
        buf151 = buf145; del buf145  # reuse
        buf152 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_30:85
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_30.run(buf150, buf151, buf152, 256, 524288, stream=stream0)
        buf155 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_31:86
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_31.run(buf150, buf151, buf152, _frozen_param186, _frozen_param187, buf155, 134217728, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:87
        buf156 = extern_kernels.convolution(buf155, _frozen_param102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        buf157 = buf152; del buf152  # reuse
        buf158 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_native_group_norm_silu_25:88
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_native_group_norm_silu_25.run(buf156, _frozen_param103, buf157, buf158, 256, 524288, stream=stream0)
        buf161 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87, hidden_states_88], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_native_group_norm_silu_26:89
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_native_group_norm_silu_26.run(buf156, _frozen_param103, buf157, buf158, _frozen_param188, _frozen_param189, buf161, 134217728, stream=stream0)
        del buf156
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:90
        buf162 = extern_kernels.convolution(buf161, _frozen_param106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 256, 256, 256), (16777216, 1, 65536, 256), 'torch.ops.aten.convolution.default')
        del buf161
        buf163 = empty_strided_cuda((8, 256, 512, 512), (67108864, 1, 131072, 256), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_32:91
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_32.run(buf150, buf162, _frozen_param107, buf163, 536870912, stream=stream0)
        del buf150
        del buf162
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] extern_kernels.convolution:92
        buf164 = extern_kernels.convolution(buf163, _frozen_param108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.arange, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33:93
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33.run(buf165, _frozen_param109, 536870912, stream=stream0)
        buf166 = buf158; del buf158  # reuse
        buf167 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_34:94
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_34.run(buf165, buf166, buf167, 256, 2097152, stream=stream0)
        buf170 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_35:95
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_35.run(buf165, buf166, buf167, _frozen_param190, _frozen_param191, buf170, 536870912, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:96
        buf171 = extern_kernels.convolution(buf170, _frozen_param112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del buf170
        buf172 = buf167; del buf167  # reuse
        buf173 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_native_group_norm_silu_36:97
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_native_group_norm_silu_36.run(buf171, _frozen_param113, buf172, buf173, 256, 1048576, stream=stream0)
        buf175 = empty_strided_cuda((2097152, 128), (128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.bias_addmm:98
        extern_kernels.bias_addmm(reinterpret_tensor(_frozen_param119, (2097152, 128), (0, 1), 0), reinterpret_tensor(buf165, (2097152, 256), (256, 1), 0), reinterpret_tensor(_frozen_param118, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf175)
        del buf165
        buf177 = empty_strided_cuda((8, 128, 512, 512), (33554432, 1, 65536, 128), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96, hidden_states_97], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_native_group_norm_silu_37:99
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_native_group_norm_silu_37.run(buf171, _frozen_param113, buf172, buf173, _frozen_param192, _frozen_param193, buf177, 268435456, stream=stream0)
        del buf171
        # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:100
        buf178 = extern_kernels.convolution(buf177, _frozen_param116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf179 = buf173; del buf173  # reuse
        buf180 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38:101
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38.run(buf175, buf178, _frozen_param117, buf179, buf180, 256, 1048576, stream=stream0)
        buf183 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100, hidden_states_101], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39:102
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39.run(buf175, buf178, _frozen_param117, buf179, buf180, _frozen_param194, _frozen_param195, buf183, 268435456, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:103
        buf184 = extern_kernels.convolution(buf183, _frozen_param122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf185 = buf180; del buf180  # reuse
        buf186 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_native_group_norm_silu_36:104
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_native_group_norm_silu_36.run(buf184, _frozen_param123, buf185, buf186, 256, 1048576, stream=stream0)
        buf189 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103, hidden_states_104], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_native_group_norm_silu_37:105
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_native_group_norm_silu_37.run(buf184, _frozen_param123, buf185, buf186, _frozen_param196, _frozen_param197, buf189, 268435456, stream=stream0)
        del buf184
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_106], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:106
        buf190 = extern_kernels.convolution(buf189, _frozen_param126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        del buf189
        buf191 = reinterpret_tensor(buf175, (8, 128, 512, 512), (33554432, 1, 65536, 128), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_104, hidden_states_106, add_13, output_tensor_12], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
        # [Provenance debug handles] triton_poi_fused_add_convolution_div_silu_40:107
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_40.run(buf191, buf178, _frozen_param117, buf190, _frozen_param127, 268435456, stream=stream0)
        del buf178
        buf192 = buf186; del buf186  # reuse
        buf193 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_41:108
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_41.run(buf191, buf192, buf193, 256, 1048576, stream=stream0)
        buf196 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_42:109
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_42.run(buf191, buf192, buf193, _frozen_param198, _frozen_param199, buf196, 268435456, stream=stream0)
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:110
        buf197 = extern_kernels.convolution(buf196, _frozen_param130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf198 = buf193; del buf193  # reuse
        buf199 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_convolution_native_group_norm_silu_36:111
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_convolution_native_group_norm_silu_36.run(buf197, _frozen_param131, buf198, buf199, 256, 1048576, stream=stream0)
        buf202 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110, hidden_states_111], Original ATen: [aten.silu, aten.convolution, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_clone_convolution_native_group_norm_silu_37:112
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convolution_native_group_norm_silu_37.run(buf197, _frozen_param131, buf198, buf199, _frozen_param200, _frozen_param201, buf202, 268435456, stream=stream0)
        del buf197
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:113
        buf203 = extern_kernels.convolution(buf202, _frozen_param134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 128, 512, 512), (33554432, 1, 65536, 128), 'torch.ops.aten.convolution.default')
        buf204 = buf199; del buf199  # reuse
        buf205 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_1], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38:114
        stream0 = get_raw_stream(0)
        triton_red_fused_add_clone_convolution_div_native_group_norm_silu_38.run(buf191, buf203, _frozen_param135, buf204, buf205, 256, 1048576, stream=stream0)
        buf208 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_1, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39:115
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_39.run(buf191, buf203, _frozen_param135, buf204, buf205, _frozen_param202, _frozen_param203, buf208, 268435456, stream=stream0)
        del buf191
        del buf203
        del buf204
        del buf205
        # Topologically Sorted Source Nodes: [sample_2, sample_3], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:116
        buf209 = extern_kernels.convolution(buf208, _frozen_param205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 3, 512, 512), (786432, 1, 1536, 3), 'torch.ops.aten.convolution.default')
        del buf208
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [sample_2, sample_3, clamp], Original ATen: [aten.silu, aten.convolution, aten.clamp]
        # [Provenance debug handles] triton_poi_fused_clamp_convolution_silu_43:117
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_convolution_silu_43.run(buf210, 6291456, stream=stream0)
    return (buf210, )


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
        arg140_1, = args
        args.clear()
        partition0_args = [arg140_1,]
        del arg140_1
        (buf210,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf210, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    global _frozen_param0
    _frozen_param0 = rand_strided((4, 4, 1, 1), (4, 1, 4, 4), device='cuda:0', dtype=torch.float16)
    global _frozen_param2
    _frozen_param2 = rand_strided((512, 4, 3, 3), (36, 1, 12, 4), device='cuda:0', dtype=torch.float16)
    global _frozen_param3
    _frozen_param3 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param6
    _frozen_param6 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param7
    _frozen_param7 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param10
    _frozen_param10 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param11
    _frozen_param11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param15
    _frozen_param15 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param17
    _frozen_param17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param19
    _frozen_param19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param21
    _frozen_param21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param24
    _frozen_param24 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param25
    _frozen_param25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param28
    _frozen_param28 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param29
    _frozen_param29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param32
    _frozen_param32 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param33
    _frozen_param33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param36
    _frozen_param36 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param37
    _frozen_param37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param40
    _frozen_param40 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param41
    _frozen_param41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param44
    _frozen_param44 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param45
    _frozen_param45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param48
    _frozen_param48 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param49
    _frozen_param49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param52
    _frozen_param52 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param53
    _frozen_param53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param54
    _frozen_param54 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param55
    _frozen_param55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param58
    _frozen_param58 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param59
    _frozen_param59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param62
    _frozen_param62 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param63
    _frozen_param63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param66
    _frozen_param66 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param67
    _frozen_param67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param70
    _frozen_param70 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param71
    _frozen_param71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param74
    _frozen_param74 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param75
    _frozen_param75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param78
    _frozen_param78 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param79
    _frozen_param79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param80
    _frozen_param80 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param81
    _frozen_param81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param84
    _frozen_param84 = rand_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param85
    _frozen_param85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param88
    _frozen_param88 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param89
    _frozen_param89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param90
    _frozen_param90 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param91
    _frozen_param91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param94
    _frozen_param94 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param95
    _frozen_param95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param98
    _frozen_param98 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param99
    _frozen_param99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param102
    _frozen_param102 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param103
    _frozen_param103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param106
    _frozen_param106 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param107
    _frozen_param107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param108
    _frozen_param108 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param109
    _frozen_param109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param112
    _frozen_param112 = rand_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param113
    _frozen_param113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param116
    _frozen_param116 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    global _frozen_param117
    _frozen_param117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param118
    _frozen_param118 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float16)
    global _frozen_param119
    _frozen_param119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param122
    _frozen_param122 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    global _frozen_param123
    _frozen_param123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param126
    _frozen_param126 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    global _frozen_param127
    _frozen_param127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param130
    _frozen_param130 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    global _frozen_param131
    _frozen_param131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param134
    _frozen_param134 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    global _frozen_param135
    _frozen_param135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    global _frozen_param140
    _frozen_param140 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param141
    _frozen_param141 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param142
    _frozen_param142 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param143
    _frozen_param143 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param144
    _frozen_param144 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param145
    _frozen_param145 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param146
    _frozen_param146 = rand_strided((8, 512, 512), (0, 1, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param147
    _frozen_param147 = rand_strided((8, 512, 512), (0, 1, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param148
    _frozen_param148 = rand_strided((8, 512, 512), (0, 1, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param149
    _frozen_param149 = rand_strided((512, 512), (1, 512), device='cuda:0', dtype=torch.float16)
    global _frozen_param150
    _frozen_param150 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param151
    _frozen_param151 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param152
    _frozen_param152 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param153
    _frozen_param153 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param154
    _frozen_param154 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param155
    _frozen_param155 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param156
    _frozen_param156 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param157
    _frozen_param157 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param158
    _frozen_param158 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param159
    _frozen_param159 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param160
    _frozen_param160 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param161
    _frozen_param161 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param162
    _frozen_param162 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param163
    _frozen_param163 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param164
    _frozen_param164 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param165
    _frozen_param165 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param166
    _frozen_param166 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param167
    _frozen_param167 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param168
    _frozen_param168 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param169
    _frozen_param169 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param170
    _frozen_param170 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param171
    _frozen_param171 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param172
    _frozen_param172 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param173
    _frozen_param173 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param174
    _frozen_param174 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param175
    _frozen_param175 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param176
    _frozen_param176 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param177
    _frozen_param177 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param178
    _frozen_param178 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param179
    _frozen_param179 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param180
    _frozen_param180 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param181
    _frozen_param181 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param182
    _frozen_param182 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param183
    _frozen_param183 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param184
    _frozen_param184 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param185
    _frozen_param185 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param186
    _frozen_param186 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param187
    _frozen_param187 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param188
    _frozen_param188 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param189
    _frozen_param189 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param190
    _frozen_param190 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param191
    _frozen_param191 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param192
    _frozen_param192 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param193
    _frozen_param193 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param194
    _frozen_param194 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param195
    _frozen_param195 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param196
    _frozen_param196 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param197
    _frozen_param197 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param198
    _frozen_param198 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param199
    _frozen_param199 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param200
    _frozen_param200 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param201
    _frozen_param201 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param202
    _frozen_param202 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param203
    _frozen_param203 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    global _frozen_param205
    _frozen_param205 = rand_strided((3, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float16)
    arg140_1 = rand_strided((8, 4, 64, 64), (16384, 1, 256, 4), device='cuda:0', dtype=torch.float16)
    return [arg140_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
