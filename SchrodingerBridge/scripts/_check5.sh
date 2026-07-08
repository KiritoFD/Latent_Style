#!/bin/bash
echo "=== check if selective_scan.o exists ==="
ls -la /tmp/mamba_full/mamba-1.2.2/build/temp.linux-x86_64-3.10/csrc/selective_scan/selective_scan.o 2>&1

echo ""
echo "=== grep 'error' in mamba_full.log ==="
grep -iE "error|fatal|failed" /tmp/mamba_full.log | head -50

echo ""
echo "=== grep 'selective_scan.cpp' in mamba_full.log ==="
grep -B2 -A5 "selective_scan.cpp" /tmp/mamba_full.log | head -50

echo ""
echo "=== Full log size ==="
wc -l /tmp/mamba_full.log

echo ""
echo "=== First 100 lines of mamba_full.log ==="
head -100 /tmp/mamba_full.log

echo ""
echo "=== Manual compile of selective_scan.cpp ==="
cd /tmp/mamba_full/mamba-1.2.2
source /root/samam_venv/bin/activate
g++ -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/tmp/mamba_full/mamba-1.2.2/csrc/selective_scan -I/root/samam_venv/lib/python3.10/site-packages/torch/include -I/root/samam_venv/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/root/samam_venv/lib/python3.10/site-packages/torch/include/TH -I/root/samam_venv/lib/python3.10/site-packages/torch/include/THC -I/usr/local/cuda/include -I/root/samam_venv/include -I/usr/include/python3.10 -c csrc/selective_scan/selective_scan.cpp -o /tmp/selective_scan_test.o -O3 -std=c++17 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE="_gcc" -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011" -DTORCH_EXTENSION_NAME=selective_scan_cuda -D_GLIBCXX_USE_CXX11_ABI=0 2>&1 | head -30
echo "RC=$?"
