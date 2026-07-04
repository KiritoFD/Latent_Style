#!/usr/bin/env bash
echo "=== Testing network connectivity ==="
echo "--- hf-mirror.com ---"
curl -sI --connect-timeout 10 https://hf-mirror.com 2>&1 | head -3
echo ""
echo "--- huggingface.co ---"
curl -sI --connect-timeout 10 https://huggingface.co 2>&1 | head -3
echo ""
echo "=== DONE ==="
