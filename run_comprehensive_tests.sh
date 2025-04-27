#!/bin/bash

# Set to halt on errors
set -e

# Create directories for logs
mkdir -p logs/metadata
mkdir -p logs/basic
mkdir -p logs/long
mkdir -p logs/multi_layer_head

echo "===== Running GCB Metadata Tests ====="
# Install pytest if needed
pip install pytest
pytest tests/test_gcb_metadata.py -v > logs/metadata/test_results.log 2>&1

echo "===== Running Basic Runtime Tests ====="
# Basic test with minimal logging
export DEBUG=1
export DEBUG_LAYER=0
export DEBUG_HEAD=0
export DEBUG_STEP_START=0
export DEBUG_STEP_END=5
export LOG_INTERVAL=1

python scripts/run_gcb.py \
    --ckpt gemma1b_gcb.pt \
    --meta gemma1b_gcb.pkl \
    --prompt "1+1=" \
    --out_len 5 \
    --temperature 0 \
    --device cuda \
    --top_k 1 > logs/basic/stdout_math.log 2> logs/basic/stderr_math.log

python scripts/run_gcb.py \
    --ckpt gemma1b_gcb.pt \
    --meta gemma1b_gcb.pkl \
    --prompt "The capital of France is" \
    --out_len 5 \
    --temperature 0 \
    --device cuda \
    --top_k 1 > logs/basic/stdout_factual.log 2> logs/basic/stderr_factual.log

echo "===== Running Long Sequence Tests ====="
# Long sequence tests to verify variance scaling
export DEBUG_STEP_END=32
export LOG_INTERVAL=2

python scripts/run_gcb.py \
    --ckpt gemma1b_gcb.pt \
    --meta gemma1b_gcb.pkl \
    --prompt "Write a short story about" \
    --out_len 32 \
    --temperature 0.7 \
    --device cuda \
    --top_k 40 > logs/long/stdout_creative.log 2> logs/long/stderr_creative.log

echo "===== Running Multi-Layer/Head Tests ====="
# Test multiple layers and heads
export DEBUG_LAYER=-1  # All layers
export DEBUG_HEAD=-1   # All heads
export DEBUG_STEP_START=0
export DEBUG_STEP_END=10
export LOG_INTERVAL=5  # Log every 5 steps to reduce volume

python scripts/run_gcb.py \
    --ckpt gemma1b_gcb.pt \
    --meta gemma1b_gcb.pkl \
    --prompt "Hello, my name is" \
    --out_len 10 \
    --temperature 0 \
    --device cuda \
    --top_k 1 > logs/multi_layer_head/stdout.log 2> logs/multi_layer_head/stderr.log

echo "===== Test Results ====="
echo "Metadata test results: logs/metadata/test_results.log"
echo "Basic test results:"
echo "  - Math prompt: logs/basic/stdout_math.log, logs/basic/stderr_math.log"
echo "  - Factual prompt: logs/basic/stdout_factual.log, logs/basic/stderr_factual.log"
echo "Long sequence test results: logs/long/stdout_creative.log, logs/long/stderr_creative.log"
echo "Multi-layer/head test results: logs/multi_layer_head/stdout.log, logs/multi_layer_head/stderr.log"
echo ""
echo "Look for:"
echo "1. Stable tail_log values across steps (not exploding)"
echo "2. core_log and tail_log variances around 1.0"
echo "3. No 'energy ratio too low' warnings after first few tokens"
echo "4. No NaN/Inf values in any tensors"