#!/bin/bash

# Set to halt on errors
set -e

echo "===== Running GCB Metadata Tests ====="
# Install pytest if needed
pip install pytest
pytest tests/test_gcb_metadata.py -v

echo "===== Running Basic Runtime Tests ====="
# Set environment variables for minimal logging
export DEBUG=1
export DEBUG_LAYER=0       # Log only layer 0
export DEBUG_HEAD=0        # Log only head 0
export DEBUG_STEP_START=0
export DEBUG_STEP_END=20   # Log first 20 steps to see scaling effects
export LOG_INTERVAL=1

python scripts/run_gcb.py \
    --ckpt gemma1b_gcb.pt \
    --meta gemma1b_gcb.pkl \
    --prompt "1+1=" \
    --out_len 20 \
    --temperature 0 \
    --device cuda \
    --top_k 1 > stdout.log 2> stderr.log

echo "===== Test Results ====="
echo "Check stdout.log for any assertion failures or warnings"
echo "Check stderr.log for tensor statistics and debugging information"