#!/bin/bash
set -e

# Read all vLLM settings from configuration.yaml (env vars take priority)
CONFIG_FILE="${CONFIG_PATH:-/workspace/configuration.yaml}"

eval $(python3 -c "
import yaml, sys

try:
    with open('$CONFIG_FILE') as f:
        j = yaml.safe_load(f).get('judge', {})
except Exception as e:
    print(f'echo \"[WARNING] Could not read config: {e}\"', file=sys.stderr)
    j = {}

def val(key, default):
    return j.get(key, default)

enabled = 'true' if val('enabled', True) else 'false'
print(f'JUDGE_ENABLED_CFG={enabled}')
print(f'VLLM_PORT_CFG={val(\"vllm_port\", 8095)}')
print(f'VLLM_MODEL_CFG={val(\"vllm_model_name\", \"zai-org/GLM-4.1V-9B-Thinking\")}')
print(f'VLLM_REVISION_CFG={val(\"vllm_revision\", \"\")}')
print(f'VLLM_API_KEY_CFG={val(\"vllm_api_key\", \"local\")}')
print(f'GPU_UTIL_CFG={val(\"gpu_memory_utilization\", 0.20)}')
print(f'MAX_MODEL_LEN_CFG={val(\"max_model_len\", 8096)}')
print(f'MAX_NUM_SEQS_CFG={val(\"max_num_seqs\", 2)}')
")

# Env vars override configuration.yaml values
JUDGE_ENABLED=${JUDGE_ENABLED:-$JUDGE_ENABLED_CFG}
VLLM_PORT=${VLLM_PORT:-$VLLM_PORT_CFG}
VLLM_MODEL=${VLLM_MODEL:-$VLLM_MODEL_CFG}
VLLM_REVISION=${VLLM_REVISION:-$VLLM_REVISION_CFG}
API_KEY=${VLLM_API_KEY:-$VLLM_API_KEY_CFG}
GPU_UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-$GPU_UTIL_CFG}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$MAX_MODEL_LEN_CFG}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$MAX_NUM_SEQS_CFG}

if [ "$JUDGE_ENABLED" = "true" ]; then
    echo "STARTING VLLM SERVER (Isolated Env)"
    echo "Model: $VLLM_MODEL"
    echo "Revision: $VLLM_REVISION"
    echo "Port: $VLLM_PORT"
    echo "GPU Util: $GPU_UTIL"
    echo "Max model len: $MAX_MODEL_LEN  Max num seqs: $MAX_NUM_SEQS"

    # Start vLLM server in background
    /opt/vllm-env/bin/vllm serve "$VLLM_MODEL" \
        --revision "$VLLM_REVISION" \
        --port "$VLLM_PORT" \
        --api-key "$API_KEY" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max_num_seqs "$MAX_NUM_SEQS" &

    VLLM_PID=$!

    # Wait for vLLM to be ready (health check)
    echo "Waiting for vLLM to become ready..."
    MAX_RETRIES=150
    COUNTER=0

    while [ $COUNTER -lt $MAX_RETRIES ]; do
        if curl -s -f "http://localhost:$VLLM_PORT/health" > /dev/null; then
            echo "vLLM is READY!"
            break
        fi

        echo "Loading model ($COUNTER/$MAX_RETRIES)"
        sleep 5
        let COUNTER=COUNTER+1
    done

    if [ $COUNTER -eq $MAX_RETRIES ]; then
        echo "vLLM failed to start within timeout."
        kill $VLLM_PID
        exit 1
    fi
else
    echo "VLLM SERVER DISABLED (judge.enabled: false in configuration.yaml)"
fi

echo "STARTING MAIN FASTAPI SERVICE (Base Env)"

# Start the main pipeline service (foreground)
exec python serve.py
