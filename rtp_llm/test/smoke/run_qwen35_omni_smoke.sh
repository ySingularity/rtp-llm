#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export CHECKPOINT_PATH="/home/xieshui.yyx/happymusic/checkpoint-8000"
export MODEL_TYPE="qwen35_omni"
export TOKENIZER_PATH="$CHECKPOINT_PATH"
export AUDIO_TOKENIZER_PATH="/home/xieshui.yyx/happymusic/happy_music_tokenizer/model.ckpt"
export SAVE_RESPONSE=True

TP_SIZE=4
PORT=8930
MAX_SEQ_LEN=4096
RESERVE_MEM=10000

echo "=== Qwen3.5 Omni Smoke Test ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "TP size: $TP_SIZE"
echo "Port: $PORT"
echo ""

LOG_DIR="$SCRIPT_DIR/smoke_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qwen35_omni_smoke.log"

echo "Starting server (log: $LOG_FILE)..."
export START_PORT=$PORT

/opt/conda310/bin/python -m rtp_llm.start_server \
    --tp_size $TP_SIZE \
    --act_type BF16 \
    --max_seq_len $MAX_SEQ_LEN \
    --reserver_runtime_mem_mb $RESERVE_MEM \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill -TERM $SERVER_PID 2>/dev/null || true
    sleep 2
    kill -9 $SERVER_PID 2>/dev/null || true
    # kill children
    pkill -P $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server health..."
MAX_WAIT=600
for i in $(seq 1 $MAX_WAIT); do
    if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Last 50 lines of log:"
        tail -50 "$LOG_FILE"
        exit 1
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
        tail -50 "$LOG_FILE"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Sending test query ==="
RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "请简单介绍一下你自己"}],
        "max_tokens": 50,
        "temperature": 0.0,
        "top_p": 0.01,
        "top_k": 1
    }')

echo "Response:"
echo "$RESPONSE" | /opt/conda310/bin/python -m json.tool 2>/dev/null || echo "$RESPONSE"

echo ""
echo "=== Smoke test complete ==="
