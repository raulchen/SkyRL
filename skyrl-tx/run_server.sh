#!/bin/bash
# TX Server runner script for testing flash attention

# Configuration
HOST="0.0.0.0"
PORT="8001"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
DB_PATH="/tmp/tinker.db"

# Backend config options
TP_SIZE=8
MAX_LORA_ADAPTERS=2
TRAIN_MICRO_BATCH=4
SAMPLE_MAX_SEQUENCES=32
SHARD_ATTENTION_HEADS=false  # Must be false for cuDNN flash attention

# Debug options (set to true to enable)
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
DUMP_XLA=${DUMP_XLA:-false}  # Dump XLA graphs to /tmp/xla_dump

# Memory optimization
LOSS_CHUNK_SIZE=${LOSS_CHUNK_SIZE:-1024}  # Chunk size for cross-entropy loss (0 = disabled)

# Build backend config JSON
BACKEND_CONFIG=$(cat <<EOF
{
  "tensor_parallel_size": ${TP_SIZE},
  "max_lora_adapters": ${MAX_LORA_ADAPTERS},
  "train_micro_batch_size": ${TRAIN_MICRO_BATCH},
  "sample_max_num_sequences": ${SAMPLE_MAX_SEQUENCES},
  "shard_attention_heads": ${SHARD_ATTENTION_HEADS},
  "enforce_eager": ${ENFORCE_EAGER},
  "gradient_checkpointing": ${GRADIENT_CHECKPOINTING},
  "loss_chunk_size": ${LOSS_CHUNK_SIZE}
}
EOF
)

# Clean up old database
rm -f "$DB_PATH"

# Backup existing log
[ -f ./tx.log ] && mv ./tx.log ./tx.log.bak

echo "Starting TX server..."
echo "  Host: $HOST:$PORT"
echo "  Model: $BASE_MODEL"
echo "  TP Size: $TP_SIZE"
echo "  Shard Attention Heads: $SHARD_ATTENTION_HEADS"
echo "  Enforce Eager (no JIT): $ENFORCE_EAGER"
echo "  Loss Chunk Size: $LOSS_CHUNK_SIZE"
echo "  Dump XLA: $DUMP_XLA"
echo ""

# Set up XLA flags for dumping if enabled
XLA_FLAGS_VAR=""
if [ "$DUMP_XLA" = "true" ]; then
    echo "Cleaning up old XLA dump..."
    rm -rf /tmp/xla_dump
    mkdir -p /tmp/xla_dump
    XLA_FLAGS_VAR="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
    echo "XLA graphs will be dumped to /tmp/xla_dump"
fi

# Run server
# XLA_PYTHON_CLIENT_PREALLOCATE=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
XLA_FLAGS="$XLA_FLAGS_VAR" \
uv run --extra tinker --extra gpu -m tx.tinker.api \
  --host "$HOST" \
  --port "$PORT" \
  --base-model "$BASE_MODEL" \
  --database-url "sqlite:///$DB_PATH" \
  --backend-config "$BACKEND_CONFIG" 2>&1 | tee ./tx.log
