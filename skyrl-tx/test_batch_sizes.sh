#!/bin/bash
# Test different batch sizes for sampling/training and report max GPU memory usage

set -e

# Configuration
SERVER_SCRIPT="./run_server.sh"
CLIENT_SCRIPT="uv run python test_long_sequence.py"
WARMUP_TIME=10  # seconds to wait for server startup
GPU_POLL_INTERVAL=1  # seconds between GPU memory polls

# Test parameters (override via command line or environment)
TEST_MODE=${TEST_MODE:-"sample"}  # "sample" or "train"
BATCH_SIZES=${BATCH_SIZES:-"4 8 16 32"}  # space-separated list
SEQ_LEN=${SEQ_LEN:-8192}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cleanup() {
    log_info "Cleaning up..."
    # Kill GPU monitor if running
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    # Kill server if running
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

kill_existing() {
    log_info "Killing existing server/client processes..."
    # Kill any existing test clients
    pkill -f "test_long_sequence.py" 2>/dev/null || true
    # Kill any existing servers
    pkill -f "tx.tinker.api" 2>/dev/null || true
    sleep 2
}

start_gpu_monitor() {
    local output_file=$1
    log_info "Starting GPU memory monitor -> $output_file"
    # Poll GPU memory and write max memory per GPU
    (
        while true; do
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null
            sleep "$GPU_POLL_INTERVAL"
        done
    ) > "$output_file" &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    GPU_MONITOR_PID=""
}

get_max_gpu_memory() {
    local monitor_file=$1
    # Sort by memory (2nd column) descending and show top 30 lines
    if [[ -f "$monitor_file" ]]; then
        sort -t',' -k2 -nr "$monitor_file" | head -30
    fi
}

get_peak_memory() {
    local monitor_file=$1
    # Return the single highest memory value across all GPUs
    if [[ -f "$monitor_file" ]]; then
        sort -t',' -k2 -nr "$monitor_file" | head -1 | cut -d',' -f2 | tr -d ' '
    else
        echo "0"
    fi
}

run_test() {
    local batch_size=$1
    local mode=$2
    local seq_len=$3

    log_info "=========================================="
    log_info "Testing: mode=$mode, batch_size=$batch_size, seq_len=$seq_len"
    log_info "=========================================="

    # Create temp file for GPU monitoring
    local gpu_monitor_file=$(mktemp)

    # Kill any existing processes
    kill_existing

    # Set environment variables for server based on mode
    if [[ "$mode" == "sample" ]]; then
        export SAMPLE_MAX_SEQUENCES=$batch_size
    else
        export TRAIN_MICRO_BATCH=$batch_size
    fi

    # Start server in background
    log_info "Starting server with SAMPLE_MAX_SEQUENCES=$SAMPLE_MAX_SEQUENCES TRAIN_MICRO_BATCH=$TRAIN_MICRO_BATCH"
    $SERVER_SCRIPT > /tmp/server_${mode}_${batch_size}.log 2>&1 &
    SERVER_PID=$!

    # Wait for server to load model
    log_info "Waiting ${WARMUP_TIME}s for server startup..."
    sleep "$WARMUP_TIME"

    # Start GPU monitoring
    start_gpu_monitor "$gpu_monitor_file"

    # Run the test
    local test_result=0
    log_info "Running test client..."
    if [[ "$mode" == "sample" ]]; then
        $CLIENT_SCRIPT --skip-train --sample-n "$batch_size" --sample-tokens "$seq_len" 2>&1 | tee /tmp/client_${mode}_${batch_size}.log || test_result=$?
    else
        $CLIENT_SCRIPT --skip-sample --train-batch "$batch_size" --train-seq "$seq_len" 2>&1 | tee /tmp/client_${mode}_${batch_size}.log || test_result=$?
    fi

    # Stop GPU monitoring
    stop_gpu_monitor

    # Kill server
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""

    # Get peak memory
    local peak_mem=$(get_peak_memory "$gpu_monitor_file")
    local status="PASS"
    if [[ $test_result -ne 0 ]]; then
        status="FAIL"
    fi

    # Report results
    echo ""
    if [[ $test_result -eq 0 ]]; then
        log_info "Result: ${GREEN}PASS${NC}"
    else
        log_error "Result: ${RED}FAIL${NC}"
    fi

    log_info "Top GPU memory usage (gpu_idx, memory_mib):"
    get_max_gpu_memory "$gpu_monitor_file"

    # Log to results file
    echo "$mode,$batch_size,$seq_len,$status,$peak_mem" >> "$RESULTS_FILE"

    # Cleanup temp file
    rm -f "$gpu_monitor_file"

    return $test_result
}

# Main
main() {
    # Create results file
    RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).csv"
    echo "mode,batch_size,seq_len,status,peak_gpu_mem_mib" > "$RESULTS_FILE"

    echo "=========================================="
    echo "Batch Size Test Runner"
    echo "=========================================="
    echo "Mode: $TEST_MODE"
    echo "Batch sizes: $BATCH_SIZES"
    echo "Sequence length: $SEQ_LEN"
    echo "Results file: $RESULTS_FILE"
    echo ""

    for batch_size in $BATCH_SIZES; do
        run_test "$batch_size" "$TEST_MODE" "$SEQ_LEN" || true
        # Small delay between tests
        sleep 5
    done

    log_info "=========================================="
    log_info "All tests complete"
    log_info "Results saved to: $RESULTS_FILE"
    log_info "=========================================="
    cat "$RESULTS_FILE"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            TEST_MODE="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --mode MODE          Test mode: 'sample' or 'train' (default: sample)"
            echo "  --batch-sizes SIZES  Space-separated batch sizes (default: '4 8 16 32')"
            echo "  --seq-len LEN        Sequence length (default: 8192)"
            echo ""
            echo "Examples:"
            echo "  $0 --mode sample --batch-sizes '4 8 16 32' --seq-len 8192"
            echo "  $0 --mode train --batch-sizes '2 4 8' --seq-len 4096"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
