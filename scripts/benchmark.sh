#!/bin/bash
# Benchmark H-JEPA Model Performance
# Usage: ./scripts/benchmark.sh [OPTIONS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH=""
DEVICE="cuda"
BATCH_SIZES="1 4 8 16 32"
NUM_RUNS=100
WARMUP_RUNS=10
OUTPUT_FILE="benchmark_results.json"

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model PATH         Path to model checkpoint [required]"
    echo "  -d, --device DEVICE      Device (cpu|cuda) [default: cuda]"
    echo "  -b, --batch-sizes SIZES  Batch sizes to test [default: \"1 4 8 16 32\"]"
    echo "  -n, --num-runs NUM       Number of benchmark runs [default: 100]"
    echo "  -w, --warmup NUM         Number of warmup runs [default: 10]"
    echo "  -o, --output FILE        Output JSON file [default: benchmark_results.json]"
    echo "  -h, --help               Show this help message"
    echo ""
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -b|--batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        -n|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate model path
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path is required${NC}"
    usage
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}H-JEPA Performance Benchmark${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Model: ${YELLOW}$MODEL_PATH${NC}"
echo -e "Device: ${YELLOW}$DEVICE${NC}"
echo -e "Batch sizes: ${YELLOW}$BATCH_SIZES${NC}"
echo -e "Runs per batch: ${YELLOW}$NUM_RUNS${NC}"
echo -e "Warmup runs: ${YELLOW}$WARMUP_RUNS${NC}"
echo ""

# Create Python benchmark script
cat > /tmp/benchmark_hjepa.py <<'EOF'
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hjepa import create_hjepa
from src.inference.optimized_model import OptimizedHJEPA, BatchInference

def benchmark_model(
    model_path,
    device,
    batch_sizes,
    num_runs,
    warmup_runs,
):
    """Run comprehensive benchmark."""

    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Create model
    model = create_hjepa(
        encoder_type=model_config.get('encoder_type', 'vit_base_patch16_224'),
        img_size=224,
        embed_dim=model_config.get('embed_dim', 768),
        predictor_depth=model_config.get('predictor', {}).get('depth', 6),
        predictor_num_heads=model_config.get('predictor', {}).get('num_heads', 12),
        num_hierarchies=model_config.get('num_hierarchies', 3),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create optimized model
    opt_model = OptimizedHJEPA(model, hierarchy_level=0)
    opt_model.to(device)
    opt_model.eval()

    results = {
        'model_path': model_path,
        'device': device,
        'model_config': model_config,
        'benchmarks': []
    }

    # Benchmark each batch size
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")

        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

        # Warmup
        print(f"  Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = opt_model(dummy_input)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        print(f"  Running benchmark ({num_runs} runs)...")
        times = []

        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()

            with torch.no_grad():
                _ = opt_model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            times.append(time.time() - start_time)

        # Calculate statistics
        times = np.array(times)

        batch_results = {
            'batch_size': batch_size,
            'mean_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'median_time_ms': float(np.median(times) * 1000),
            'throughput_images_per_sec': float(batch_size / np.mean(times)),
            'latency_per_image_ms': float((np.mean(times) / batch_size) * 1000),
        }

        results['benchmarks'].append(batch_results)

        print(f"  Mean time: {batch_results['mean_time_ms']:.2f} ms")
        print(f"  Throughput: {batch_results['throughput_images_per_sec']:.2f} images/sec")
        print(f"  Latency per image: {batch_results['latency_per_image_ms']:.2f} ms")

    # Memory usage (if CUDA)
    if device == 'cuda':
        results['memory'] = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
        }

        print(f"\nMemory usage:")
        print(f"  Allocated: {results['memory']['allocated_mb']:.2f} MB")
        print(f"  Reserved: {results['memory']['reserved_mb']:.2f} MB")
        print(f"  Max allocated: {results['memory']['max_allocated_mb']:.2f} MB")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-sizes', nargs='+', type=int, required=True)
    parser.add_argument('--num-runs', type=int, default=100)
    parser.add_argument('--warmup-runs', type=int, default=10)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    results = benchmark_model(
        args.model_path,
        args.device,
        args.batch_sizes,
        args.num_runs,
        args.warmup_runs,
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
EOF

# Run benchmark
echo -e "${YELLOW}Running benchmark...${NC}"
python3 /tmp/benchmark_hjepa.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --batch-sizes $BATCH_SIZES \
    --num-runs "$NUM_RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    --output "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Results saved to: ${YELLOW}$OUTPUT_FILE${NC}"
echo ""

# Print summary
echo -e "${GREEN}Summary:${NC}"
python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)
print('')
print('Batch Size | Throughput (img/s) | Latency (ms/img)')
print('-' * 55)
for b in results['benchmarks']:
    print(f'{b[\"batch_size\"]:10d} | {b[\"throughput_images_per_sec\"]:18.2f} | {b[\"latency_per_image_ms\"]:16.2f}')
"

echo ""
echo -e "${YELLOW}View full results with:${NC} cat $OUTPUT_FILE | jq ."
