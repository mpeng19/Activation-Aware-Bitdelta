#!/bin/bash

BASE_DIR="experiments/llama7b_calibration_finance"
mkdir -p $BASE_DIR

BASE_MODEL_MAP="{\"0\": \"70GiB\"}"
FINETUNED_MODEL_MAP="{\"0\": \"70GiB\"}"
COMPRESSED_MODEL_MAP="{\"1\": \"70GiB\"}"

CALIBRATION_SIZES=(32 64 128 256 512)

run_experiment() {
    local calibration_size=$1
    
    local EXP_DIR="$BASE_DIR/calib${calibration_size}"
    mkdir -p $EXP_DIR

    echo "Running experiment with calibration size: $calibration_size"
    echo "Saving to directory: $EXP_DIR"
    
    echo "Starting training..."
    start_time=$(date +%s)
    
    python bitdelta/train.py \
        --base_model "meta-llama/Llama-2-7b-hf" \
        --finetuned_model "meta-llama/Llama-2-7b-chat-hf" \
        --num_calibration_samples $calibration_size \
        --save_dir $EXP_DIR \
        --batch_size 4 \
        --num_steps 250 \
        --base_model_device 0 \
        --finetuned_model_device 0 \
        --finetuned_compressed_model_device 1 \
        --base_model_memory_map "$BASE_MODEL_MAP" \
        --finetuned_model_memory_map "$FINETUNED_MODEL_MAP" \
        --finetuned_compressed_model_memory_map "$COMPRESSED_MODEL_MAP" \
        --debug True \
        --use_activation_aware \

    end_time=$(date +%s)
    total_time=$(($end_time - $start_time))
    echo "Total time: $total_time seconds"
    echo "$total_time" > "$EXP_DIR/total_time.txt"

    echo "Starting evaluation..."
    python bitdelta/eval_ppl.py \
        --base_model "meta-llama/Llama-2-7b-hf" \
        --dataset_name "atrost/financial_phrasebank" \
        --subset "default" \
        --split "train" \
        --save_dir $EXP_DIR \
        --num_eval_samples 300 \
        --model_diff $EXP_DIR/diff.pt
}

collect_results() {
    export BASE_DIR="$BASE_DIR"
    
    python << 'EOF'
import pandas as pd
import json
import os

BASE_DIR = os.environ['BASE_DIR']
calibration_sizes = [32, 64, 128, 256, 512]

results = []
for size in calibration_sizes:
    exp_dir = f"{BASE_DIR}/calib{size}"
    
    try:
        with open(os.path.join(exp_dir, "ppl.txt")) as f:
            ppl = float(f.read().strip())
    except:
        ppl = None

    try:
        with open(os.path.join(exp_dir, "total_time.txt")) as f:
            total_time = float(f.read().strip())
    except:
        total_time = None

    results.append({
        'dataset': "financial_phrasebank",
        'calibration_size': size,
        'perplexity': ppl,
        'total_time': total_time,
    })

df = pd.DataFrame(results)
print("\nResults:")
print(df)
df.to_csv(f"{BASE_DIR}/finance_results.csv", index=False)
EOF
}

# Run experiments for each calibration size
for size in "${CALIBRATION_SIZES[@]}"; do
    echo "Running experiment for calibration size $size"
    run_experiment $size
done

collect_results

echo "All experiments complete! Results saved to $BASE_DIR/finance_results.csv"