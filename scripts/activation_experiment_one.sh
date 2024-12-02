#!/bin/bash

BASE_DIR="experiments/llama7b"
mkdir -p $BASE_DIR

BASE_MODEL_MAP="{\"0\": \"70GiB\"}"
FINETUNED_MODEL_MAP="{\"0\": \"70GiB\"}"
COMPRESSED_MODEL_MAP="{\"1\": \"70GiB\"}"

run_experiment() {
    local exp_name=$1
    local activation_aware=$2

    echo "Running experiment: $exp_name"
    
    local aa_flags=()
    if [ "$activation_aware" = true ]; then
        aa_flags=(--use_activation_aware --num_calibration_samples 256)
    fi

    EXP_DIR="$BASE_DIR/$exp_name"
    mkdir -p $EXP_DIR

    echo "Starting training..."
    start_time=$(date +%s)

    python bitdelta/train.py \
        --base_model "meta-llama/Llama-2-7b-hf" \
        --finetuned_model "meta-llama/Llama-2-7b-chat-hf" \
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
        "${aa_flags[@]}"

    end_time=$(date +%s)
    total_time=$(($end_time - $start_time))
    echo "Total time for $exp_name: $total_time seconds"
    echo "$total_time" > "$EXP_DIR/total_time.txt"

    echo "Starting evaluation..."
    python bitdelta/eval_ppl.py \
        --base_model "meta-llama/Llama-2-7b-hf" \
        --dataset_name wikitext \
        --subset wikitext-2-raw-v1 \
        --save_dir $EXP_DIR \
        --num_eval_samples 100 \
        --model_diff $EXP_DIR/diff.pt
}

# Run standard BitDelta
#echo "Running standard BitDelta..."
#run_experiment "standard" false

# Run activation-aware BitDelta
echo "Running activation-aware BitDelta..."
run_experiment "activation_aware" true

# Collect results
export BASE_DIR="$BASE_DIR"

python << 'EOF'
import pandas as pd
import json
import os

BASE_DIR = os.environ['BASE_DIR']

results = []
for exp_name in ["standard", "activation_aware"]:
    exp_dir = f"{BASE_DIR}/{exp_name}"
    
    try:
        with open(os.path.join(exp_dir, "ppl.txt")) as f:
            ppl = float(f.read().strip())
    except:
        ppl = None

    try:
        with open(os.path.join(exp_dir, "train_loss.json")) as f:
            train_losses = json.load(f)
            final_loss = train_losses[-1]
            avg_loss = sum(train_losses) / len(train_losses)
    except:
        final_loss = None
        avg_loss = None

    try:
        with open(os.path.join(exp_dir, "e2e_latency_memory.txt")) as f:
            lines = f.readlines()
            total_time_line = next((line for line in lines if "Total time" in line), None)
            if total_time_line:
                total_time = float(total_time_line.strip().split(":")[1].strip().split()[0])
            else:
                total_time = None

            initial_memory_line = next((line for line in lines if "Initial memory usage" in line), None)
            if initial_memory_line:
                initial_memory = float(initial_memory_line.strip().split(":")[1].strip().split()[0])
            else:
                initial_memory = None

            final_memory_line = next((line for line in lines if "Final memory usage" in line), None)
            if final_memory_line:
                final_memory = float(final_memory_line.strip().split(":")[1].strip().split()[0])
            else:
                final_memory = None
    except:
        total_time = None
        initial_memory = None
        final_memory = None

    results.append({
        'experiment': exp_name,
        'perplexity': ppl,
        'final_train_loss': final_loss,
        'avg_train_loss': avg_loss,
        'total_time': total_time,
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
    })

pd.DataFrame(results).to_csv(f"{BASE_DIR}/results.csv", index=False)
EOF

echo "Experiments complete! Results saved to $BASE_DIR/results.csv"
