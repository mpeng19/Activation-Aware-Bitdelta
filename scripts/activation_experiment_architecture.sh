#!/bin/bash

BASE_DIR="experiments/llm_architecture_comparison"
mkdir -p $BASE_DIR

BASE_MODEL_MAP="{\"0\": \"40GiB\"}"
FINETUNED_MODEL_MAP="{\"0\": \"40GiB\"}"
COMPRESSED_MODEL_MAP="{\"1\": \"40GiB\"}"

LLM_PAIRS=(
    #"meta-llama/Llama-2-7b-hf,meta-llama/Llama-2-7b-chat-hf" -- complete
    #"meta-llama/Llama-2-13b-hf,meta-llama/Llama-2-13b-chat-hf" -- OOM
    #"meta-llama/Llama-2-70b-hf,meta-llama/Llama-2-70b-chat-hf" -- OOM
    #"meta-llama/Llama-2-7b-hf,lmsys/vicuna-7b-v1.5" -- complete
    "meta-llama/Llama-2-13b-hf,lmsys/vicuna-13b-v1.3" -- OOM
    "meta-llama/Llama-2-13b-hf,WizardLMTeam/WizardLM-13B-V1.2" -- OOM
    #"mistralai/Mistral-7B-v0.1,mistralai/Mistral-7B-Instruct-v0.1" -- complete
    #"mistralai/Mistral-7B-v0.1,HuggingFaceH4/zephyr-7b-beta" -- complete
)

run_experiment() {
    local exp_name=$1
    local base_model=$2
    local finetuned_model=$3
    local activation_aware=$4
    local batch_size=4

    echo "Running experiment: ${exp_name}_${activation_aware}"
    
    local aa_flags=()
    if [ "$activation_aware" = "activation_aware" ]; then
        aa_flags=(--use_activation_aware --num_calibration_samples 256)
    fi

    EXP_DIR="$BASE_DIR/${exp_name}_${activation_aware}"
    mkdir -p $EXP_DIR

    echo "Starting training..."
    start_time=$(date +%s)

    python bitdelta/train.py \
        --base_model "$base_model" \
        --finetuned_model "$finetuned_model" \
        --save_dir $EXP_DIR \
        --batch_size $batch_size \
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
    echo "Total time for ${exp_name}_${activation_aware}: $total_time seconds"
    echo "$total_time" > "$EXP_DIR/total_time.txt"

    echo "Starting evaluation..."
    python bitdelta/eval_ppl.py \
        --base_model "$base_model" \
        --dataset_name wikitext \
        --subset wikitext-2-raw-v1 \
        --save_dir $EXP_DIR \
        --num_eval_samples 100 \
        --model_diff $EXP_DIR/diff.pt
}

# Run experiments for each LLM pair
for llm_pair in "${LLM_PAIRS[@]}"; do
    IFS=',' read -r base_model finetuned_model <<< "$llm_pair"
    exp_name="$(basename $base_model)_$(basename $finetuned_model)"

    # Run Standard Experiment
    echo "Running standard experiment for $exp_name..."
    run_experiment "$exp_name" "$base_model" "$finetuned_model" "standard"

    # Run Activation-Aware Experiment
    echo "Running activation-aware experiment for $exp_name..."
    run_experiment "$exp_name" "$base_model" "$finetuned_model" "activation_aware"
done

# Collect results
export BASE_DIR="$BASE_DIR"

python << 'EOF'
import pandas as pd
import json
import os

BASE_DIR = os.environ['BASE_DIR']

results = []
for dir_name in os.listdir(BASE_DIR):
    exp_dir = os.path.join(BASE_DIR, dir_name)
    if not os.path.isdir(exp_dir):
        continue

    parts = dir_name.split("_")
    exp_name = "_".join(parts[:-1])
    activation_aware = parts[-1]

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
        'experiment': exp_name,
        'activation_aware': activation_aware,
        'perplexity': ppl,
        'total_time': total_time,
    })

df = pd.DataFrame(results)
print("\nResults:")
print(df)
df.to_csv(f"{BASE_DIR}/results.csv", index=False)
EOF

echo "Experiments complete! Results saved to $BASE_DIR/results.csv"
