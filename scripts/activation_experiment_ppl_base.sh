#!/bin/bash

BASE_DIR="experiments/llama7b_standard_domains"
mkdir -p $BASE_DIR

BASE_MODEL_MAP="{\"0\": \"70GiB\"}"
FINETUNED_MODEL_MAP="{\"0\": \"70GiB\"}"
COMPRESSED_MODEL_MAP="{\"1\": \"70GiB\"}"

DOMAINS=(
    "FremyCompany/AGCT-Dataset"
    "bigcode/the-stack"
    "atrost/financial_phrasebank"
    "wikitext"
)
SUBSETS=(
    "default"
    "default"
    "default"
    "wikitext-2-raw-v1"
)

NUM_EVAL_SAMPLES=(
    100
    10
    300
    100
)

run_experiment() {
    local domain=$1
    local subset=$2
    local eval_samples=$3

    local EXP_DIR="$BASE_DIR/${domain//\//_}"
    mkdir -p $EXP_DIR

    echo "Running experiment for domain: $domain"
    echo "Saving to directory: $EXP_DIR"
    
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
        --debug True

    end_time=$(date +%s)
    total_time=$(($end_time - $start_time))
    echo "Total time: $total_time seconds"
    echo "$total_time" > "$EXP_DIR/total_time.txt"

    echo "Starting evaluation..."
    python bitdelta/eval_ppl.py \
        --base_model "meta-llama/Llama-2-7b-hf" \
        --dataset_name $domain \
        --subset $subset \
        --split "train" \
        --save_dir $EXP_DIR \
        --num_eval_samples $eval_samples \
        --model_diff $EXP_DIR/diff.pt
}

collect_results() {
    export BASE_DIR="$BASE_DIR"
    
    python << 'EOF'
import pandas as pd
import json
import os

BASE_DIR = os.environ['BASE_DIR']

domains = [
    "curaihealth_medical_questions_pairs",
    "bigcode_the_stack",
    "atrost_financial_phrasebank",
    "wikitext"
]

results = []
for domain in domains:
    exp_dir = f"{BASE_DIR}/{domain}"
    
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
        'domain': domain,
        'perplexity': ppl,
        'total_time': total_time,
    })

df = pd.DataFrame(results)
print("\nResults:")
print(df)
df.to_csv(f"{BASE_DIR}/results.csv", index=False)
EOF
}

# Run experiments for each domain
for i in "${!DOMAINS[@]}"; do
    run_experiment "${DOMAINS[$i]}" "${SUBSETS[$i]}" "${NUM_EVAL_SAMPLES[$i]}"
done

collect_results

echo "All experiments complete! Results saved to $BASE_DIR/results.csv"
