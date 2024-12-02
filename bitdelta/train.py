import os
import time
import psutil
import json
import torch
import torch.nn.functional as F
from bitdelta.diff import compress_diff, save_diff, save_full_model
from bitdelta.misc import find_corr_stddev

from bitdelta.utils import get_model, parse_args, get_tokenizer
from bitdelta.activation_utils import collect_activation_stats
from tqdm import tqdm
from bitdelta.data import get_dataset, get_dataloader

import json

args = parse_args()

# create save_dir if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

tokenizer = get_tokenizer(args.base_model)

with torch.no_grad():
    base_model = get_model(args.base_model, args.base_model_device, args.base_model_memory_map)
    finetuned_model = get_model(args.finetuned_model, args.finetuned_model_device, args.finetuned_model_memory_map)
    
start_time = time.time()
process = psutil.Process()
initial_memory = process.memory_info().rss / (1024 ** 2)

# get corr/stddev stats
# if args.debug:
#     print(f"finding corr/stddev stats...")
#     corrs, stddevs = find_corr_stddev(base_model, finetuned_model)
#     corr = sum(corrs) / len(corrs)
#     stddev = sum(stddevs) / len(stddevs)
#     # save in args.save_dir as csv
#     with open(os.path.join(args.save_dir, "corr_stddev.csv"), "w") as f:
#         f.write(f"corr,stddev\n{corr},{stddev}")

# Get activation stats
activation_stats = None
if args.use_activation_aware:
    calibration_batch_size = 1
    
    calibration_dataset = get_dataset(
        args.dataset_name,
        args.subset,
        "train",
        size=args.num_calibration_samples,
    )
    calibration_dataloader = get_dataloader(
        calibration_dataset,
        tokenizer,
        batch_size=calibration_batch_size,
        num_workers=4,
        max_length=args.max_length,
    )
    activation_stats = collect_activation_stats(
        finetuned_model,
        calibration_dataloader,
        num_samples=args.num_calibration_samples
    )

finetuned_compressed_model = get_model(args.finetuned_model, args.finetuned_compressed_model_device, args.finetuned_compressed_model_memory_map)

print(f"compressing diff...")
compress_diff(
    base_model,
    finetuned_model,
    finetuned_compressed_model,
    activation_stats=activation_stats if args.use_activation_aware else None
)


# save untrained delta
save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff_untrained.pt"))

if not args.use_activation_aware:
    train_num_samples = args.batch_size * args.num_steps
    train_dataset = get_dataset(
        args.dataset_name,
        args.subset,
        "train",
        size=train_num_samples,
    )
    train_dataloader = get_dataloader(
        train_dataset,
        tokenizer,
        args.batch_size,
        num_workers=4,
        max_length=args.max_length,
    )

    bar = tqdm(train_dataloader)

    optimizer = torch.optim.AdamW(finetuned_compressed_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps)

    train_loss_list = []
    latency_list = []
    memory_usage_list = []

    for step, batch in enumerate(bar):
        step_start_time = time.time()

        batch1 = {k: v.to(finetuned_model.device) for k, v in batch.items()}
        with torch.inference_mode():
            finetuned_outputs = finetuned_model(**batch1)

        batch2 = {k: v.to(finetuned_compressed_model.device) for k, v in batch.items()}
        finetuned_compressed_outputs = finetuned_compressed_model(**batch2)

        loss = F.mse_loss(
            finetuned_outputs.logits.clone().to(finetuned_compressed_outputs.logits.device),
            finetuned_compressed_outputs.logits,
        )

        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        step_end_time = time.time()
        step_latency = (step_end_time - step_start_time) * 1000
        latency_list.append(step_latency)

        current_memory = process.memory_info().rss / (1024 ** 2)
        memory_usage_list.append(current_memory)

        bar.set_description(f"train loss: {loss.item()}")

    # Save latency data
    with open(os.path.join(args.save_dir, "latency.json"), "w") as f:
        json.dump(latency_list, f)

    # Save memory usage data
    with open(os.path.join(args.save_dir, "memory_usage.json"), "w") as f:
        json.dump(memory_usage_list, f)

    # save loss list
    if args.debug:
        with open(os.path.join(args.save_dir, "train_loss.json"), "w") as f:
            json.dump(train_loss_list, f)

    # save trained delta
    save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff.pt"))
else:
    print("Skipping training/distillation step because --use_activation_aware is enabled")
    save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff.pt"))
    
end_time = time.time()
total_time = end_time - start_time
final_memory = process.memory_info().rss / (1024 ** 2)

print(f"Total time for {'activation-aware' if args.use_activation_aware else 'standard'} BitDelta: {total_time:.2f} seconds")
print(f"Initial memory usage: {initial_memory:.2f} MB")
print(f"Final memory usage: {final_memory:.2f} MB")

with open(os.path.join(args.save_dir, "e2e_latency_memory.txt"), "w") as f:
    f.write(f"Total time: {total_time:.2f} seconds\n")
    f.write(f"Initial memory usage: {initial_memory:.2f} MB\n")
    f.write(f"Final memory usage: {final_memory:.2f} MB\n")

del base_model, finetuned_model, finetuned_compressed_model
torch.cuda.empty_cache()

if args.save_full_model:
    print("saving uncalibrated model")
    save_full_model(args.base_model, args.finetuned_model, os.path.join(args.save_dir, "diff_untrained.pt"), os.path.join(args.save_dir, "uncalibrated_model"), device="cpu")
    print("saving calibrated model")
    save_full_model(args.base_model, args.finetuned_model, os.path.join(args.save_dir, "diff.pt"), os.path.join(args.save_dir, "calibrated_model"), device="cpu")
