import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm

import psutil
process = psutil.Process()

def collect_activation_stats(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 128
) -> Dict[str, torch.Tensor]:
    activation_stats = {}
    hooks = []
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            acts_list = activations.setdefault(name, [])
            if sum([act.shape[0] for act in acts_list]) >= num_samples:
                return
            acts = input[0].detach().cpu().view(-1, input[0].size(-1))
            acts_list.append(acts)
        return hook
    
    for name, module in model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    full_name = f"{name}.{subname}"
                    if not full_name.startswith('model.'):
                        full_name = f"model.{full_name}"
                    hook_handle = submodule.register_forward_hook(hook_fn(full_name))
                    hooks.append(hook_handle)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activation statistics"):
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            model(**batch)
            if all(sum([act.shape[0] for act in acts_list]) >= num_samples for acts_list in activations.values()):
                break

    for name, acts_list in activations.items():
        acts = torch.cat(acts_list, dim=0)[:num_samples]  # (num_samples, hidden_size)
        cov_matrix = torch.cov(acts.T)
        activation_stats[name] = cov_matrix

    for hook in hooks:
        hook.remove()
    
    return activation_stats


def compute_optimal_scale(delta: torch.Tensor, activation_covariance: torch.Tensor) -> torch.Tensor:
    """
    Compute optimal scale factor using activation variances.

    Args:
        delta: Weight difference matrix (out_features, in_features)
        activation_variance: Activation variances (in_features,)

    Returns:
        Optimal scale factor (scalar)
    """
    sign_delta = torch.sign(delta)
    numerator = torch.trace(delta @ activation_covariance @ sign_delta.T)
    denominator = torch.trace(sign_delta @ activation_covariance @ sign_delta.T)
    scale = numerator / denominator
    scale = torch.abs(scale)
    return scale

