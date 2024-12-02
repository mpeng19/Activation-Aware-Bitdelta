import torch
import torch.nn as nn
import gc

from bitdelta.binary_gemm_kernel import pack, unpack, binary_bmm
from bitdelta.utils import get_model, get_tokenizer
from bitdelta.activation_utils import compute_optimal_scale

class BinaryDiff(nn.Module):
    def __init__(self, base, finetune):
        super().__init__()
        diff = finetune - base
        quantile = diff.float().abs().mean()

        mask = torch.ones_like(diff)
        mask[diff < 0] = 0
        mask = pack(mask.bool().T)

        self.register_buffer("mask", mask)
        self.register_buffer("base", base.T)
        self.register_parameter(
            "coeff",
            nn.Parameter(
                torch.tensor(
                    quantile,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=base.device,
                )
            ),
        )
        del base, finetune, diff

    def forward(self, x):
        # print(x.shape, self.base.shape, self.coeff.shape, self.mask.shape)
        # [B, seq, in] @ [in, out] + [B, seq, in] @ [B, in/32, out]

        # TODO: This can be faster
        repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        return x @ self.base + self.coeff * binary_bmm(x, repeated_mask)
    
class ActivationAwareBinaryDiff(nn.Module):
    def __init__(self, base, finetune, activation_variance):
        super().__init__()
        diff = finetune - base
        
        mask = torch.ones_like(diff)
        mask[diff < 0] = 0
        mask = pack(mask.bool().T)
        
        scale = compute_optimal_scale(diff, activation_variance)

        self.register_buffer("mask", mask)
        self.register_buffer("base", base.T)
        self.register_buffer(
            "coeff",
            scale.to(dtype=torch.float32, device=base.device)
        )



def compress_diff(base_model, finetuned_model, finetuned_compressed_model, activation_stats=None):
    def compress_submodule(name, subname, module, submodule):
        target_device = submodule.weight.device
                        
        base_weight = base_model.get_submodule(f"{name}.{subname}").weight.detach().to(target_device)
        finetuned_weight = finetuned_model.get_submodule(f"{name}.{subname}").weight.detach().to(target_device)

        layer_name = f"{name}.{subname}"
        if activation_stats is not None and layer_name in activation_stats:
            #print(f"Using activation-aware scaling for {layer_name}")
            activation_variance = activation_stats[layer_name].to(target_device)
            compressed = ActivationAwareBinaryDiff(
                base=base_weight,
                finetune=finetuned_weight,
                activation_variance=activation_variance,
            ).to(target_device)
        else:
            #print(f"Using standard scaling for {layer_name}")
            compressed = BinaryDiff(
                base=base_weight,
                finetune=finetuned_weight,
            ).to(target_device)

        del submodule, base_weight
        setattr(module, subname, None)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(module, subname, compressed)

    for name, module in finetuned_compressed_model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    compress_submodule(name, subname, module, submodule)



def save_diff(finetuned_compressed_model, save_dir):
    diff_dict = {}

    for name, module in finetuned_compressed_model.named_modules():
        if isinstance(module, BinaryDiff) or isinstance(module, ActivationAwareBinaryDiff):
            # diff_dict[name + ".mask"] = (module.mask == 1).bool().cpu()
            diff_dict[name + ".mask"] = module.mask.cpu()
            diff_dict[name + ".coeff"] = module.coeff.cpu()

    for name, param in finetuned_compressed_model.named_parameters():
        if param.requires_grad:
            diff_dict[name] = param.cpu()

    torch.save(diff_dict, save_dir)

@torch.no_grad()
def load_diff(model: nn.Module, diff_dir: str):
    diff_dict = torch.load(diff_dir, map_location='cpu')

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            device = module.weight.device
        else:
            continue
        if name + ".mask" in diff_dict:
            coeff = diff_dict[name + ".coeff"].to(device)
            mask = diff_dict[name + ".mask"].to(device)

            weight = (unpack(mask).to(device) * 2 - 1) * coeff
            weight = weight.T.to(device=device, dtype=module.weight.dtype)
            module.weight.add_(weight)
        elif name + ".weight" in diff_dict:
            weight = diff_dict[name + ".weight"].to(device=device, dtype=module.weight.dtype)
            module.weight = nn.Parameter(weight)
        elif name + '.A' in diff_dict:
            A = diff_dict[name + '.A'].to(device)
            B = diff_dict[name + '.B'].to(device)
            mask = (A @ B).T.to(device=device, dtype=module.weight.dtype)
            module.weight.add_(mask)

    model.config.vocab_size = model.lm_head.weight.size(0)


def save_full_model(base_model_name, finetuned_model_name, diff_dir, save_dir, device):
    base_model = get_model(base_model_name, device)
    tokenizer = get_tokenizer(finetuned_model_name)
    load_diff(base_model, diff_dir)

    base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del base_model