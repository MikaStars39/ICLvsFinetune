import torch

def get_precision(
    precision: str,
):
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "fp32":
        return torch.float32
    else:
        raise ValueError("Not a valid data type")

