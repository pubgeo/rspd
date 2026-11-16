"""Explicit, process-wide deterministic settings for evaluation entrypoints."""

import os
import random

import numpy as np
import torch


def configure_reproducibility(seed: int = 2334) -> dict:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"seed": seed, "deterministic_algorithms": True, "tf32": False, "torch_version": str(torch.__version__)}
