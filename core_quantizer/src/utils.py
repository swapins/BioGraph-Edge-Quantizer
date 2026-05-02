import torch
import numpy as np
import random

def seed_everything(seed=42):
    """Enforces deterministic behavior across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Critical for research-grade reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Only available in newer torch versions for strict determinism
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass