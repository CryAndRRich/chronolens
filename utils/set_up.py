import os
import random
import warnings

import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Cài đặt seed ngẫu nhiên để đảm bảo tính tái lập
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    
    print(f"Random seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """
    Cài đặt seed cho mỗi worker trong torch.DataLoader để đảm bảo tính tái lập
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed) 
    torch.cuda.manual_seed(worker_seed)