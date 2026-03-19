from typing import List, Dict, Any
import torch


class CONFIG_MODEL:
    RANDOM_SEED: int = 42
    DEVICE: torch.device = torch.device("cuda")

    NUM_EPOCHS: int = 100
    EARLY_STOPPING: int = 15

    START_THRESHOLD: float = 0.01
    END_THRESHOLD: float = 0.05

    START_WEIGHTS: List[float] = [100.0, 100.0, 100.0, 1.0, 100.0, 100.0, 100.0, 1.0]
    END_WEIGHTS: List[float] = [0.0, 1.0, 1.0, 100.0, 0.0, 1.0, 1.0, 100.0]

    TOTAL_EPOCHS: int = 10

    M_LIST_LOSS: List[float] = [1.0, 12.0, 31.0, 99.0, 1.0, 12.0, 31.0, 99.0]
    M_LIST_METRIC: List[float] = [12.0, 31.0, 99.0, 12.0, 31.0, 99.0]
    W_LIST_VALS: List[float] = [1.0, 1.0, 100.0, 1.0, 1.0, 100.0]

    MODEL_KWARGS: Dict[str, Dict[str, Any]] = {
        "chrono_r": {
            "vocab_size": None, 
            "seq_length": None, 
            "embedding_dim": None, 
            "num_layers": 2, 
            "dropout_rate": 0.3
        },
        "chrono_c": {
            "vocab_size": None, 
            "seq_length": None, 
            "embedding_dim": None, 
            "kernel_sizes": [3, 5, 7], 
            "expansion_factor": 2, 
            "dropout_rate": 0.3
        },
        "chrono_g": {
            "vocab_size": None, 
            "seq_length": None, 
            "embedding_dim": None,
            "dilations": [1, 2, 4, 8], 
            "window": 3,
            "heads": 4, 
            "dropout_rate": 0.3
        }
    }
    
    OTIMIZER_KWARGS: Dict[str, Any] = {
        "lr": 1e-3,
        "weight_decay": 1e-2
    }

    SCHEDULER_KWARGS: Dict[str, Any] = {
        "max_lr": 1e-3,
        "pct_start": 0.2,
        "anneal_strategy": "cos",
        "final_div_factor": 10000.0,
    }