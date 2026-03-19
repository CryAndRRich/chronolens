from typing import Any, Dict, Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim

from model import get_model
from preprocess import DataManager


def update_model_kwargs(data: DataManager,
                        model_kwargs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if "vocab_size" in model_kwargs.keys():
        model_kwargs["vocab_size"] = data.VOCAB_SIZE

    if "seq_length" in model_kwargs.keys():
        model_kwargs["seq_length"] = data.SEQ_LENGTH

    if "embedding_dim" in model_kwargs.keys():
        model_kwargs["embedding_dim"] = data.EMBEDDING_DIM

    return model_kwargs
    

def get_model_optim_schedule(model_name: str,
                             data_manager: DataManager, 
                             model_kwargs: Dict[str, Any], 
                             optim_kwargs: Dict[str, Any],
                             scheduler_kwargs: Dict[str, Any],
                             device: torch.device) -> Tuple[torch.nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Chuẩn bị mô hình, bộ tối ưu hóa và bộ điều chỉnh tốc độ học
    """
    model_kwargs = update_model_kwargs(data_manager, model_kwargs)
    model = get_model(
        name=model_name,
        **model_kwargs  
    ).to(device)
    
    optimizer = optim.AdamW(
        params=model.parameters(), 
        **optim_kwargs
    ) 
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        **scheduler_kwargs
    )
    
    return (model, optimizer, scheduler)