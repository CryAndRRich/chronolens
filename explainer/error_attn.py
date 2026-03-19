from typing import Dict, Tuple
from collections import defaultdict
import torch
import torch.nn.functional as F

def extract_error_attention(model: torch.nn.Module, 
                            dataloader: torch.utils.data.DataLoader, 
                            target_task_idx: int,
                            id_to_idx: Dict[int, int], 
                            device: torch.device) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Phân tách và so sánh Attention giữa nhóm đoán đúng và nhóm đoán sai
    """
    model.attn_pool.attn._forward_hooks.clear()
    captured_data = {}

    def hook_fn(module, input, output):
        captured_data["raw_scores"] = output.detach()

    hook_handle = model.attn_pool.attn.register_forward_hook(hook_fn)
    idx_to_id = {idx: original_id for original_id, idx in id_to_idx.items()}
    
    correct_attn_sum = defaultdict(float)
    correct_count = defaultdict(int)
    wrong_attn_sum = defaultdict(float)
    wrong_count = defaultdict(int)

    model.eval()
    try:
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                mask = (batch_x != 0)
                
                outputs = model(batch_x)
                preds = outputs[target_task_idx] 
                labels = batch_y[:, target_task_idx]
                
                is_correct = (torch.abs(preds - labels) <= 5.0).cpu().numpy()
                
                raw_scores = captured_data["raw_scores"] 
                fill_value = torch.finfo(raw_scores.dtype).min
                masked_scores = raw_scores.masked_fill(~mask.unsqueeze(-1), fill_value)
                attn_weights = F.softmax(masked_scores, dim=1).squeeze(-1) 
                
                batch_actions = batch_x.cpu().numpy()
                batch_weights = attn_weights.cpu().numpy()
                
                for b_idx in range(len(batch_actions)):
                    user_actions = batch_actions[b_idx]
                    user_weights = batch_weights[b_idx]
                    correct_flag = is_correct[b_idx]
                    
                    for action_idx, weight in zip(user_actions, user_weights):
                        if action_idx != 0 and action_idx in idx_to_id: 
                            original_id = idx_to_id[action_idx]
                            
                            if correct_flag:
                                correct_attn_sum[original_id] += weight
                                correct_count[original_id] += 1
                            else:
                                wrong_attn_sum[original_id] += weight
                                wrong_count[original_id] += 1
    finally:
        hook_handle.remove()
        
    avg_correct = {orig_id: (correct_attn_sum[orig_id] / correct_count[orig_id]) * 100 for orig_id in correct_attn_sum}
    avg_wrong = {orig_id: (wrong_attn_sum[orig_id] / wrong_count[orig_id]) * 100 for orig_id in wrong_attn_sum}
    
    return avg_correct, avg_wrong