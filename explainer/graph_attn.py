from typing import Dict, Tuple
from collections import defaultdict
import torch

def extract_graph_edges(model: torch.nn.Module, 
                        dataloader: torch.utils.data.DataLoader, 
                        id_to_idx: Dict[int, int], 
                        device: torch.device,
                        layer_idx: int = 0) -> Dict[Tuple[int, int], float]:
    """
    Trích xuất trọng số cạnh từ GatedDirectedGAT để tìm ra mối quan hệ nhân quả
    """
    target_layer = model.layers[layer_idx].gnn.gat_past.dropout
    target_layer._forward_hooks.clear()
    
    captured_data = {}
    def hook_fn(module, input, output):
        captured_data["alpha"] = input[0].detach()

    hook_handle = target_layer.register_forward_hook(hook_fn)
    idx_to_id = {idx: original_id for original_id, idx in id_to_idx.items()}
    
    edge_weight_sum = defaultdict(float)
    edge_count = defaultdict(int)

    model.eval()
    
    try:
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                
                _ = model(batch_x)
                alpha = captured_data["alpha"] 
                
                alpha_mean = alpha.mean(dim=1).cpu().numpy()
                batch_actions = batch_x.cpu().numpy()
                
                for b in range(len(batch_actions)):
                    seq = batch_actions[b]
                    seq_len = len(seq)
                    
                    for i in range(seq_len):    
                        for j in range(seq_len): 
                            if seq[i] != 0 and seq[j] != 0 and i != j:
                                if seq[i] in idx_to_id and seq[j] in idx_to_id:
                                    src_id = idx_to_id[seq[j]]
                                    tgt_id = idx_to_id[seq[i]]
                                    
                                    weight = alpha_mean[b, i, j]
                                    if weight > 0:
                                        edge_weight_sum[(src_id, tgt_id)] += weight
                                        edge_count[(src_id, tgt_id)] += 1
    finally:
        hook_handle.remove()
        
    avg_edge_dict = {}
    for edge_pair in edge_weight_sum:
        avg_weight = (edge_weight_sum[edge_pair] / edge_count[edge_pair]) * 100
        avg_edge_dict[edge_pair] = avg_weight
        
    return dict(sorted(avg_edge_dict.items(), key=lambda item: item[1], reverse=True))