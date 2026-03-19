from typing import List, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def extract_deep_features(model: nn.Module, 
                          dataloader: DataLoader, 
                          device: torch.device) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    model.eval()
    extracted_embeddings = []
    
    def hook_fn(module, input, output):
        extracted_embeddings.append(output.detach().cpu().numpy())
        
    handle = model.attn_pool.register_forward_hook(hook_fn)
    
    preds = []
    y_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                y_true.append(y.cpu().numpy())
            else:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                
            x = x.to(device)
            
            outputs = model(x) 
            
            preds_batch = torch.stack(outputs, dim=1).cpu().numpy()
            preds.append(preds_batch)
            
    handle.remove()
    
    embeddings = np.concatenate(extracted_embeddings, axis=0)
    preds = np.concatenate(preds, axis=0)
    
    x_embed = np.concatenate([embeddings, preds], axis=1) 
    
    if len(y_true) > 0:
        y_true = np.concatenate(y_true, axis=0)
        return x_embed, y_true
    else:
        return x_embed
    

class SequenceFeatureExtractor:
    def __init__(self, 
                 feature_cols: List[str],
                 top_k_vocab: int = 50) -> None:
        self.vectorizer = CountVectorizer(max_features=top_k_vocab, token_pattern=r"(?u)\b\w+\b")
        self.feature_cols = feature_cols
        self.is_fitted = False

    def extract(self, x_df: np.ndarray) -> np.ndarray:
        seqs = x_df[self.feature_cols].values 
        
        seq_lengths = np.sum(seqs != 0, axis=1)
        consecutive_repeats = np.sum((seqs[:, 1:] == seqs[:, :-1]) & (seqs[:, 1:] != 0), axis=1)
        padding_ratio = (len(self.feature_cols) - seq_lengths) / len(self.feature_cols)
        
        num_uniques = np.zeros(len(seqs), dtype=np.float32)
        diversity = np.zeros(len(seqs), dtype=np.float32)
        first_codes = np.zeros(len(seqs), dtype=np.float32)
        last_codes = np.zeros(len(seqs), dtype=np.float32)
        
        most_frequent_codes = np.zeros(len(seqs), dtype=np.float32)
        obsession_ratios = np.zeros(len(seqs), dtype=np.float32)
        center_of_mass = np.zeros(len(seqs), dtype=np.float32)
        
        seq_strings = [] 
        for i, row in enumerate(seqs):
            non_zeros = row[row != 0]
            if len(non_zeros) > 0:
                vals, counts = np.unique(non_zeros, return_counts=True)
                num_uniques[i] = len(vals)
                diversity[i] = num_uniques[i] / len(non_zeros)
                first_codes[i] = non_zeros[0]
                last_codes[i] = non_zeros[-1] 
                
                best_idx = np.argmax(counts)
                most_frequent_codes[i] = vals[best_idx]
                obsession_ratios[i] = counts[best_idx] / len(non_zeros)
                
                center_of_mass[i] = np.mean(np.arange(len(non_zeros))) / len(non_zeros)
                
                seq_strings.append(" ".join(non_zeros.astype(str)))
            else:
                seq_strings.append("")

        if not self.is_fitted:
            bow_features = self.vectorizer.fit_transform(seq_strings).toarray()
            self.is_fitted = True
        else:
            bow_features = self.vectorizer.transform(seq_strings).toarray()
        
        manual_features = np.column_stack([
            seq_lengths,
            num_uniques,
            diversity,
            first_codes,
            last_codes,
            consecutive_repeats,
            padding_ratio,
            most_frequent_codes,
            obsession_ratios,
            center_of_mass,
            bow_features  
        ])
        
        return manual_features.astype(np.float32)