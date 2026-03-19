from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling1D(nn.Module):
    def __init__(self, in_features: int) -> None:
        """
        Khởi tạo lớp pooling sử dụng cơ chế attention để tổng hợp thông tin 
        từ các bước thời gian khác nhau trong một chuỗi đầu vào
        """
        super(AttentionPooling1D, self).__init__()
        self.attn = nn.Linear(in_features=in_features,
                              out_features=1)

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.attn(x) 
        if mask is not None:
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask.unsqueeze(-1), fill_value)

        attn_weights = F.softmax(scores, dim=1)
        weighted_sum = torch.sum(x * attn_weights, dim=1)
        return weighted_sum
    

class GCEFusion(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 dropout: float = 0.2) -> None:
        """
        Lớp GCEFusion thực hiện kết hợp thông tin giữa đặc trưng cục bộ (local features) và 
        đặc trưng toàn cục (global features)
        """
        super(GCEFusion, self).__init__()
        self.global_memory = nn.Embedding(num_embeddings=vocab_size, 
                                          embedding_dim=d_model, 
                                          padding_idx=0)
        
        self.fusion_gate = nn.Linear(in_features=d_model * 2, 
                                     out_features=d_model)
        
        self.transform = nn.Linear(in_features=d_model, 
                                   out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(normalized_shape=d_model)
        
        nn.init.xavier_normal_(self.fusion_gate.weight)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, 
                local_features: torch.Tensor, 
                input_ids: torch.Tensor) -> torch.Tensor:
        global_features = self.global_memory(input_ids) 
        
        concat_features = torch.cat([local_features, global_features], dim=-1)
        gate_score = torch.sigmoid(self.fusion_gate(concat_features)) 
        
        fused_features = gate_score * local_features + (1.0 - gate_score) * global_features
        
        out = self.dropout(F.gelu(self.transform(fused_features)))
        return self.norm(local_features + out) 
    

class CascadeRegressionHead(nn.Module):
    def __init__(self, 
                 d_model: int) -> None:
        """
        Lớp CascadeRegressionHead thực hiện dự đoán các giá trị liên quan đến thời gian và công suất
        """
        super(CascadeRegressionHead, self).__init__()
        self.head_start = nn.Sequential(
            nn.Linear(in_features=d_model, 
                      out_features=64), 
            nn.GELU(), 
            nn.Linear(in_features=64, 
                      out_features=3)
        )
        
        self.head_factory = nn.Sequential(
            nn.Linear(in_features=d_model, 
                      out_features=128), 
            nn.LayerNorm(normalized_shape=128), 
            nn.GELU(), 
            nn.Dropout(p=0.2), 
            nn.Linear(in_features=128,
                      out_features=64), 
            nn.GELU(), 
            nn.Linear(in_features=64, 
                      out_features=2)
        )
        
        self.start_proj = nn.Sequential(
            nn.Linear(in_features=3, 
                      out_features=16),
            nn.GELU(),
            nn.Linear(in_features=16, 
                      out_features=32)
        )
        
        self.head_end = nn.Sequential(
            nn.Linear(in_features=d_model + 32, 
                      out_features=64), 
            nn.GELU(), 
            nn.Linear(in_features=64, 
                      out_features=3)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        start_preds = self.head_start(x)  
        
        factory_preds = self.head_factory(x) 
        
        start_context = self.start_proj(start_preds)
        end_in = torch.cat([x, start_context], dim=-1)
        end_preds = self.head_end(end_in)    
        
        s_year = torch.sigmoid(start_preds[:, 0])
        e_year = torch.sigmoid(end_preds[:, 0])
        
        s_month = 1.0 + 11.0 * torch.sigmoid(start_preds[:, 1])
        e_month = 1.0 + 11.0 * torch.sigmoid(end_preds[:, 1])
        
        s_day = 1.0 + 30.0 * torch.sigmoid(start_preds[:, 2])
        e_day = 1.0 + 30.0 * torch.sigmoid(end_preds[:, 2])
        
        s_power = 99.0 * torch.sigmoid(factory_preds[:, 0])
        e_power = 99.0 * torch.sigmoid(factory_preds[:, 1])
        
        return [
            s_year, s_month, s_day, s_power,
            e_year, e_month, e_day, e_power
        ]