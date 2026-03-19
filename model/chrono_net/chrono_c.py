import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List
import torch
import torch.nn as nn

from model.chrono_net.layers import AttentionPooling1D, CascadeRegressionHead
from model.chrono_net import register_model


class ModernTCNBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 kernel_size: int, 
                 expansion_factor: int = 2, 
                 dropout: float = 0.2) -> None:
        """
        ModernTCN block với kiến trúc depthwise separable convolution
        
        Tham số:
            d_model: Số chiều của embedding đầu vào
            kernel_size: Kích thước kernel cho convolution
            expansion_factor: Hệ số mở rộng cho hidden layer trong block
            dropout: Tỷ lệ dropout cho các lớp fully connected
        """
        super(ModernTCNBlock, self).__init__()
        
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(in_channels=d_model, 
                                out_channels=d_model, 
                                kernel_size=kernel_size, 
                                padding=padding, 
                                groups=d_model)
        
        self.norm = nn.GroupNorm(num_groups=1, 
                                 num_channels=d_model)
        
        hidden_dim = int(d_model * expansion_factor)
        self.pwconv1 = nn.Conv1d(in_channels=d_model, 
                                 out_channels=hidden_dim, 
                                 kernel_size=1)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.pwconv2 = nn.Conv1d(in_channels=hidden_dim, 
                                 out_channels=d_model, 
                                 kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.pwconv2(x)
        x = self.dropout2(x)
        return x + res


@register_model("chrono_c")
class ChronoC(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 kernel_sizes: List[int], 
                 seq_length: int = 66, 
                 embedding_dim: int = 256, 
                 expansion_factor: int = 2, 
                 dropout_rate: float = 0.3) -> None:
        """
        Khởi tạo mô hình ChronoC
        
        Tham số:
            vocab_size: Kích thước từ điển
            kernel_sizes: Danh sách kích thước kernel cho các block TCN
            seq_length: Độ dài tối đa của chuỗi đầu vào
            embedding_dim: Số chiều của vector nhúng từ điển
            expansion_factor: Hệ số mở rộng cho hidden layer trong block TCN
            dropout_rate: Tỷ lệ dropout cho các lớp fully connected và block TCN
        """
        super(ChronoC, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        self.pos_embedding = nn.Embedding(num_embeddings=seq_length, 
                                          embedding_dim=embedding_dim)
        
        blocks = []
        for ks in kernel_sizes:
            blocks.append(
                ModernTCNBlock(d_model=embedding_dim, 
                               kernel_size=ks, 
                               expansion_factor=expansion_factor, 
                               dropout=dropout_rate)
            )
        self.backbone = nn.Sequential(*blocks)
        
        self.attn_pool = AttentionPooling1D(in_features=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.cascade_head = CascadeRegressionHead(d_model=embedding_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            if m.padding_idx is not None:
                nn.init.constant_(m.weight[m.padding_idx], 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size, seq_len = x.size()
        
        mask = (x != 0) 
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)
        pos_emb = pos_emb * mask.unsqueeze(-1).float() 
        
        embedded = self.embedding(x) + pos_emb
        embedded = embedded * mask.unsqueeze(-1).float()
        embedded = embedded.permute(0, 2, 1) 
        
        features = self.backbone(embedded)    
        features = features.permute(0, 2, 1) 
        pooled_features = self.attn_pool(features, mask=mask)
        
        x_drop = self.dropout(pooled_features)
        
        return self.cascade_head(x_drop)