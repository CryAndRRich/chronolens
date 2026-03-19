import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.chrono_net.layers import AttentionPooling1D, GCEFusion, CascadeRegressionHead
from model.chrono_net import register_model


class ResidualBiGRUBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.2) -> None:
        """
        Lớp ResidualBiGRUBlock thực hiện một block RNN với kết nối residual và normalization
        """
        super(ResidualBiGRUBlock, self).__init__()
        self.gru = nn.GRU(input_size=d_model, 
                          hidden_size=d_model // 2, 
                          batch_first=True, 
                          bidirectional=True)
        self.norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out, _ = self.gru(x)
        out = self.dropout(out)
        return self.norm(res + out)


class MultiScaleConv1D(nn.Module):
    def __init__(self, d_model: int) -> None:
        """
        Lớp MultiScaleConv1D thực hiện các phép convolution với nhiều kích thước 
        kernel khác nhau để trích xuất đặc trưng đa quy mô
        """
        super(MultiScaleConv1D, self).__init__()
        dim = d_model // 4
        self.conv1 = nn.Conv1d(in_channels=d_model, 
                               out_channels=dim, 
                               kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model, 
                               out_channels=dim, 
                               kernel_size=3, 
                               padding=1)
        self.conv5 = nn.Conv1d(in_channels=d_model,
                               out_channels=dim, 
                               kernel_size=5, 
                               padding=2)
        self.conv_dilated = nn.Conv1d(in_channels=d_model, 
                                      out_channels=dim, 
                                      kernel_size=3, 
                                      padding=2, 
                                      dilation=2) 
        self.proj = nn.Linear(in_features=d_model, 
                              out_features=d_model)
        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2) 
        c1 = F.gelu(self.conv1(x_t))
        c3 = F.gelu(self.conv3(x_t))
        c5 = F.gelu(self.conv5(x_t))
        cd = F.gelu(self.conv_dilated(x_t))
        
        out = torch.cat([c1, c3, c5, cd], dim=1).transpose(1, 2)
        return self.norm(x + self.proj(out))
    

@register_model("chrono_r")
class ChronoR(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 seq_length: int = 66,
                 embedding_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout_rate: float = 0.3) -> None:
        """
        Khởi tạo mô hình ChronoR
        
        Tham số:
            vocab_size: Kích thước từ điển
            embedding_dim: Số chiều của vector nhúng từ điển
            hidden_dim: Số chiều của hidden state
            num_layers: Số lớp RNN
            dropout_rate: Tỷ lệ dropout cho các lớp fully connected
        """
        super(ChronoR, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        self.pos_embedding = nn.Embedding(num_embeddings=seq_length, 
                                          embedding_dim=embedding_dim)
        
        self.multi_scale_conv = MultiScaleConv1D(d_model=embedding_dim)
        self.conv_dropout = nn.Dropout(p=dropout_rate)
        
        self.rnn_blocks = nn.ModuleList([
            ResidualBiGRUBlock(d_model=embedding_dim, 
                               dropout=dropout_rate) 
            for _ in range(num_layers)
        ])
        
        self.gce_fusion = GCEFusion(vocab_size, embedding_dim, dropout_rate)
        
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=4, 
            dim_feedforward=embedding_dim * 2, 
            dropout=dropout_rate, 
            batch_first=True, 
            activation="gelu"
        )
        
        self.attn_pool = AttentionPooling1D(embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cascade_head = CascadeRegressionHead(embedding_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, L = x.size()
        mask = (x != 0)
        mask_float = mask.unsqueeze(-1).float()
        
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h_emb = (self.embedding(x) + self.pos_embedding(pos)) * mask_float
        
        h_conv = self.multi_scale_conv(h_emb)
        h_conv = self.conv_dropout(h_conv) * mask_float
        
        h_rnn = h_conv
        for rnn_block in self.rnn_blocks:
            h_rnn = rnn_block(h_rnn) * mask_float
            
        h_fuse = self.gce_fusion(h_rnn, x) * mask_float
        
        h_attn = self.transformer(h_fuse, src_key_padding_mask=~mask)
        
        h_final = h_emb + h_conv + h_attn
        
        x_pool = self.dropout(self.attn_pool(h_final, mask))
        return self.cascade_head(x_pool)