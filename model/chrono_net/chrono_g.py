import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.chrono_net.layers import AttentionPooling1D, GCEFusion, CascadeRegressionHead
from model.chrono_net import register_model

class DenseGAT(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 heads: int = 4, 
                 dropout: float = 0.2) -> None:
        """
        Graph Attention Network (GAT) cho đồ thị dày đặc
        """
        super(DenseGAT, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.head_dim = out_channels // heads
        
        self.lin = nn.Linear(in_features=in_channels, 
                             out_features=out_channels, 
                             bias=False)
        self.att_src = nn.Parameter(data=torch.Tensor(1, heads, 1, self.head_dim))
        self.att_dst = nn.Parameter(data=torch.Tensor(1, heads, 1, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)
        
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, 
                x: torch.Tensor, 
                adj_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()
        h = self.lin(x)
        h = h.view(B, L, self.heads, self.head_dim).transpose(1, 2) 
        
        alpha_src = (h * self.att_src).sum(dim=-1, keepdim=True)
        alpha_dst = (h * self.att_dst).sum(dim=-1, keepdim=True)
        e = alpha_src + alpha_dst.transpose(2, 3)
        e = self.leaky_relu(e)
        
        adj_mask = adj_mask.unsqueeze(1) 

        fill_value = torch.finfo(e.dtype).min
        e = e.masked_fill(~adj_mask, fill_value)

        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        
        out = torch.matmul(alpha, h) 
        out = out.transpose(1, 2).contiguous().view(B, L, self.out_channels) 
        return out
    

class GatedDirectedGAT(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 heads: int = 4, 
                 dropout: float = 0.2) -> None:
        """
        Lớp GAT có cổng (gated) cho đồ thị có hướng, kết hợp thông tin từ cả hai hướng
        """
        super(GatedDirectedGAT, self).__init__()
        self.gat_past = DenseGAT(in_channels=d_model, 
                                 out_channels=d_model,
                                 heads=heads, 
                                 dropout=dropout)
        self.gat_future = DenseGAT(in_channels=d_model, 
                                   out_channels=d_model, 
                                   heads=heads, 
                                   dropout=dropout)
        self.msg_proj = nn.Linear(in_features=d_model * 2, 
                                  out_features=d_model)
        self.gru_cell = nn.GRUCell(input_size=d_model, 
                                   hidden_size=d_model)
        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, 
                x: torch.Tensor, 
                adj_past: torch.Tensor, 
                adj_future: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        m_comb = F.relu(self.msg_proj(torch.cat([self.gat_past(x, adj_past), self.gat_future(x, adj_future)], dim=-1)))
        h_new = self.gru_cell(m_comb.view(-1, C), x.view(-1, C)).view(B, L, C)
        return self.norm(h_new) * mask.unsqueeze(-1).float()

class GraphormerLight(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 heads: int = 4, 
                 dropout: float = 0.2) -> None:
        """
        Graphormer Light kết hợp GAT có cổng với attention toàn cục
        """
        super(GraphormerLight, self).__init__()
        self.gnn = GatedDirectedGAT(d_model=d_model, 
                                    heads=heads, 
                                    dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, 
                                               num_heads=heads, 
                                               dropout=dropout, 
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, 
                      out_features=d_model * 2), 
            nn.GELU(), 
            nn.Dropout(p=dropout), 
            nn.Linear(in_features=d_model * 2, 
                      out_features=d_model)
        )

    def forward(self, 
                x: torch.Tensor, 
                adj_past: torch.Tensor, 
                adj_future: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.gnn(x, adj_past, adj_future, mask))
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=~mask)
        x = self.norm2(x + attn_out)
        return x + self.ffn(x)


@register_model("chrono_g")
class ChronoG(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 dilations: List[int], 
                 seq_length: int = 66, 
                 embedding_dim: int = 256, 
                 window: int = 3, 
                 heads: int = 4, 
                 dropout_rate: float = 0.3) -> None:
        """
        Khởi tạo mô hình ChronoG
        
        Tham số:
            vocab_size: Kích thước từ điển
            dilations: Danh sách các độ giãn cách để áp dụng trong các lớp GAT
            seq_length: Độ dài tối đa của chuỗi đầu vào
            embedding_dim: Số chiều của vector nhúng từ điển
            window: Kích thước cửa sổ để xác định các nút lân cận trong đồ thị
            heads: Số lượng attention heads trong mỗi lớp GAT
            dropout_rate: Tỷ lệ dropout để áp dụng trong mô hình
        """
        super(ChronoG, self).__init__()

        self.window = window
        self.dilations = dilations
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_length, embedding_dim)
        
        self.layers = nn.ModuleList([
            GraphormerLight(embedding_dim, heads, dropout_rate) 
            for _ in dilations
        ])
        
        self.gce_fusion = GCEFusion(vocab_size, embedding_dim, dropout_rate)
        
        self.attn_pool = AttentionPooling1D(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.cascade_head = CascadeRegressionHead(embedding_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02) 
        elif isinstance(m, nn.GRUCell):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def _get_directed_adj_masks(self, 
                                logical_pos: torch.Tensor, 
                                valid_nodes: torch.Tensor, 
                                window: int, 
                                dilation: int, 
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tạo các mặt nạ adjacency cho đồ thị có hướng dựa trên vị trí logic của các nút 
        và các tham số cửa sổ và độ giãn cách.
        
        Tham số:
            logical_pos: Tensor chứa vị trí logic của các nút trong chuỗi
            valid_nodes: Tensor boolean chỉ ra các nút hợp lệ (không phải padding)
            window: Kích thước cửa sổ để xác định các nút lân cận
            dilation: Độ giãn cách để áp dụng khi xác định các nút lân cận
            device: Thiết bị (CPU hoặc GPU) để tạo các tensor mặt nạ
        
        Trả về:
            Tuple[torch.Tensor, torch.Tensor]: Hai tensor boolean tương ứng với mặt nạ adjacency cho các cạnh 
                                               hướng về quá khứ và tương lai
        """
        diff_mat = logical_pos.unsqueeze(2) - logical_pos.unsqueeze(1)
        abs_diff = torch.abs(diff_mat)
        
        in_window = (abs_diff <= window * dilation) & (abs_diff % dilation == 0)
        
        past_window = in_window & (diff_mat < 0)
        future_window = in_window & (diff_mat > 0)
        
        adj_past = valid_nodes & past_window
        adj_future = valid_nodes & future_window
        
        eye = torch.eye(n=logical_pos.size(1), 
                        dtype=torch.bool, 
                        device=device).unsqueeze(0)
        adj_past = adj_past | eye 
        adj_future = adj_future | eye 
        
        return (adj_past, adj_future)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, L = x.size()
        mask = (x != 0)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        
        h = (self.embedding(x) + self.pos_embedding(pos)) * mask.unsqueeze(-1).float()
        log_pos = (torch.cumsum(mask.long(), dim=1) - 1).clamp(min=0)
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        
        for i, layer in enumerate(self.layers):
            adj_p, adj_f = self._get_directed_adj_masks(log_pos, valid, self.window, self.dilations[i], x.device)
            h = layer(h, adj_p, adj_f, mask)
            
        h = self.gce_fusion(h, x)
            
        x_drop = self.dropout(self.attn_pool(h, mask))
        return self.cascade_head(x_drop)