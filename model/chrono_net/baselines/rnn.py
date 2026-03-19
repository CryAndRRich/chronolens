import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from typing import List
import torch
import torch.nn as nn

from model.chrono_net import register_model

class BaselineSeqModel(nn.Module):
    def __init__(self, 
                 rnn_type: str, 
                 vocab_size: int, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 embedding_dim: int = 256, 
                 hidden_dim: int = 128, 
                 num_layers: int = 1,
                 dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.rnn_type = rnn_type.upper()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
            
        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(input_size=embedding_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=num_layers, 
                              batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=embedding_dim, 
                               hidden_size=hidden_dim, 
                               num_layers=num_layers, 
                               batch_first=True,
                               dropout=dropout_rate if num_layers > 1 else 0)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=embedding_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=num_layers, 
                              batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0)
            
        self.classifier_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=hidden_dim, 
                          out_features=hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=hidden_dim // 2, 
                          out_features=num_classes)
            ) for num_classes in num_classes_list
        ])

        self.apply(self._init_weights)
        self.embedding.weight.data.copy_(w2v_tensor)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mask = (x != 0).unsqueeze(-1).float()
        emb = self.embedding(x)
        
        out, _ = self.rnn(emb) 
        
        sum_out = torch.sum(out * mask, dim=1)
        valid_lens = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled_features = sum_out / valid_lens
        
        outputs = [head(pooled_features) for head in self.classifier_heads]
        return outputs


@register_model("baseline_rnn")
class BaselineRNN(BaselineSeqModel):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 embedding_dim: int = 256, 
                 hidden_dim: int = 128, 
                 num_layers: int = 1,
                 dropout_rate: float = 0.3) -> None:
        super().__init__(
            rnn_type="RNN", 
            vocab_size=vocab_size, 
            num_classes_list=num_classes_list, 
            w2v_tensor=w2v_tensor, 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout_rate=dropout_rate
        )


@register_model("baseline_lstm")
class BaselineLSTM(BaselineSeqModel):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 embedding_dim: int = 256, 
                 hidden_dim: int = 128, 
                 num_layers: int = 1,
                 dropout_rate: float = 0.3) -> None:
        super().__init__(
            rnn_type="LSTM", 
            vocab_size=vocab_size, 
            num_classes_list=num_classes_list, 
            w2v_tensor=w2v_tensor, 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout_rate=dropout_rate
        )


@register_model("baseline_gru")
class BaselineGRU(BaselineSeqModel):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 embedding_dim: int = 256, 
                 hidden_dim: int = 128, 
                 num_layers: int = 1,
                 dropout_rate: float = 0.3) -> None:
        super().__init__(
            rnn_type="GRU", 
            vocab_size=vocab_size, 
            num_classes_list=num_classes_list, 
            w2v_tensor=w2v_tensor, 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout_rate=dropout_rate
        )