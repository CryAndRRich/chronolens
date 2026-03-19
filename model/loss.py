from typing import List, Tuple
import torch
import torch.nn as nn


class HybridThresholdLoss(nn.Module):
    def __init__(self, 
                 start_threshold: float, 
                 end_threshold: float,
                 m_list_loss: List[float],
                 start_weights: List[float],
                 end_weights: List[float],
                 total_epochs: int,
                 device: torch.device) -> None:
        """
        HybridThresholdLoss kết hợp giữa Huber Loss và MSE Loss với trọng số động và ngưỡng động
        
        Tham số:
            start_threshold: Ngưỡng ban đầu để phân biệt giữa Huber và MSE
            end_threshold: Ngưỡng cuối cùng để phân biệt giữa Huber và MSE
            m_list_loss: Danh sách hệ số chuẩn hóa cho từng thuộc tính
            start_weights: Trọng số ban đầu cho từng thuộc tính
            end_weights: Trọng số cuối cùng cho từng thuộc tính
            total_epochs: Tổng số epoch để huấn luyện, dùng để điều chỉnh động ngưỡng và trọng số
            device: Thiết bị để lưu trữ tensor (CPU hoặc GPU)
        """
        super(HybridThresholdLoss, self).__init__()
        self.device = device
        self.M = torch.tensor(m_list_loss, device=device).view(1, -1)
        
        self.start_w = torch.tensor(start_weights, device=device).view(1, -1)
        self.end_w = torch.tensor(end_weights, device=device).view(1, -1)
        self.w = self.start_w.clone()
        
        self.start_delta = start_threshold
        self.end_delta = end_threshold
        self.delta = start_threshold 

        self.total_epochs = total_epochs

    def update_params(self, curr_epoch: int) -> Tuple[float, float, float]:
        progress = curr_epoch / self.total_epochs
        
        self.w = self.start_w + (self.end_w - self.start_w) * progress
        
        curr_delta = self.start_delta + ((self.end_delta - self.start_delta) * progress)
        self.delta = curr_delta

        return self.w[0, 0].item(), self.w[0, 1].item(), self.w[0, 3].item()
        
    def forward(self, 
                preds_list: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        if isinstance(preds_list, list):
            preds = torch.stack(preds_list, dim=1)
        else:
            preds = preds_list

        abs_err = torch.abs(targets.float() - preds) / self.M
        quad = torch.clamp(abs_err, max=self.delta)
        lin = abs_err - quad

        hybrid_loss = self.w * (0.5 * (quad ** 2) + self.delta * lin)
        return torch.mean(torch.sum(hybrid_loss, dim=1))