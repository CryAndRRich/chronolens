from typing import List, Dict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONFIG_DATA, CONFIG_MODEL


def post_process_predictions(preds_tensor: torch.Tensor) -> torch.Tensor:
    """
    Áp dụng các quy tắc hậu xử lý cho dự đoán:
    - Làm tròn dự đoán về số nguyên gần nhất
    - Điều chỉnh tháng về khoảng 1-12
    - Điều chỉnh ngày về khoảng 1 đến số ngày tối đa của tháng tương ứng
    - Điều chỉnh chỉ số nhà máy về khoảng 0-99
    """
    preds = preds_tensor.clone()
    preds = torch.round(preds)
    
    preds[:, [0, 3]] = (preds[:, [0, 3]] - 1) % 12 + 1
    preds[:, [0, 3]] = torch.clamp(preds[:, [0, 3]], 1, 12)

    days_in_month_t = torch.tensor(CONFIG_DATA.DAYS_IN_MONTH, 
                                   device=preds.device, 
                                   dtype=preds.dtype)
    
    for col_idx, month_idx in [(1, 0), (4, 3)]:
        m_indices = (preds[:, month_idx].long() - 1).clamp(0, 11)
        max_days = days_in_month_t[m_indices]
        
        preds[:, col_idx] = torch.clamp(preds[:, col_idx], min=1)
        preds[:, col_idx] = torch.min(preds[:, col_idx], max_days)

    preds[:, [2, 5]] = torch.clamp(preds[:, [2, 5]], 0, 99)
    
    return preds


def run_inference(model: nn.Module, 
                  loader: DataLoader, 
                  device: torch.device) -> Dict[str, List[int]]:
    """
    Chạy inference trên tập dữ liệu và thu thập dự đoán cho mỗi thuộc tính
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            outputs = model(x.to(device))
            all_preds.append(torch.stack(outputs, dim=1) if isinstance(outputs, list) else outputs)
    
    all_preds = torch.cat(all_preds, dim=0)[:, [1, 2, 3, 5, 6, 7]]
    return post_process_predictions(all_preds).cpu().numpy()


def evaluate_wmse(y_true: np.ndarray, 
                  y_pred: np.ndarray) -> float:
    """
    Tính Weighted Mean Squared Error (WMSE) giữa y_true và y_pred
    """
    M, W = np.array(CONFIG_MODEL.M_LIST_METRIC), np.array(CONFIG_MODEL.W_LIST_VALS)
    return np.mean(np.sum(W * np.power((y_true - y_pred) / M, 2), axis=1))


def evaluate_wmape(y_true: np.ndarray,
                   y_pred: np.ndarray) -> float:
    """
    Tính Weighted Mean Absolute Percentage Error (WMAPE) giữa y_true và y_pred
    """
    sum_abs_error = np.sum(np.abs(y_true - y_pred))
    sum_abs_true = np.sum(np.abs(y_true))
    return (sum_abs_error / (sum_abs_true + 1e-8)) * 100

def get_stats(y_true: np.ndarray, 
              y_pred: np.ndarray) -> None:
    """
    Tính và in ra các chỉ số MAE, MSE, RMSE, WMAPE cho từng thuộc tính và tổng thể
    """
    print("\n" + "=" * 76)
    print(f"{'THUỘC TÍNH':<12} | {'MAE':<10} | {'MSE':<12} | {'RMSE':<10} | {'WMAPE':<10}")
    print("-" * 76)

    for i, attr in enumerate(CONFIG_DATA.ATTRIBUTE_LIST):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        mae = np.mean(np.abs(y_t - y_p))
        mse = np.mean(np.square(y_t - y_p))
        rmse = np.sqrt(mse)
        wmape_val = evaluate_wmape(y_t, y_p)
        
        print(f"Attr {attr:<7} | {mae:<10.4f} | {mse:<12.4f} | {rmse:<10.4f} | {wmape_val:<10.4f}")

    print("=" * 76)

    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_mse = np.mean(np.square(y_true - y_pred))
    overall_rmse = np.sqrt(overall_mse)
    overall_wmape = evaluate_wmape(y_true, y_pred)
    final_wmse = evaluate_wmse(y_true, y_pred)

    print(f"{'MAE':<15}: {overall_mae:.4f}")
    print(f"{'MSE':<15}: {overall_mse:.4f}")
    print(f"{'RMSE':<15}: {overall_rmse:.4f}")
    print(f"{'WMAPE':<15}: {overall_wmape:.4f}")
    print(f"{'WMSE':<15}: {final_wmse:.4f}")
    print("=" * 76)