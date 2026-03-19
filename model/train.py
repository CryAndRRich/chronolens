from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import CONFIG_MODEL
from model.loss import HybridThresholdLoss
from preprocess import DataManager
from utils.evaluate import run_inference, evaluate_wmse
from utils.prepare_model import get_model_optim_schedule


def train_model(model_name: str, 
                data_manager: DataManager, 
                train_loader: DataLoader, 
                val_loader: DataLoader,  
                y_val: torch.Tensor, 
                num_epochs: int,
                early_stopping: int,
                checkpoint_file: str,
                device: torch.device,
                verbose: bool = False) -> int:
    """
    Huấn luyện mô hình với Early Stopping và lưu checkpoint của mô hình tốt nhất
    
    Tham số:
        model_name: Tên mô hình (ví dụ: "chrono_r", "chrono_c", "chrono_g")
        data_manager: Đối tượng quản lý dữ liệu
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập validation
        y_val: Tensor chứa nhãn thực tế của tập validation
        num_epochs: Số epoch tối đa để huấn luyện
        early_stopping: Số epoch để dừng sớm nếu không cải thiện
        checkpoint_file: Đường dẫn để lưu checkpoint của mô hình tốt nhất
        device: Thiết bị để huấn luyện (CPU hoặc GPU)
        verbose: Nếu True, in thông tin huấn luyện sau mỗi epoch
    
    Trả về:
        best_epoch: Epoch mà mô hình đạt điểm số tốt nhất trên tập validation
    """
    
    scheduler_kwargs = deepcopy(CONFIG_MODEL.SCHEDULER_KWARGS)
    scheduler_kwargs["epochs"] = num_epochs
    scheduler_kwargs["steps_per_epoch"] = len(train_loader)
    
    model, optimizer, scheduler = get_model_optim_schedule(
        model_name=model_name,
        data_manager=data_manager,  
        model_kwargs=CONFIG_MODEL.MODEL_KWARGS[model_name],
        optim_kwargs=CONFIG_MODEL.OTIMIZER_KWARGS,
        scheduler_kwargs=scheduler_kwargs,
        device=device
    )
    loss_fn = HybridThresholdLoss(
        start_threshold=CONFIG_MODEL.START_THRESHOLD,
        end_threshold=CONFIG_MODEL.END_THRESHOLD,
        m_list_loss=CONFIG_MODEL.M_LIST_LOSS,
        start_weights=CONFIG_MODEL.START_WEIGHTS,
        end_weights=CONFIG_MODEL.END_WEIGHTS,
        total_epochs=CONFIG_MODEL.TOTAL_EPOCHS,
        device=device
    )
    
    best_score = float("inf")
    best_epoch = 0 
    scaler = GradScaler()
    early_count = 0
    
    for epoch in range(num_epochs):
        if epoch <= loss_fn.total_epochs:
            loss_fn.update_params(epoch)
        
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                preds = model(x)
                
            loss = loss_fn(preds, y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        y_pred_val = run_inference(model, val_loader, device)
        val_score = evaluate_wmse(y_val, y_pred_val)
        
        if verbose:
            print(f"Epoch {epoch + 1:02d}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Train loss: {avg_train_loss:.4f} | Val score: {val_score:.4f}")
        
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch + 1 
            torch.save(model.state_dict(), checkpoint_file)
            if verbose:
                print(f"  => Mô hình tốt nhất! (Val score: {val_score:.4f})")
            early_count = 0
        else:
            early_count += 1
            if verbose:
                print(f"  => Mô hình không cải thiện ({early_count}/{early_stopping})")
            if early_count >= early_stopping:
                if verbose:
                    print(f"  => Dừng sớm tại epoch {epoch + 1} (Best score: {best_score:.4f})")
                break
                
    return best_epoch 


def retrain_model(model_name: str,
                  data_manager: DataManager, 
                  data_loader: DataLoader, 
                  num_epochs: int,
                  checkpoint_file: str,
                  device: torch.device,
                  verbose: bool = False) -> int:
    
    scheduler_kwargs = deepcopy(CONFIG_MODEL.SCHEDULER_KWARGS)
    scheduler_kwargs["epochs"] = num_epochs
    scheduler_kwargs["steps_per_epoch"] = len(data_loader)
    
    model, optimizer, scheduler = get_model_optim_schedule(
        model_name=model_name,
        data_manager=data_manager,  
        model_kwargs=CONFIG_MODEL.MODEL_KWARGS[model_name],
        optim_kwargs=CONFIG_MODEL.OTIMIZER_KWARGS,
        scheduler_kwargs=scheduler_kwargs,
        device=device
    )
    loss_fn = HybridThresholdLoss(
        start_threshold=CONFIG_MODEL.START_THRESHOLD,
        end_threshold=CONFIG_MODEL.END_THRESHOLD,
        m_list_loss=CONFIG_MODEL.M_LIST_LOSS,
        start_weights=CONFIG_MODEL.START_WEIGHTS,
        end_weights=CONFIG_MODEL.END_WEIGHTS,
        total_epochs=CONFIG_MODEL.TOTAL_EPOCHS,
        device=device
    )
    
    scaler = GradScaler()

    for epoch in range(num_epochs):
        if epoch <= loss_fn.total_epochs:
            loss_fn.update_params(epoch)
        
        model.train()
        total_loss = 0.0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                preds = model(x)
                
            loss = loss_fn(preds, y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        
        if verbose:
            print(f"Retrain Epoch {epoch + 1:02d}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Train Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), checkpoint_file)
    return model