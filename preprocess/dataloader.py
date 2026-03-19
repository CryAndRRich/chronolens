from typing import List, Tuple, Union, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class UserBehaviorDataset(Dataset):
    def __init__(self, 
                 x_df: pd.DataFrame, 
                 y_df: Optional[pd.DataFrame],
                 feature_cols: List[str], 
                 attr_cols: List[str],
                 augment: bool = False) -> None:
        """
        Dataset tùy chỉnh cho dữ liệu hành vi người dùng, hỗ trợ tăng cường dữ liệu
        
        Tham số:
            x_df: DataFrame chứa đặc trưng đầu vào
            y_df: DataFrame chứa nhãn, có thể là None nếu không có nhãn
            feature_cols: Danh sách tên cột đặc trưng trong x_df
            attr_cols: Danh sách tên cột thuộc tính trong y_df
            augment: Nếu True, sẽ áp dụng tăng cường dữ liệu ngẫu nhiên khi truy cập mẫu
        """
        self.x_data = x_df[feature_cols].values
        self.augment = augment 
        
        self.has_labels = y_df is not None
        if self.has_labels:
            self.y_data = y_df[attr_cols].values

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_tensor = torch.tensor(self.x_data[idx], dtype=torch.long)
        
        # Tăng cường dữ liệu ngẫu nhiên:
        if self.augment:
            actions = x_tensor[x_tensor != 0]
            n_actions = len(actions)
            total_slots = len(x_tensor) 
            
            # Ngẫu nhiên masking 10% số hành động
            if torch.rand(1).item() < 0.5:
                n_mask = (n_actions // 10) + 1
                possible_indices = torch.arange(n_actions - 1)
                
                if len(possible_indices) >= n_mask:
                    mask_indices = possible_indices[torch.randperm(len(possible_indices))[:n_mask]]
                    actions[mask_indices] = 0
            
            if torch.rand(1).item() < 0.5:
                new_indices = torch.randperm(total_slots)[:n_actions].sort()[0]
                dilated_x = torch.zeros_like(x_tensor)
                dilated_x[new_indices] = actions
                x_tensor = dilated_x
            else:
                new_x = torch.zeros_like(x_tensor)
                new_x[:n_actions] = actions
                x_tensor = new_x

        if self.has_labels:
            y_tensor = torch.tensor(self.y_data[idx], dtype=torch.float32)
            return x_tensor, y_tensor
        
        return x_tensor


def create_dataloaders(x_train: pd.DataFrame, 
                       y_train: pd.DataFrame, 
                       x_val: pd.DataFrame, 
                       y_val: pd.DataFrame, 
                       x_test: pd.DataFrame, 
                       feature_cols: List[str], 
                       attr_cols: List[str],
                       batch_size: int,
                       num_workers: int,
                       seed_worker,
                       data_generator: torch.Generator) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho tập train, test và val, với tăng cường dữ liệu cho các chuỗi ngắn
    
    Tham số:
        x_train, y_train, x_val, y_val, x_test: DataFrame chứa dữ liệu đặc trưng và nhãn
        feature_cols: Danh sách tên cột đặc trưng trong x_df
        attr_cols: Danh sách tên cột thuộc tính trong y_df
        batch_size: Kích thước batch cho DataLoader
        num_workers: Số lượng worker để tải dữ liệu song song
        seed_worker: Hàm để cài đặt seed cho mỗi worker trong DataLoader
        data_generator: Generator để đảm bảo tính tái lập khi tạo DataLoader
    
    Trả về:
        Tuple[DataLoader, DataLoader, DataLoader]: Tuple chứa DataLoader cho tập train, val và test
    """

    train_dataset = UserBehaviorDataset(
        x_train, y_train, 
        feature_cols, 
        attr_cols, 
        augment=True
    )
    val_dataset = UserBehaviorDataset(
        x_val, y_val, 
        feature_cols, 
        attr_cols, 
        augment=False
    )
    test_dataset = UserBehaviorDataset(
        x_test, None, 
        feature_cols, 
        attr_cols,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        worker_init_fn=seed_worker, 
        generator=data_generator,
        pin_memory=True, 
        drop_last=True, 
        persistent_workers=True  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,           
        pin_memory=True,
        persistent_workers=False 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader