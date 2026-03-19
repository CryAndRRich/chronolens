from typing import Callable, List, Tuple, Dict
import numpy as np
import pandas as pd


def drop_duplicates(x_df: pd.DataFrame,
                    y_df: pd.DataFrame,
                    feature_cols: List[str],
                    name: str ="Train") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loại bỏ các hàng trùng lặp
    """
    len_before = len(x_df)
    keep_idx = x_df.drop_duplicates(subset=feature_cols, keep="first").index
    
    x_df = x_df.loc[keep_idx].reset_index(drop=True)
    y_df = y_df.loc[keep_idx].reset_index(drop=True)
    
    # print(f" - Loại bỏ {len_before - len(x_df)} hàng trùng lặp trong {name}")
    return x_df, y_df


def drop_overlap(x_train: pd.DataFrame, 
                 y_train: pd.DataFrame, 
                 x_val: pd.DataFrame, 
                 feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loại bỏ các chuỗi trong Train bị trùng với Val
    """
    len_before = len(x_train)
    
    val_features_unique = x_val[feature_cols].drop_duplicates().copy()
    val_features_unique["is_in_val"] = True
    
    x_train_copy = x_train.copy()
    x_train_copy["old_idx"] = x_train_copy.index

    train_merged = x_train_copy.merge(val_features_unique, on=feature_cols, how="left")
    
    clean_train_idx = train_merged[train_merged["is_in_val"].isna()]["old_idx"]
    
    x_train = x_train.loc[clean_train_idx].drop(columns=["old_idx"]).reset_index(drop=True)
    y_train = y_train.loc[clean_train_idx].reset_index(drop=True)
    
    # print(f" - Loại bỏ {len_before - len(x_train)} chuỗi Train bị trùng với Val")
    return x_train, y_train


def validate_and_clean_dates(x_df: pd.DataFrame, 
                             y_df: pd.DataFrame, 
                             days_map: Dict[int, int],
                             name: str ="Train") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kiểm tra và sửa lỗi ngày tháng trong các cột attr_2 (ngày bắt đầu) và attr_5 (ngày kết thúc)
    """
    max_days_start = y_df["attr_1"].map(days_map)
    invalid_start_day_mask = y_df["attr_2"] > max_days_start

    max_days_end = y_df["attr_4"].map(days_map)
    invalid_end_day_mask = y_df["attr_5"] > max_days_end

    all_invalid_mask = invalid_start_day_mask | invalid_end_day_mask
    invalid_count = all_invalid_mask.sum()

    if invalid_count > 0:
        # print(f" - Phát hiện {invalid_count} hàng có lỗi ngày trong {name}")
        
        y_df.loc[invalid_start_day_mask, "attr_2"] = max_days_start[invalid_start_day_mask]
        y_df.loc[invalid_end_day_mask, "attr_5"] = max_days_end[invalid_end_day_mask]
        
    # else:
    #     print(f" - Không có lỗi ngày tháng trong {name}")

    return x_df, y_df


def build_vocab_mapping(dfs: List[pd.DataFrame], 
                        feature_cols: List[str]) -> Tuple[Dict[int, int], int]:
    """
    Quét qua dữ liệu để tạo bộ từ điển id_to_idx
    """
    x_data = pd.concat([df[feature_cols] for df in dfs], axis=0)
    unique_ids = pd.unique(x_data.values.ravel())
    unique_ids = [uid for uid in unique_ids if pd.notna(uid) and uid != 0]
    
    id_to_idx = {0: 0, "UNK": 1}
    for idx, raw_id in enumerate(sorted(unique_ids), start=2):
        id_to_idx[raw_id] = idx
        
    vocab_size = max(id_to_idx.values()) + 1

    # print(f" - Kích thước từ điển {vocab_size} (PAD=0, UNK=1)")
    return id_to_idx, vocab_size


def apply_vocab_mapping(df: pd.DataFrame, 
                        map_func: Callable, 
                        feature_cols: List[str]) -> pd.DataFrame:
    """
    Áp dụng bộ từ điển vào DataFrame
    """
    df_mapped = df.copy()
    for col in feature_cols:
        df_mapped.loc[:, feature_cols] = map_func(df_mapped[feature_cols].values)
    return df_mapped


def manual_augment(x_train: pd.DataFrame, 
                   y_train: pd.DataFrame, 
                   x_test: pd.DataFrame, 
                   feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tăng cường dữ liệu bằng cách nhân bản các chuỗi có độ dài 
    tương tự trong tập Train để cân bằng với tập Test
    """
    train_lengths = (x_train[feature_cols] != 0).sum(axis=1)
    test_lengths = (x_test[feature_cols] != 0).sum(axis=1)
    
    train_counts = train_lengths.value_counts()
    test_counts = test_lengths.value_counts()
    
    augmented_x = [x_train]
    augmented_y = [y_train]
    
    for length, c_test in test_counts.items():
        c_train = train_counts.get(length, 0)

        if c_train > 0 and c_test > c_train:
            if c_train == 0:
                multiplier = 10
            else:
                multiplier = min(c_test / c_train, 10)
            
            n_repeats = int(multiplier) - 1
            
            if n_repeats > 0:
                mask = (train_lengths == length)
                x_to_dup = x_train[mask]
                y_to_dup = y_train[mask]
                
                if len(x_to_dup) > 0:
                    for _ in range(n_repeats):
                        augmented_x.append(x_to_dup)
                        augmented_y.append(y_to_dup)
    x_train_new = pd.concat(augmented_x, ignore_index=True)
    y_train_new = pd.concat(augmented_y, ignore_index=True)
    
    idx = np.random.permutation(len(x_train_new))
    x_train_new = x_train_new.iloc[idx].reset_index(drop=True)
    y_train_new = y_train_new.iloc[idx].reset_index(drop=True)
    
    # print(f" - Train tăng từ {len(x_train)} lên {len(x_train_new)} mẫu")
    return x_train_new, y_train_new