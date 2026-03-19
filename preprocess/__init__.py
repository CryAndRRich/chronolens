from typing import Tuple, Union
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocess.preprocess_data import drop_duplicates, drop_overlap, validate_and_clean_dates, build_vocab_mapping, apply_vocab_mapping, manual_augment
from preprocess.dataloader import UserBehaviorDataset, create_dataloaders
from config import CONFIG_DATA

__all__ = ["DataManager", "UserBehaviorDataset"]


class DataManager:
    def __init__(self, 
                 input_root: str, 
                 work_dir: str, 
                 config_data: CONFIG_DATA,
                 seed_worker,
                 data_generator: torch.Generator,
                 random_seed: int) -> None:
        
        self.INPUT_ROOT = input_root
        self.WORK_DIR = work_dir
        self.SEED_WORKER = seed_worker
        self.DATA_GENERATOR = data_generator
        self.RANDOM_SEED = random_seed

        # Cấu hình hằng số
        self.SEQ_LENGTH = config_data.SEQ_LENGTH
        self.FEATURE_COLS = config_data.FEATURE_COLS
        self.ATTRIBUTE_LIST = config_data.ATTRIBUTE_LIST
        self.ATTRIBUTE_COLS = config_data.ATTRIBUTE_COLS
        self.EMBEDDING_DIM = config_data.EMBEDDING_DIM
        self.BATCH_SIZE = config_data.BATCH_SIZE
        self.NUM_WORKERS = config_data.NUM_WORKERS
        self.DAYS_MAP = config_data.DAYS_MAP

        # Biến trạng thái quan trọng
        self.id_to_idx = {}
        self.VOCAB_SIZE = 0
        self.UNK_TOKEN = 0

        # Khởi chạy Pipeline
        self._load_raw_csv()
        self._setup_initial_pipeline()

    def _load_raw_csv(self) -> None:
        """
        Tải dữ liệu thô ban đầu
        """
        self.x_train = pd.read_csv(f"{self.INPUT_ROOT}/X_train.csv")
        self.y_train = pd.read_csv(f"{self.INPUT_ROOT}/Y_train.csv")
        self.x_val = pd.read_csv(f"{self.INPUT_ROOT}/X_val.csv")
        self.y_val = pd.read_csv(f"{self.INPUT_ROOT}/Y_val.csv")
        self.x_test = pd.read_csv(f"{self.INPUT_ROOT}/X_test.csv")

    def _setup_initial_pipeline(self) -> None:
        """
        Pipeline tiền xử lý gốc rễ
        """
        # Định dạng kiểu dữ liệu
        for df in [self.x_train, self.x_val, self.x_test]:
            df[self.FEATURE_COLS] = df[self.FEATURE_COLS].fillna(0).astype(np.int64)
        for df in [self.y_train, self.y_val]:
            df["start_year"] = 0
            df["end_year"] = (df["attr_4"] < df["attr_1"]).astype(np.int64)
            df[self.ATTRIBUTE_COLS] = df[self.ATTRIBUTE_COLS].fillna(0).astype(np.float32)

        # Loại bỏ trùng lặp
        self.x_train, self.y_train = drop_duplicates(
            self.x_train, self.y_train, self.FEATURE_COLS, name="Train"
        )

        self.x_val, self.y_val = drop_duplicates(
            self.x_val, self.y_val, self.FEATURE_COLS, name="Val"
        )

        # Chống rò rỉ
        self.x_train, self.y_train = drop_overlap(
            self.x_train, self.y_train, self.x_val, self.FEATURE_COLS
        )

        # Xử lý ngày tháng
        self.x_train, self.y_train = validate_and_clean_dates(
            self.x_train, self.y_train, self.DAYS_MAP, name="Train"
        )

        self.x_val, self.y_val = validate_and_clean_dates(
            self.x_val, self.y_val, self.DAYS_MAP, name="Val"
        )

        # Tạo từ điển và ánh xạ
        self.id_to_idx, self.VOCAB_SIZE = build_vocab_mapping(
            [self.x_train, self.x_val, self.x_test], self.FEATURE_COLS
        )
        self.map_func = np.vectorize(lambda x: self.id_to_idx.get(x, self.UNK_TOKEN))
        for df in [self.x_train, self.x_val, self.x_test]:
            df = apply_vocab_mapping(df, self.map_func, self.FEATURE_COLS)

    def __create_dataloader(self,
                            seed_worker,
                            data_generator: torch.Generator,
                            augment=True) -> None:
        
        if augment:
            x_train_augment, y_train_augment = manual_augment(
                self.x_train, self.y_train, self.x_test, self.FEATURE_COLS
            )
        else:
            x_train_augment, y_train_augment = self.x_train, self.y_train

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            x_train_augment, 
            y_train_augment, 
            self.x_val, 
            self.y_val, 
            self.x_test, 
            self.FEATURE_COLS, 
            self.ATTRIBUTE_COLS,
            self.NUM_WORKERS, 
            self.BATCH_SIZE, 
            seed_worker=seed_worker, 
            data_generator=data_generator
        )

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Trả về dữ liệu đã được xử lý
        """
        return (self.x_train, self.y_train, self.x_val, self.y_val, self.x_test)
    
    def get_dataloaders(self, augment: bool = True) -> Union[Tuple[DataLoader, DataLoader, DataLoader], DataLoader]:
        """
        Trả về DataLoader cho tập train, test và val
        """
        self.__create_dataloader(
            seed_worker=self.SEED_WORKER, 
            data_generator=self.DATA_GENERATOR,
            augment=augment)
        return (self.train_loader, self.val_loader, self.test_loader)