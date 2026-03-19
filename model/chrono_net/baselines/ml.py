import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from typing import List
import numpy as np
import torch
import torch.nn as nn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model.chrono_net import register_model

class MLBaselineWrapper(nn.Module):
    def __init__(self, 
                 ml_type: str, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 random_seed: int = 42) -> None:
        super().__init__()
        self.ml_type = ml_type.upper()
        self.num_classes_list = num_classes_list
        
        # Dùng nn.Embedding để trích xuất đặc trưng
        self.embedding = nn.Embedding.from_pretrained(w2v_tensor, freeze=True)
        
        # Khởi tạo lõi mô hình Machine Learning
        if self.ml_type == "XGBOOST":
            base_model = XGBClassifier(use_label_encoder=False, 
                                       eval_metric="mlogloss", 
                                       random_state=random_seed, 
                                       n_jobs=-1)
        elif self.ml_type == "LIGHTGBM":
            base_model = LGBMClassifier(random_state=random_seed, 
                                        n_jobs=-1, 
                                        verbose=-1)
        elif self.ml_type == "RANDOMFOREST":
            base_model = RandomForestClassifier(n_estimators=100, 
                                                random_state=random_seed, 
                                                n_jobs=-1)
        elif self.ml_type == "SVM":
            base_model = SVC(kernel="rbf", 
                             probability=True, 
                             random_state=random_seed)
        else:
            raise ValueError(f"Không hỗ trợ mô hình ML: {ml_type}")
            
        # Bọc lại để dự đoán 6 thuộc tính cùng lúc
        self.multi_target_model = MultiOutputClassifier(base_model, n_jobs=1)
        self.is_fitted = False
        self.label_encoders = []

    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            mask = (x != 0).unsqueeze(-1).float()
            emb = self.embedding(x)
            
            sum_emb = torch.sum(emb * mask, dim=1)
            valid_lens = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled_features = sum_emb / valid_lens
            
        return pooled_features.cpu().numpy()

    def fit(self, 
            x_np: np.ndarray, 
            y_np: np.ndarray) -> None:
        self.label_encoders = [LabelEncoder() for _ in range(y_np.shape[1])]
        
        y_encoded = np.zeros_like(y_np)
        for i in range(y_np.shape[1]):
            # Mã hóa nhãn [1, 2, ..., 12] thành [0, 1, ..., 11]
            y_encoded[:, i] = self.label_encoders[i].fit_transform(y_np[:, i])
            
        self.multi_target_model.fit(x_np, y_encoded)
        self.is_fitted = True

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError(f"Mô hình {self.ml_type} chưa được huấn luyện! Hãy gọi .fit() trước.")
            
        features = self.extract_features(x)
        probas = self.multi_target_model.predict_proba(features)
        
        outputs = []
        for i, prob_array in enumerate(probas):
            batch_size = x.size(0)
            num_classes = self.num_classes_list[i]
            tensor_prob = torch.zeros((batch_size, num_classes), dtype=torch.float32, device=x.device)
            
            original_classes = self.label_encoders[i].classes_
            for new_idx, old_val in enumerate(original_classes):
                tensor_prob[:, int(old_val)] = torch.tensor(prob_array[:, new_idx], dtype=torch.float32, device=x.device)
                
            outputs.append(tensor_prob)
            
        return outputs
    

@register_model("baseline_xgboost")
class XGBoostModel(MLBaselineWrapper):
    def __init__(self, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 random_seed: int) -> None:
        super().__init__(
            ml_type="XGBOOST", 
            num_classes_list=num_classes_list, 
            w2v_tensor=w2v_tensor, 
            random_seed=random_seed
        )


@register_model("baseline_lightgbm")
class LightGBMModel(MLBaselineWrapper):
    def __init__(self, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 random_seed: int) -> None:
        super().__init__(
            ml_type="LIGHTGBM", 
            num_classes_list=num_classes_list, 
            w2v_tensor=w2v_tensor, 
            random_seed=random_seed
        )


@register_model("baseline_randomforest")
class RandomForestModel(MLBaselineWrapper):
    def __init__(self, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 random_seed: int) -> None:
        super().__init__(
            ml_type="RANDOMFOREST", 
            num_classes_list=num_classes_list, 
            w2v_tensor=w2v_tensor, 
            random_seed=random_seed
        )


@register_model("baseline_svm")
class SVMModel(MLBaselineWrapper):
    def __init__(self, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 random_seed: int) -> None:
        super().__init__(
            ml_type="SVM", 
            num_classes_list=num_classes_list, 
            w2v_tensor=w2v_tensor, 
            random_seed=random_seed
        )