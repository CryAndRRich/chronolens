from typing import Any, Dict, Tuple
import tqdm

import numpy as np
import xgboost as xgb

import optuna
from optuna_integration import XGBoostPruningCallback


def get_optuna_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    return params


def train_xgb_model(x_train: np.ndarray, 
                    y_train: np.ndarray, 
                    x_val: np.ndarray, 
                    y_val: np.ndarray, 
                    x_full: np.ndarray, 
                    y_full: np.ndarray, 
                    attr_name: str,
                    n_trials: int,
                    fixed_params: Dict[str, Any],
                    random_seed: int) -> Tuple[xgb.Booster, xgb.Booster]:
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dfull = xgb.DMatrix(x_full, label=y_full)
    
    def objective_xgb(trial: optuna.Trial) -> float:
        params = params = {
            **fixed_params,
            **get_optuna_xgb_params(trial),
        }
        pruning_callback = XGBoostPruningCallback(trial, "val-rmse")
        
        bst_cv = xgb.train(
            params,
            dtrain,
            num_boost_round=800,
            evals=[(dval, "val")],
            early_stopping_rounds=40,
            callbacks=[pruning_callback],
            verbose_eval=False 
        )
        return bst_cv.best_score

    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    with tqdm(total=n_trials, desc=f"Tuning Attr_{attr_name}") as pbar:
        def tqdm_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective_xgb, n_trials=n_trials, callbacks=[tqdm_callback])

    best_params = {
        **fixed_params, 
        **study.best_params
    }
    
    print(f"Tham số tốt nhất cho Attr_{attr_name}: {best_params}")

    best_params["learning_rate"] = best_params["learning_rate"] * 0.5

    print(f"-> Dò tìm số vòng lặp tối ưu trên tập Validation...")
    search_xgb = xgb.train(
        best_params,
        dtrain,
        num_boost_round=3500, 
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    optimal_iters = search_xgb.best_iteration
    
    retrain_iters = int(optimal_iters * 1.1) 
    print(f"-> Retrain trên Train + Val với max {retrain_iters} vòng lặp...")
    
    final_xgb = xgb.train(
        best_params,
        dfull,
        num_boost_round=retrain_iters, 
        evals=[(dfull, "train")], 
        early_stopping_rounds=150, 
        verbose_eval=500
    )
    
    print(f"Hoàn tất Attr_{attr_name} | Retrain Iteration: {final_xgb.best_iteration}\n" + "=" * 50)

    return search_xgb, final_xgb
    