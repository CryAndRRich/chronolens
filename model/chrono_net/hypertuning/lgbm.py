from typing import Any, Dict, Tuple
from tqdm.auto import tqdm
import numpy as np
import lightgbm as lgb

import optuna
from optuna_integration import LightGBMPruningCallback

def get_optuna_lgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.8), 
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
    }
    return params


def train_lgb_model(x_train: np.ndarray, 
                    y_train: np.ndarray, 
                    x_val: np.ndarray, 
                    y_val: np.ndarray, 
                    x_full: np.ndarray, 
                    y_full: np.ndarray, 
                    attr_name: str,
                    n_trials: int,
                    fixed_params: Dict[str, Any],
                    random_seed: int) -> Tuple[lgb.Booster, lgb.Booster]:
    
    train_data = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, free_raw_data=False)
    
    full_train_data = lgb.Dataset(x_full, label=y_full, free_raw_data=False)
    
    def objective_lgb(trial: optuna.Trial) -> float:
        params = {
            **fixed_params,
            **get_optuna_lgb_params(trial),
        }
        pruning_callback = LightGBMPruningCallback(trial, "rmse")
        
        gbm_cv = lgb.train(
            params,
            train_data,
            num_boost_round=800, 
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                pruning_callback
            ]
        )
        return gbm_cv.best_score["valid_0"]["rmse"]

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
        study.optimize(objective_lgb, n_trials=n_trials, callbacks=[tqdm_callback])

    best_params = {
        **fixed_params, 
        **study.best_params
    }
    
    print(f"Tham số tốt nhất cho Attr_{attr_name}: {best_params}")

    best_params["learning_rate"] = best_params["learning_rate"] * 0.5

    print(f"-> Dò tìm số vòng lặp tối ưu trên tập Validation...")
    search_lgb = lgb.train(
        best_params,
        train_data,
        num_boost_round=3500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False)
        ]
    )
    optimal_iters = search_lgb.best_iteration
    
    retrain_iters = int(optimal_iters * 1.1) 
    print(f"-> Retrain trên Train + Val với max {retrain_iters} vòng lặp...")
    
    final_lgb = lgb.train(
        best_params,
        full_train_data,
        num_boost_round=retrain_iters, 
        valid_sets=[full_train_data], 
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=500)
        ]
    )
    
    print(f"Hoàn tất Attr_{attr_name} | Retrain Iteration: {final_lgb.best_iteration}\n" + "=" * 50)
    
    return search_lgb, final_lgb