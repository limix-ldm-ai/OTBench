import os
import json
import math
import torch
import torch.nn as nn
import numpy as np
# import zero
from .. import lib
from ..tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .utils_train import get_model, make_dataset
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, roc_auc_score


def impute(  # only used to predict y
    parent_dir,
    real_data_path,
    model_path,
    model_type,
    model_params,
    resample_rounds,
    impute_condition,
    w_num,
    w_cat,
    trial_times,
    num_timesteps=1000,
    gaussian_loss_type='mse',
    scheduler='cosine',
    T_dict=None,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    seed=0,
    change_val=False,
):
    # zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=False,  # imputation need to impute y, thus y need to be concat with x as :[y,x]
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)

    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device
    )

    diffusion.to(device)
    diffusion.eval()

    task_type = D.task_type

    x_num_train = D.X_num['train'] if D.X_num is not None else None
    x_cat_train = D.X_cat['train'] if D.X_cat is not None else None
    x_num_test = D.X_num['test'] if D.X_num is not None else None
    x_cat_test = D.X_cat['test'] if D.X_cat is not None else None

    real_y_test = D.y['test']

    
    if torch.is_tensor(real_y_test):
        real_y_test_np = real_y_test.detach().cpu().numpy()
    else:
        real_y_test_np = np.asarray(real_y_test)

    
    real_y_test_np = real_y_test_np.reshape(-1)

    num_mask_idx, cat_mask_idx = [], []
    avg = None

    if task_type.value in ['binclass', 'multiclass']:
        cat_mask_idx += [0]
    elif task_type.value == 'regression':
        num_mask_idx += [0]
        # avg = x_num_train[:, num_mask_idx].mean(0).to(device) if x_num_train is not None else None
    else:
        raise ValueError(f'Unsupported task_type: {task_type}')

    metric_records = {
        'r2': [],
        'rmse': [],
        'acc': [],
        'f1': [],
        'auc': [],
    }
    x_num_test = torch.from_numpy(x_num_test).to(device) if x_num_test is not None else None
    x_cat_test = torch.from_numpy(x_cat_test).to(device) if x_cat_test is not None else None

    for _ in range(trial_times):
        
        x_num_test_trial = x_num_test.clone() if x_num_test is not None else None
        x_cat_test_trial = x_cat_test.clone() if x_cat_test is not None else None

        # if num_mask_idx:
        #     x_num_test_trial[:, num_mask_idx] = avg
        # if cat_mask_idx:
        #     x_cat_test_trial[:, cat_mask_idx] = torch.tensor(
        #         model_params['num_classes'],
        #         dtype=x_cat_test_trial.dtype,
        #         device=x_cat_test_trial.device
        #     )

        impute_output = diffusion.sample_impute(
            x_num_test_trial,
            x_cat_test_trial,
            num_mask_idx,
            cat_mask_idx,
            resample_rounds=resample_rounds,
            impute_condition=impute_condition,
            w_num=w_num,
            w_cat=w_cat,
            return_cat_probs=task_type.value in ['binclass', 'multiclass'],
        )
        if isinstance(impute_output, tuple):
            X_imputed, impute_extra = impute_output
        else:
            X_imputed, impute_extra = impute_output, {}

        X_num_imputed = X_imputed[:, :num_numerical_features_]
        X_cat_imputed = X_imputed[:, num_numerical_features_:]

        if task_type.value in ['binclass', 'multiclass']:
            y_imputed = X_cat_imputed[:, 0]
        elif task_type.value == 'regression':
            y_imputed = X_num_imputed[:, 0]

        if torch.is_tensor(y_imputed):
            y_imputed_np = y_imputed.detach().cpu().numpy()
        else:
            y_imputed_np = np.asarray(y_imputed)

        y_imputed_np = y_imputed_np.reshape(-1)

        if task_type.value == 'regression':
            r2 = r2_score(real_y_test_np, y_imputed_np)
            rmse = math.sqrt(mean_squared_error(real_y_test_np, y_imputed_np))
            metric_records['r2'].append(float(r2))
            metric_records['rmse'].append(float(rmse))

        else:
            
            y_pred = y_imputed_np.astype(np.int64)
            y_true = real_y_test_np.astype(np.int64)

            acc = accuracy_score(y_true, y_pred)

            if task_type.value == 'binclass':
                f1 = f1_score(y_true, y_pred, average='binary')
            else:
                f1 = f1_score(y_true, y_pred, average='macro')

            metric_records['acc'].append(float(acc))
            metric_records['f1'].append(float(f1))

            cat_probs = impute_extra.get('cat_probs')
            if cat_probs is not None:
                y_prob = cat_probs[:, :int(K[0])].numpy()
                try:
                    if task_type.value == 'binclass':
                        metric_records['auc'].append(float(roc_auc_score(y_true, y_prob[:, 1])))
                    else:
                        metric_records['auc'].append(float(roc_auc_score(
                            y_true,
                            y_prob,
                            multi_class='ovr',
                            average='macro',
                        )))
                except ValueError as exc:
                    print(f"Skipping AUC for this trial: {exc}")

    
    results = {}

    if task_type.value == 'regression':
        results['r2-count'] = float(len(metric_records['r2']))
        results['r2-mean'] = float(np.mean(metric_records['r2']))
        results['r2-std'] = float(np.std(metric_records['r2']))

        results['rmse-count'] = float(len(metric_records['rmse']))
        results['rmse-mean'] = float(np.mean(metric_records['rmse']))
        results['rmse-std'] = float(np.std(metric_records['rmse']))

    else:
        results['acc-count'] = float(len(metric_records['acc']))
        results['acc-mean'] = float(np.mean(metric_records['acc']))
        results['acc-std'] = float(np.std(metric_records['acc']))

        results['f1-count'] = float(len(metric_records['f1']))
        results['f1-mean'] = float(np.mean(metric_records['f1']))
        results['f1-std'] = float(np.std(metric_records['f1']))

        results['auc-count'] = float(len(metric_records['auc']))
        if metric_records['auc']:
            results['auc-mean'] = float(np.mean(metric_records['auc']))
            results['auc-std'] = float(np.std(metric_records['auc']))
        else:
            results['auc-mean'] = None
            results['auc-std'] = None

    os.makedirs(parent_dir, exist_ok=True)
    save_path = os.path.join(parent_dir, 'impute.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    



