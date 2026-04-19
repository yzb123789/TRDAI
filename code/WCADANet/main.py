from WCADAN_Task import WCADANetClassifier
import argparse
import os
from scipy.stats import mannwhitneyu
import torch.distributed
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import torch.backends.cudnn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from lib.utils import normalize_reg_label
from qhoptim.pyt import QHAdam
from config.default import cfg
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
    mean_squared_log_error,
)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch v1.4, WCADANet Task Training')
    parser.add_argument('-c', '--config', type=str, required=False, default='config/WCADAN.yaml', metavar="FILE", help='Path to config file')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU ID')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.benchmark = True if len(args.gpu_id) < 2 else False
    if args.config:
        cfg.merge_from_file(args.config)
    cfg.freeze()
    task = cfg.task
    seed = cfg.seed
    train_config = {'dataset': cfg.dataset, 'resume_dir': cfg.resume_dir, 'logname': cfg.logname}
    fit_config = dict(cfg.fit)
    model_config = dict(cfg.model)
    print('Using config: ', cfg)

    return train_config, fit_config, model_config, task, seed, len(args.gpu_id)

def set_task_model(task, std=None, seed=42):

    clf = WCADANetClassifier(
        optimizer_fn=QHAdam,
        optimizer_params=dict(lr=fit_config['lr'], weight_decay=1e-5, nus=(0.8, 1.0)),
        scheduler_params=dict(gamma=0.95, step_size=20),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        layer=model_config['layer'],
        base_outdim=model_config['base_outdim'],
        k=model_config['k'],
        drop_rate=model_config['drop_rate'],
        seed=seed
    )
    eval_metric = ['auc']
    return clf, eval_metric
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, recall_score, accuracy_score, confusion_matrix
)

def compute_all_metrics(y_true, y_probs, target_precision=0.900):
    """
    Calculate all classification metrics at a given Precision
    """
    # Calculate the PR curve and its corresponding thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find the index closest to the target Precision (ignoring the last point without a threshold)
    idx = np.argmin(np.abs(precisions[:-1] - target_precision))
    best_threshold = thresholds[idx]
    
    # Use this threshold to convert probabilities into 0/1 classes
    y_pred_binary = (y_probs >= best_threshold).astype(int)
    
    # Calculate basic metrics
    auroc = roc_auc_score(y_true, y_probs)
    aupr = average_precision_score(y_true, y_probs)
    precision_at_idx = precisions[idx]
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    acc = accuracy_score(y_true, y_pred_binary)
    
    # Calculate Specificity
    # tn: Negative samples predicted correctly, fp: Negative samples predicted incorrectly
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "Precision": precision_at_idx,
        "Recall": recall,
        "F1": f1,
        "Accuracy": acc,
        "Specificity": specificity
    }
if __name__ == '__main__':

    print('===> Setting configuration ...')
    train_config, fit_config, model_config, task, seed, n_gpu = get_args()
    logname = None if train_config['logname'] == '' else train_config['dataset'] + '/' + train_config['logname']
    print('===> Getting data ...')
    omics_cols = list(range(65, 68))
    special_cols = [0, 70, 71]

    target_cols = special_cols + omics_cols
    data = pd.read_csv("./data/oncoKB_tier1_withgpt_ran7.csv",sep=',')
    # data = pd.read_csv("./data/BRCA_pos_neg_1:10.csv",sep=',')
    data.rename(columns={'Hugo_Symbol':'index'},inplace=True)
    data_vector = pd.read_csv("../../data/WCADANet_data/ALL_proteome_gpt_desc_vector.csv",sep=',')
    df_merge = pd.merge(data,data_vector,on='index')
    df_merge = df_merge.drop(['index','gene_desc'],axis=1)
    # df_merge = df_merge.drop(['index'],axis=1)
    
    # Define features and target variables
    X = df_merge.drop(columns=["class"]).values 
    print(X.shape)
    y = df_merge["class"].values

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    fold_results = []
    fold_results_aupr = []
    fold_results_acc = []
    fold_U = []
    # Initialize related objects
    best_auroc = 0.0
    best_epoch = 0
    best_model_state_dict = None
    best_scaler = None
    all_fold_results = []
    # X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(train_config['dataset'])
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f'===> Fold {fold + 1}')

        # Split the data
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        scaler = preprocessing.MinMaxScaler()
        
        X_train = scaler.fit_transform(X_train)

        X_valid = scaler.transform(X_valid)
        mu, std = None, None
        if task == 'regression':
            mu, std = y_train.mean(), y_train.std()
            print(f"Fold {fold + 1} - mean = {mu:.5f}, std = {std:.5f}")
            y_train = normalize_reg_label(y_train, std, mu)
            y_valid = normalize_reg_label(y_valid, std, mu)

        # Initialize the model
        clf, eval_metric = set_task_model(task, std, seed)
        
        # Train the model
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=eval_metric,
            max_epochs=fit_config['max_epochs'], patience=fit_config['patience'],
            batch_size=fit_config['batch_size'], virtual_batch_size=fit_config['virtual_batch_size'],
            logname=logname,
            resume_dir=train_config['resume_dir'],
            n_gpu=n_gpu,
            scaler = scaler,
            F = fold + 1
        )

        best_model_path = os.path.join(clf.log.log_dir, f'checkpoint{fold + 1}_best.pth')
        print(f"DEBUG: Checkpoint path -> {best_model_path}")
        
        if os.path.exists(best_model_path):
            print(f"Loading Best Model from: {best_model_path}")
            
            # Directly call load_model
            # Parameter descriptions：
            # - input_dim: Input dimensions, directly use the shape of the data.
            # - output_dim: Output dimensions, directly reuse the dimensions stored in the current clf object (clf.output_dim).
            clf.load_model(
                filepath=best_model_path, 
                input_dim=X_train.shape[1], 
                output_dim=clf.output_dim, 
                n_gpu=n_gpu
            )
            
        else:
            print("Warning: Best model not found! Using last epoch model.")
        preds_valid = clf.predict(X_valid)


        if task == 'classification':
         
            # If y_valid is a 2D array, convert it into a 1D array
            if len(y_valid.shape) > 1:
                y_valid = y_valid.reshape(-1)  # or y_valid = y_valid.flatten()

            # If preds_valid is a 2D array, convert it into a 1D array.
            if len(preds_valid.shape) > 1:
                preds_valid = preds_valid.reshape(-1)  # or preds_valid = preds_valid.flatten()
            combined = pd.DataFrame({'y_valid': y_valid, 'preds_valid': preds_valid})
            fold_metric = compute_all_metrics(y_valid, preds_valid, target_precision=0.900)
            all_fold_results.append(fold_metric)

            print(f"Fold {fold+1} done.")
            # Extract the prediction scores of positive and negative samples
            pos_scores = combined[combined['y_valid'] == 1]['preds_valid'].values  # Prediction scores of positive samples
            neg_scores = combined[combined['y_valid'] == 0]['preds_valid'].values  # Prediction scores of negative samples

            if len(pos_scores) > 0 and len(neg_scores) > 0:
                mwu_stat, p_value = mannwhitneyu(pos_scores, neg_scores, alternative='greater')
            else:
                p_value = 1.0  # Avoid errors caused by empty samples
            print(f"Fold {fold + 1} - Validation p-v: {p_value}")
            fold_U.append(p_value)
            
            
            # Calculate the Precision-Recall curve
            aupr = average_precision_score(y_valid, preds_valid)
            fold_results_aupr.append(aupr)
            
            print(f"Fold {fold + 1} - Validation AUPR: {aupr}")
            valid_auroc = roc_auc_score(y_valid, preds_valid)
            print(f"Fold {fold + 1} - Validation AUROC: {valid_auroc:.5f}")
            fold_results.append(valid_auroc)
            
        elif task == 'regression':
            valid_mse = mean_squared_error(y_pred=preds_valid, y_true=y_valid)
            print(f"Fold {fold + 1} - Validation MSE: {valid_mse}")
            fold_results.append(valid_mse)

    # Output the average results of 5-fold cross-validation
    if task == 'classification':
        print(fold_results)
        # --- AUROC statistics ---
        auroc_mean = np.mean(fold_results)
        # ddof=1 means calculating the Sample Standard Deviation
        auroc_std = np.std(fold_results, ddof=1)   
        
        print(f"5-Fold Cross Validation Average p-value: {np.mean(fold_U):.15f}")
        print(f"5-Fold Cross Validation AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")

        print(f"  > Mean: {auroc_mean:.5f}")
        print(f"  > Std : {auroc_std:.5f}")

        # --- AUPR statistics ---
        aupr_mean = np.mean(fold_results_aupr)
        aupr_std = np.std(fold_results_aupr, ddof=1)

        print(f"5-Fold Cross Validation AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}")
        print(f"  > Mean: {aupr_mean:.5f}")
        print(f"  > Std : {aupr_std:.5f}")
        # print(f"5-Fold Cross Validation Average ACC: {np.mean(fold_results_acc):.5f}")
        df_res = pd.DataFrame(all_fold_results)
        df_res.index = [f"Fold {i+1}" for i in range(len(all_fold_results))]

        # Calculate statistical rows
        stats = pd.DataFrame({
            'Mean': df_res.mean(),
            'Std': df_res.std(ddof=1),
            'Min': df_res.min(),
            'Max': df_res.max()
        }).T

        final_table = pd.concat([df_res, stats])

        print("\n" + "="*95)
        print("DRUGGABLE GENE PREDICTION TASK - PERFORMANCE SUMMARY")
        print("="*95)
        header = "Fold | AUROC  | AUPR  "
        print(header)
        print("-" * 95)

        for idx, row in final_table.iterrows():
            label = f"{idx:<4}"
            print(f"{label} | {row['AUROC']:.4f} | {row['AUPR']:.4f}")

        print("="*95)
    elif task == 'regression':
        print(f"5-Fold Cross Validation Average MSE: {np.mean(fold_results):.5f}")
