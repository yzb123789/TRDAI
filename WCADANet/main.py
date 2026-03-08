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
from data.dataset import *
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

# ==========================================
# 1. 核心计算函数：在固定 Precision 处获取所有指标
# ==========================================
def compute_all_metrics(y_true, y_probs, target_precision=0.900):
    """
    在给定的 Precision 下，计算所有分类指标
    """
    # 计算 PR 曲线及其对应的阈值
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # 找到最接近目标 Precision 的索引（忽略最后一个没有阈值的点）
    idx = np.argmin(np.abs(precisions[:-1] - target_precision))
    best_threshold = thresholds[idx]
    
    # 使用该阈值将概率转换为 0/1 类别
    y_pred_binary = (y_probs >= best_threshold).astype(int)
    
    # 计算基础指标
    auroc = roc_auc_score(y_true, y_probs)
    aupr = average_precision_score(y_true, y_probs)
    precision_at_idx = precisions[idx]
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    acc = accuracy_score(y_true, y_pred_binary)
    
    # 计算 Specificity (特异性)
    # tn: 负样本当中预测对的，fp: 负样本当中预测错的
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

    # 2. 定义其他需要单独指定的列
    special_cols = [0, 70, 71]

    # 3. 合并所有需要的列索引
    # 结果是: [0, 1, 2, ..., 69, 71, 72]
    # 注意：这里会跳过第 70 列
    target_cols = special_cols + omics_cols
    data = pd.read_csv("./data/oncoKB_tier1_withgpt_ran7.csv",sep=',')
    # data = pd.read_csv("./data/BRCA_pos_neg_1:10.csv",sep=',')
    data.rename(columns={'Hugo_Symbol':'index'},inplace=True)
    data_vector = pd.read_csv("./data/ALL_proteome_gpt_desc_vector.csv",sep=',')
    df_merge = pd.merge(data,data_vector,on='index')
    df_merge = df_merge.drop(['index','gene_desc'],axis=1)
    # df_merge = df_merge.drop(['index'],axis=1)
    
    # 定义特征和目标变量
    X = df_merge.drop(columns=["class"]).values  # 假设 target 是目标变量的列名
    print(X.shape)
    y = df_merge["class"].values

    # 初始化五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    fold_results = []
    fold_results_aupr = []
    fold_results_acc = []
    fold_U = []
    # 初始化相关对象
    best_auroc = 0.0
    best_epoch = 0
    best_model_state_dict = None
    best_scaler = None
    all_fold_results = []
    # X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(train_config['dataset'])
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f'===> Fold {fold + 1}')

        # 分割数据
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        scaler = preprocessing.MinMaxScaler()
        
        X_train = scaler.fit_transform(X_train)

        X_valid = scaler.transform(X_valid)
        # X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_valid)
        # 如果是回归任务，标准化标签
        mu, std = None, None
        if task == 'regression':
            mu, std = y_train.mean(), y_train.std()
            print(f"Fold {fold + 1} - mean = {mu:.5f}, std = {std:.5f}")
            y_train = normalize_reg_label(y_train, std, mu)
            y_valid = normalize_reg_label(y_valid, std, mu)

        # 初始化模型
        clf, eval_metric = set_task_model(task, std, seed)
        
        # 训练模型
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
            
            # 2. 直接调用 load_model
            # 参数说明：
            # - input_dim: 输入维度，直接用数据的 shape
            # - output_dim: 输出维度，直接复用当前 clf 对象里存好的维度 (clf.output_dim)
            clf.load_model(
                filepath=best_model_path, 
                input_dim=X_train.shape[1], 
                output_dim=clf.output_dim, 
                n_gpu=n_gpu
            )
            
        else:
            print("Warning: Best model not found! Using last epoch model.")
        preds_valid = clf.predict(X_valid)

        # fpr, tpr, _ = roc_curve(y_valid, preds_valid) 
        # fold_auc = auc(fpr, tpr) 
        # fold_pr = average_precision_score(y_valid, preds_valid) # 计算精确率-召回率曲线，选择精确率为0.8时的阈值 
        # precision, recall, thresholds = precision_recall_curve(y_valid, preds_valid) 
        # closest_idx = np.argmin(np.abs(precision[:-1] - 0.9)) # 找到最接近目标精确率的阈值索引 
        # threshold_at_precision_09 = thresholds[closest_idx] 
        # print(f"Fold {fold + 1} - Precision at 0.9: {precision[closest_idx]:.4f}, Threshold: {threshold_at_precision_09:.9f}") 
        # _plus = np.sum(np.abs(shap_values[1][:,0:22]), axis=1)
        # _plus = np.mean(_plus,axis=0)
        # mut = np.sum(np.abs(shap_values[1][:,22:52]), axis=1)
        # mut = np.mean(mut,axis=0)
        # pro = np.sum(np.abs(shap_values[1][:,52:54]), axis=1)
        # pro = np.mean(pro,axis=0)
        # cnv = np.sum(np.abs(shap_values[1][:,54:63]), axis=1)
        # cnv = np.mean(cnv,axis=0)
        # rna = np.sum(np.abs(shap_values[1][:,63:65]), axis=1)
        # rna = np.mean(rna,axis=0)
        # methy = np.sum(np.abs(shap_values[1][:,65:67]), axis=1)
        # methy = np.mean(methy,axis=0)
        # pho = np.sum(np.abs(shap_values[1][:,67:69]), axis=1)
        # pho = np.mean(pho,axis=0)

        if task == 'classification':
            #     
            # 确保顺序一致
         
            # 如果 y_valid 是二维数组，转换为一维数组
            if len(y_valid.shape) > 1:
                y_valid = y_valid.reshape(-1)  # 或者 y_valid = y_valid.flatten()

            # 如果 preds_valid 是二维数组，转换为一维数组
            if len(preds_valid.shape) > 1:
                preds_valid = preds_valid.reshape(-1)  # 或者 preds_valid = preds_valid.flatten()
            combined = pd.DataFrame({'y_valid': y_valid, 'preds_valid': preds_valid})
            fold_metric = compute_all_metrics(y_valid, preds_valid, target_precision=0.900)
            all_fold_results.append(fold_metric)

            print(f"Fold {fold+1} done.")
            # 提取正样本和负样本的预测分数
            pos_scores = combined[combined['y_valid'] == 1]['preds_valid'].values  # 正样本预测分数
            neg_scores = combined[combined['y_valid'] == 0]['preds_valid'].values  # 负样本预测分数

            if len(pos_scores) > 0 and len(neg_scores) > 0:
                mwu_stat, p_value = mannwhitneyu(pos_scores, neg_scores, alternative='greater')
            else:
                p_value = 1.0  # 避免空样本导致报错
            print(f"Fold {fold + 1} - Validation p-v: {p_value}")
            fold_U.append(p_value)
            
            
            # 计算 Precision-Recall 曲线
            aupr = average_precision_score(y_valid, preds_valid)
            fold_results_aupr.append(aupr)
            
            # print("111",preds_valid)
            # print("222",y_valid)
            # valid_acc = accuracy_score(y_pred=preds_valid, y_true=y_valid)
            print(f"Fold {fold + 1} - Validation AUPR: {aupr}")
            valid_auroc = roc_auc_score(y_valid, preds_valid)
            print(f"Fold {fold + 1} - Validation AUROC: {valid_auroc:.5f}")
            fold_results.append(valid_auroc)
            # 计算准确率
            # valid_acc = accuracy_score(y_valid, preds_valid)
            # fold_results_acc.append(valid_acc)
            # # 打印准确率
            # print(f"Fold {fold + 1} - Validation Accuracy: {valid_acc:.5f}")
            
        elif task == 'regression':
            valid_mse = mean_squared_error(y_pred=preds_valid, y_true=y_valid)
            print(f"Fold {fold + 1} - Validation MSE: {valid_mse}")
            fold_results.append(valid_mse)

    # 输出五折交叉验证的平均结果
    if task == 'classification':
        print(fold_results)
        # --- 1. AUROC 统计 ---
        auroc_mean = np.mean(fold_results)
        # ddof=1 表示计算样本标准差 (Sample Standard Deviation)
        auroc_std = np.std(fold_results, ddof=1)   
        
        print(f"5-Fold Cross Validation Average p-value: {np.mean(fold_U):.15f}")
        print(f"5-Fold Cross Validation AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
        # 单独打印方便复制
        print(f"  > Mean: {auroc_mean:.5f}")
        print(f"  > Std : {auroc_std:.5f}")

        # --- 2. AUPR 统计 ---
        aupr_mean = np.mean(fold_results_aupr)
        aupr_std = np.std(fold_results_aupr, ddof=1)

        print(f"5-Fold Cross Validation AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}")
        print(f"  > Mean: {aupr_mean:.5f}")
        print(f"  > Std : {aupr_std:.5f}")
        # print(f"5-Fold Cross Validation Average ACC: {np.mean(fold_results_acc):.5f}")
        df_res = pd.DataFrame(all_fold_results)
        df_res.index = [f"Fold {i+1}" for i in range(len(all_fold_results))]

        # 计算统计行
        stats = pd.DataFrame({
            'Mean': df_res.mean(),
            'Std': df_res.std(ddof=1),
            'Min': df_res.min(),
            'Max': df_res.max()
        }).T

        # 合并展示
        final_table = pd.concat([df_res, stats])

        # 打印美化后的表格
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
