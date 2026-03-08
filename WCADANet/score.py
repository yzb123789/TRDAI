from DAN_Task import DANetClassifier, DANetRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from lib.multiclass_utils import infer_output_dim
from lib.utils import normalize_reg_label
import numpy as np
import argparse
from data.dataset import get_data
import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch v1.4, DANet Testing')
    parser.add_argument('-d', '--dataset', type=str, default='forest', help='Dataset Name for extracting data')
    parser.add_argument('-m', '--model_file', type=str, default='./logs/best/checkpoint4.pth', metavar="FILE", help='Inference model path')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU ID')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    dataset = args.dataset
    model_file = args.model_file
    task = 'regression' if dataset in ['year', 'yahoo', 'MSLR'] else 'classification'

    return dataset, model_file, task, len(args.gpu_id)

def set_task_model(task):
    if task == 'classification':
        clf = DANetClassifier()
        metric = roc_auc_score
    elif task == 'regression':
        clf = DANetRegressor()
        metric = mean_squared_error
    return clf, metric

def prepare_data(task, y_train, y_valid, y_test):
    output_dim = 1
    mu, std = None, None
    if task == 'classification':
        output_dim, train_labels = infer_output_dim(y_train)
        target_mapper = {class_label: index for index, class_label in enumerate(train_labels)}
        y_train = np.vectorize(target_mapper.get)(y_train)
        y_valid = np.vectorize(target_mapper.get)(y_valid)
        y_test = np.vectorize(target_mapper.get)(y_test)

    elif task == 'regression':
        mu, std = y_train.mean(), y_train.std()
        print("mean = %.5f, std = %.5f" % (mu, std))
        y_train = normalize_reg_label(y_train, mu, std)
        y_valid = normalize_reg_label(y_valid, mu, std)
        y_test = normalize_reg_label(y_test, mu, std)
        
    return output_dim, std, y_train, y_valid, y_test

if __name__ == '__main__':
    dataset, model_file, task, n_gpu = get_args()
    print('===> Getting data ...')
    # X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset)
    
    data = pd.read_csv("./ALL_cnv0.1_all_lostNaN_mutsecond_withpho.csv",sep=',')
    # data.rename(columns={'Hugo_Symbol':'index'},inplace=True)
    data_vector = pd.read_csv("./data/ALL_proteome_gpt_desc_vector.csv",sep=',')
    df_merge = pd.merge(data,data_vector,on='index')
    gene = df_merge
    df_merge = df_merge.drop(['index'],axis=1)
    # print(df_merge)

    # 定义特征和目标变量
    # X = df_merge.drop(columns=["class"]).values  # 假设 target 是目标变量的列名
    # y = df_merge["class"].values
    scaler = joblib.load('./logs/best/scalar_oncoKB4.pkl')
    X = scaler.transform(df_merge)
    # output_dim, std, y_train, y_valid, y_test = prepare_data(task, y_train, y_valid, y_test)
    clf, metric = set_task_model(task)

    filepath = model_file
    clf.load_model(filepath, input_dim=X.shape[1], output_dim=2, n_gpu=n_gpu)
    
    preds_test = clf.predict(X)
    # print(preds_test[:,1])


    # 假设你已经有了 preds_test 和 df_merge DataFrame
    # preds_test[1] 是预测结果，df_merge['index'] 是基因列

    # 创建一个新的 DataFrame，将基因列和预测结果列合并
    # print(preds_test)
    # print(preds_test.shape)
    result_df = pd.DataFrame({
        'Gene': gene['index'],  # 基因列
        'Prediction': preds_test[:, 0]  # 预测结果列
    })
    
    
    result_df.sort_values(by='Prediction', ascending=False, inplace=True)

    # 保存结果到 CSV 文件
    result_df.to_csv('PANCAN-predictions_sigmoid.csv', index=False)


#     test_value = roc_auc_score(y, preds_test[:, 1])
#     # 计算 AUPR
#     test_aupr = average_precision_score(y, preds_test[:, 1])
#     print(f"FINAL TEST AUPR FOR cosmic : {test_aupr}")
    
#     if task == 'classification':
#         print(f"FINAL TEST AUROC FOR cosmic : {test_value}")

#     elif task == 'regression':
#         print(f"FINAL TEST MSE FOR {dataset} : {test_value}")
