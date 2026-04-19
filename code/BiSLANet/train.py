from model import BiSLANet
from time import time
import numpy as np
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainner import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import joblib
from sklearn.model_selection import StratifiedKFold
cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
parser = argparse.ArgumentParser(description="BiSLANet for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='sample')

args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    experiment = None
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}')


    print("start...")
    print(f"dataset:{args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'../../data/BiSLANet_data'
    dataFolder = os.path.join(dataFolder)

    train_path = os.path.join(dataFolder, 'ALL_drug_smiles_withoutduplicate.csv')
    train_path_vector = os.path.join(dataFolder, 'ALL_drug_smiles_withoutduplicate_vector.csv')

    df_train = pd.read_csv(train_path)
    # labels_train = df_character['class'].values
    df_number_train = df_train.drop(['Hugo_Symbol','gene_desc'],axis=1)

    df_vector_train = pd.read_csv(train_path_vector,sep=',')
    df_vector_number_train = df_vector_train.drop(['Hugo_Symbol'],axis=1)

    merged_df_train = pd.concat([df_number_train,df_vector_number_train], axis=1)
    # merged_df_train = pd.concat([df_number_train], axis=1)
    merged_df_train.to_csv('./temp.csv', sep = ',')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
    results = []
    best_model = None
    best_fold = None
    best_performance = -float('inf')  # Initialize with a very low value
    auroc_values = []
    auprc_values = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(merged_df_train, merged_df_train['class'])):
        print(f"Training fold {fold+1}/5")

        # Split the dataset into train and validation sets
        train_data = merged_df_train.iloc[train_idx]
        val_data = merged_df_train.iloc[val_idx]
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        # Drop 'SMILES' and 'class' columns before applying scaler
        train_features = train_data.drop(['SMILES', 'class'], axis=1)
        # print(train_features)
        val_features = val_data.drop(['SMILES', 'class'], axis=1)
        # train_features = train_data.drop(['Canonical_SMILES','Isomeric_SMILES','Drug', 'class'], axis=1)
        # val_features = val_data.drop(['Canonical_SMILES','Isomeric_SMILES','Drug', 'class'], axis=1)
        # Normalize the train data using fit_transform and the validation data using transform
        scaler = preprocessing.MinMaxScaler()
        
        # Apply fit_transform to the training set
        train_data_scaled = scaler.fit_transform(train_features)
       # Save the scaler for future use (testing phase)
        train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_features.columns)

        # Apply transform to the validation set (using the scaler fitted on the training set)
        val_data_scaled = scaler.transform(val_features)
        
        val_data_scaled = pd.DataFrame(val_data_scaled, columns=val_features.columns)

        # Reattach 'SMILES' and 'class' columns back to the scaled features
        # train_data = pd.concat([train_data_scaled, train_data.loc[:, ['Isomeric_SMILES', 'class']]], axis=1)
        # val_data = pd.concat([val_data_scaled, val_data.loc[:, ['Isomeric_SMILES', 'class']]], axis=1)
        # train_data.rename(columns={'Isomeric_SMILES':'SMILES'},inplace=True)
        # val_data.rename(columns={'Isomeric_SMILES':'SMILES'},inplace=True)
        train_data = pd.concat([train_data_scaled, train_data.loc[:, ['SMILES', 'class']]], axis=1)
        val_data = pd.concat([val_data_scaled, val_data.loc[:, ['SMILES', 'class']]], axis=1)
        # Initialize datasets
        train_dataset = DTIDataset(train_data.index.values, train_data)
        val_dataset = DTIDataset(val_data.index.values, val_data)

        # Dataloader parameters
        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': graph_collate_func}
        
        # DataLoader
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(val_dataset, **params)

        # Initialize model, optimizer
        model = BiSLANet(device=device, **cfg).to(device=device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

        trainer = Trainer(model, opt, device, training_generator, val_generator,None, args.data, **cfg)
        fold_result = trainer.train()
        
        auroc = fold_result["auroc"]
        auprc = fold_result["auprc"]
        model = fold_result["best_model"]

        auroc_values.append(auroc)
        auprc_values.append(auprc)
        if auroc > best_performance:
            best_performance = auroc
            joblib.dump(scaler,'../../data/BiSLANet_data/model/scalar_oncoKB.pkl')
            # print(scaler.feature_names_in_)
            torch.save(model.state_dict(),"../../data/BiSLANet_data/model/best_model.pth")
      
    
        
    avg_auroc = sum(auroc_values) / len(auroc_values)
    avg_auprc = sum(auprc_values) / len(auprc_values)
    var_auroc = np.var(auroc_values)
    var_auprc = np.var(auprc_values)
    
    print(f"\nAverage AUROC: {avg_auroc}")
    print(f"Average AUPRC: {avg_auprc}")
    print(f"Variance of AUROC: {var_auroc}")
    print(f"Variance of AUPRC: {var_auprc}")


    return 0



if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s, ")
