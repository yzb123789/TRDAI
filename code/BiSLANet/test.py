from model import BINDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from test_trainner import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from domain_adaptator import Discriminator
import joblib
import numpy as np
from scipy.stats import mannwhitneyu

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
parser = argparse.ArgumentParser(description="BINDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='sample')
parser.add_argument('--split', default='random1', type=str, metavar='S', help="split task", choices=['random', 'random1', 'random2', 'random3', 'random4'])
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    experiment = None
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}')
    print("start...")
    print(f"dataset:{args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    dataFolder = f'../datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))
    test_path = os.path.join(dataFolder, "cosmic_pos_neg1-10.csv")
    scaler = joblib.load('./result/scalar_oncoKB.pkl')
    # Main program: iterate over different datasets
    df_character = pd.read_csv(test_path,sep=',')

    df_number = df_character.drop(['gene_desc','Protein','class', 'SMILES'],axis=1)
    # df_number = df_character.drop(['Isomeric_SMILES','class','index','gene_desc','Canonical_SMILES','Drug'],axis=1)
    # print(df_number)
    test_path_vector = os.path.join(dataFolder, 'ALL_proteome_gpt_desc_vector.csv')
    df_vector = pd.read_csv(test_path_vector,sep=',')

    df_merge = pd.merge(df_number,df_vector,on='index')
    # df_merge.to_csv("./temp.csv",sep=",")
    df_merge = df_merge.drop(['index'],axis=1)


    samples_test = scaler.transform(df_merge)
    samples_test = pd.DataFrame(samples_test)

    merged_df_test = pd.concat([samples_test, df_character.loc[:, ['class', 'SMILES']]], axis=1)
    # print(merged_df_test)


    test_dataset = DTIDataset(merged_df_test.index.values, merged_df_test)

    model = BINDTI(device=device, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                                                               'drop_last':True, 'collate_fn': graph_collate_func}


    test_generator = DataLoader(test_dataset, **params)
    torch.backends.cudnn.benchmark = True

    weights_dict = torch.load("./result/best_model.pth")

    model.load_state_dict(weights_dict, strict=False)
    trainer = Trainer(model, opt, device, None, None, test_generator, args.data, args.split, **cfg)
    auroc, auprc, _, _, _, _, _, _, _, y_pred, y_label = trainer.test(dataloader="test")

    y_pred = np.array(y_pred)
    y_label = np.array(y_label)


    pos_scores = y_pred[y_label == 1]
    neg_scores = y_pred[y_label == 0]


    stat, pvalue = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    print(f"测试集AUROC: {auroc:.5e}")
    print(f"测试集AUPR: {auprc:.5e}")
    print(f"测试集U检验p-value: {pvalue:.5e}")


    return 0


if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s, ")
 