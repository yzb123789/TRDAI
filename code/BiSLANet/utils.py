import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(batch):
    # 从批次数据中解包
    # print("Length of batch:", len(batch))
    drug_graphs, protein_features,v_s,atom_num,  labels= zip(*batch)
    
    # 批处理药物图数据
    drug_graphs = dgl.batch(drug_graphs)
    # print(drug_graphs)
    # 批处理蛋白质特征（假设每个蛋白质为 837 维的一维向量）
    protein_features = torch.stack([p.clone().detach().float() for p in protein_features])
    # print(protein_features)
    # print(protein_features.shape)
    v_s_ = torch.stack([s.clone().detach().float() for s in v_s])
    cls_hiddens = torch.mean(v_s_, dim=1)
    
    temp1 = cls_hiddens[0].unsqueeze(0)
    out = torch.repeat_interleave(temp1, torch.tensor(atom_num[0][0]), dim=0)
    # print(out.shape)
    # print(atom_num)
    temp = torch.zeros(290-atom_num[0][0], 133)
    # print(temp.shape)
    # print(v_s_.shape[0])
    out = torch.cat((out, temp), dim=0)    
    out = out.unsqueeze(0)
    for i in range(1,v_s_.shape[0]):
        temp1 = cls_hiddens[0].unsqueeze(0)
        fg_hiddens = torch.repeat_interleave(temp1, torch.tensor(atom_num[i][0]), dim=0)
        # print(cls_hiddens.shape)
        temp = torch.zeros(290-atom_num[i][0], 133)
        # print(temp.shape)
        temp2 = torch.cat((fg_hiddens, temp), dim=0)
        temp2 = temp2.unsqueeze(0)
        out = torch.cat((out, temp2), dim=0)
    # print(out.shape)
    
    # print(v_s_)
    # print(v_s_.shape)
    # nun = torch.stack([torch.tensor(atom, dtype=torch.float32) for atom in nun_atoms])
    # 批处理标签
    labels = torch.tensor(labels, dtype=torch.float32)
    # print("111",nun_atoms) 
    
    return drug_graphs, protein_features,out ,labels


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in /"
                f"sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
