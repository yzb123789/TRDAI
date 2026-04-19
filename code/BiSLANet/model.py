import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from ACmix import ACmix
from BiSLAModule import BiSLABlock
import math
import joblib
from einops import reduce
from torch.nn.utils.weight_norm import weight_norm
class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,  stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.w_q = nn.Linear(133, 32)
        self.w_k = nn.Linear(133, 32)
        self.w_v = nn.Linear(133, 32)

        self.dense = nn.Linear(32, 133)
        self.BatchNorm = nn.BatchNorm1d(133, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)

        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.BatchNorm(hidden_states + fg_hiddens)

        return hidden_states
def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss
class BilinearPooling(nn.Module):
    def __init__(self, in_channels, out_channels, c_m, c_n):
        super().__init__()

        self.convA = nn.Conv1d(in_channels, c_m, kernel_size=1, stride=1, padding=0)
        self.convB = nn.Conv1d(in_channels, c_n, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(c_m, out_channels, bias=True)

    def forward(self, x):
        '''
        x: (batch, channels, seq_len)
        A: (batch, c_m, seq_len)
        B: (batch, c_n, seq_len)
        att_maps.permute(0, 2, 1): (batch, seq_len, c_n)
        global_descriptors: (batch, c_m, c_n)
        '''        
        A = self.convA(x) 
        B = self.convB(x)
        att_maps = F.softmax(B, dim=-1)
        global_descriptors = torch.bmm(A, att_maps.permute(0, 2, 1))
        global_descriptor = torch.mean(global_descriptors, dim=-1)
        out = self.linear(global_descriptor).unsqueeze(1)

        return out
class MutualAttentation(nn.Module):
    def __init__(self, in_channels, att_size, c_m, c_n):
        super().__init__()
        self.bipool = BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linearS = nn.Linear(in_channels, att_size)
        self.linearT = nn.Linear(in_channels, att_size)
    
    def forward(self, source, target):
        '''
        source: (batch, channels, seq_len)
        target: (batch, channels, seq_len)
        global_descriptor: (batch, 1, channels)
        '''
        global_descriptor = self.bipool(source)
        target_org = target
        target = self.linearT(target.permute(0, 2, 1)).permute(0, 2, 1)
        global_descriptor = self.linearS(global_descriptor)
        att_maps = torch.bmm(global_descriptor, target)
        att_maps = torch.sigmoid(att_maps)
        out_target = torch.add(target_org, torch.mul(target_org, att_maps))
        out_target = F.relu(out_target)

        return out_target

class InceptionDWConv1d(nn.Module):
    """ Inception depthwise convolution for 1D data """

    def __init__(self, in_features, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_features * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv1d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv1d(gc, gc, kernel_size=band_kernel_size, padding=band_kernel_size // 2, groups=gc)
        self.dwconv_h = nn.Conv1d(gc, gc, kernel_size=band_kernel_size, padding=band_kernel_size // 2, groups=gc)
        self.split_indexes = (in_features - 3 * gc, gc, gc, gc)

    def forward(self, x):
        # Split the input along the feature dimension
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class BiSLANet(nn.Module):
    def __init__(self, device='cuda', **config):
        super(BiSLANet, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.cross_intention = BiSLABlock(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)
        self.mlp_classifier = MLPDecoder(768, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
       
        # self.hidden_size = 128
        self.linear = nn.Sequential(
            nn.Linear(133, 128),  
            nn.ReLU()
        )
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)
        prot_filter_size = [4, 8, 12]
        drug_filter_size = [4, 6, 8]
        self.prot_mut_att1 = MutualAttentation(128, 128 ,128, 8)
        self.drug_mut_att1 = MutualAttentation(128, 128, 128, 8)

        self.prot_mut_att2 = MutualAttentation(128*2, 128, 128, 8)
        self.drug_mut_att2 = MutualAttentation(128*2, 128, 128, 8)

        self.prot_mut_att3 = MutualAttentation(128*3, 128, 128, 8)
        self.drug_mut_att3 = MutualAttentation(128*3, 128, 128, 8)

        self.prot_conv1 = Conv1dReLU(128, 128, prot_filter_size[0])
        self.prot_conv2 = Conv1dReLU(128, 128 * 2, prot_filter_size[1])
        self.prot_conv3 = Conv1dReLU(128 * 2, 128 * 3, prot_filter_size[2])
        # self.prot_pool = nn.AdaptiveMaxPool1d(1)


        self.drug_conv1 = Conv1dReLU(128, 128, drug_filter_size[0])
        self.drug_conv2 = Conv1dReLU(128, 128 * 2, drug_filter_size[1])
        self.drug_conv3 = Conv1dReLU(128 * 2, 128 * 3, drug_filter_size[2])
        # self.drug_pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, bg_d, v_p,f_fgs, mode="train"):
        fg_states = f_fgs

        fg_states = self.linear(fg_states)

#         cls_hiddens = torch.mean(fg_states, dim=1)

#         cls_hiddens = cls_hiddens.unsqueeze(1).expand(-1, 290, -1)

        
 
        drug_x= self.drug_extractor(bg_d,f_fgs)#v_d.shape(64, 290, 128)
        # print(drug_x)
        prot_x = self.protein_extractor(v_p)#v_p.shape:(64, 1200, 128)
        

        drug_x = drug_x + self.alpha * fg_states
        # print(drug_x)
        # print(prot_x)
        # print(drug_x.shape)
        # print(prot_x.shape)
        prot_x = prot_x.permute(0, 2, 1)
        drug_x = drug_x.permute(0, 2, 1)

        prot_x = self.prot_conv1(prot_x)
        drug_x = self.drug_conv1(drug_x)
        prot_x_g = self.prot_mut_att1(drug_x, prot_x)
        drug_x_g = self.drug_mut_att1(prot_x, drug_x)
        # print(drug_x)
        # print(drug_x.shape)

        prot_x = self.prot_conv2(prot_x_g)
        drug_x = self.drug_conv2(drug_x_g)
        prot_x_g = self.prot_mut_att2(drug_x, prot_x)
        drug_x_g = self.drug_mut_att2(prot_x, drug_x)

        prot_x = self.prot_conv3(prot_x_g)
        drug_x = self.drug_conv3(drug_x_g)
        prot_x_g = self.prot_mut_att3(drug_x, prot_x)
        drug_x_g = self.drug_mut_att3(prot_x, drug_x)
        # print(prot_x_g.shape)
        # print(drug_x_g.shape)
        
        
        prot_x_g = prot_x_g.permute(0, 2, 1)
        drug_x_g = drug_x_g.permute(0, 2, 1)
        # print(prot_x_g)
        # print(drug_x_g.shape)       
        # print(drug_x_g)
        # print(drug_x_g.shape)
        f, v_d, v_p, att = self.cross_intention(drug=drug_x_g, protein=prot_x_g)
        # print(v_d)
        # print(v_d.shape)
        
        # prot_x = self.prot_pool(prot_x_g).squeeze(-1)
        # drug_x = self.drug_pool(drug_x_g).squeeze(-1)
        # f = torch.cat([prot_x, drug_x], dim=-1)
        # print(f)
        # print(f.shape)
        score = self.mlp_classifier(f)
        
        
        if mode == "train":
            return drug_x, drug_x, score, score
        elif mode == "eval":
            return drug_x, drug_x, score, score

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        # self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.alpha.data.fill_(0.3)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph,fg_out):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        # node_feats = node_feats + fg_out
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        # print(node_feats.shape)
        return node_feats


class ProteinACmix(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinACmix, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 128), 
            nn.ReLU()       
        )
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.block = InceptionDWConv1d(128)  

        
    def forward(self, v):
        # print(v)
        v = v.unsqueeze(-1)
        v = self.linear(v.float())
        v = v.transpose(2, 1)#64*128*837
        # print(v.shape)
        v = self.bn1(F.relu(self.block(v)))


        v = v.view(v.size(0), v.size(2), -1)
        return v

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        # print(x)
        x = self.bn1(F.relu(self.fc1(x)))
        # print(x)
        x = self.bn2(F.relu(self.fc2(x)))
        # print(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        # print(x)
        # print(x.shape)
        return x