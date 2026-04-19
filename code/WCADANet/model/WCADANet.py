import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import model.sparsemax as sparsemax

def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return
class ChannelAttention(nn.Module):
    def __init__(self, input_dim, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Squeeze: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: Fully Connected Layers (with reduction)
        self.fc1 = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.fc2 = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        avg_out = self.avg_pool(x).view(x.size(0), -1)  # [B, D]
        
        # Excitation
        out = self.fc1(avg_out)  # [B, D//r]
        out = F.relu(out)
        out = self.fc2(out)  # [B, D]
        out = self.sigmoid(out).view(x.size(0), x.size(1), 1)  # [B, D, 1]
        
        # Apply attention (broadcast the attention map to each time step)
        return x * out

class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """
    def __init__(self, input_dim, virtual_batch_size=512):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training == True:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

class LearnableLocality(nn.Module):

    def __init__(self, input_dim, k):
        super(LearnableLocality, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.rand(k, input_dim)))
        self.smax = sparsemax.Entmax15(dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)  # [B, k, D]
        return masked_x

class WCA_AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, k, virtual_batch_size, bias=True):
        super(WCA_AbstractLayer, self).__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, k=k)
        self.fc = nn.Conv1d(base_input_dim * k, 2 * k * base_output_dim, kernel_size=1, groups=k, bias=bias)
        initialize_glu(self.fc, input_dim=base_input_dim * k, output_dim=2 * k * base_output_dim)
        self.bn = GBN(2 * base_output_dim * k, virtual_batch_size)
        self.k = k
        self.base_output_dim = base_output_dim
        
        self.channel_attention = ChannelAttention(input_dim=2 * base_output_dim * k)

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, k, D]
        x = self.fc(x.contiguous().view(b, -1, 1))  # [B, k, D] -> [B, k * D, 1] -> [B, k * (2 * D'), 1]
        x = self.bn(x)
        

        x = self.channel_attention(x)  
        
        chunks = x.chunk(self.k, 1)  # k * [B, 2 * D', 1]
        x = sum([F.relu(torch.sigmoid(x_[:, :self.base_output_dim, :]) * x_[:, self.base_output_dim:, :]) for x_ in chunks])  # k * [B, D', 1] -> [B, D', 1]
        
        return x.squeeze(-1)
    
class WCA_Block(nn.Module):
    def __init__(self, input_dim, base_outdim, k, virtual_batch_size, fix_input_dim, drop_rate):
        super(WCA_Block, self).__init__()
        self.conv1 = WCA_AbstractLayer(input_dim, base_outdim // 2, k, virtual_batch_size)
        self.conv2 = WCA_AbstractLayer(base_outdim // 2, base_outdim, k, virtual_batch_size)

        self.downsample = nn.Sequential(
            nn.Dropout(drop_rate),
            WCA_AbstractLayer(fix_input_dim, base_outdim, k, virtual_batch_size)
        )
        
        # Learnable weight for residual connection
        self.weight = nn.Parameter(torch.ones(1))  # Learnable weight to scale the residual

    def forward(self, x, pre_out=None):
        if pre_out is None:
            pre_out = x
        out = self.conv1(pre_out)
        out = self.conv2(out)
        identity = self.downsample(x)
        # Weighted residual connection
        out = self.weight * out + (1 - self.weight) * identity  # Apply the weighted sum
        return F.leaky_relu(out, 0.01)



class WCADANet(nn.Module):
    def __init__(self, input_dim, num_classes, layer_num, base_outdim, k, virtual_batch_size, drop_rate=0.1):
        super(WCADANet, self).__init__()
        params = {'base_outdim': base_outdim, 'k': k, 'virtual_batch_size': virtual_batch_size,
                  'fix_input_dim': input_dim, 'drop_rate': drop_rate}
        self.init_layer = WCA_Block(input_dim, **params)
        self.lay_num = layer_num
        self.layer = nn.ModuleList()
        for i in range((layer_num // 2) - 1):
            self.layer.append(WCA_Block(base_outdim, **params))
        self.drop = nn.Dropout(0.1)

        self.fc = nn.Sequential(nn.Linear(base_outdim, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 1))
    def forward(self, x):
        
        out = self.init_layer(x)
        for i in range(len(self.layer)):
            out = self.layer[i](x, out)
        out = self.drop(out)
        out = self.fc(out)
        return out