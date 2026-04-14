import torch
import torch.nn as nn
from einops import reduce 
from einops import rearrange
class SpatialAttention1D(nn.Module):
    """Spatial Attention for 1D data (sequence-level attention)."""
    def __init__(self):
        super(SpatialAttention1D, self).__init__()
        self.sa = nn.Conv1d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # Average pooling along feature dimension
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along feature dimension
        x2 = torch.cat([x_avg, x_max], dim=1)  # Concatenate along channel dimension
        sattn = self.sa(x2)  # Spatial attention
        return sattn


class ChannelAttention1D(nn.Module):
    """Channel Attention for 1D data."""
    def __init__(self, dim, reduction=8):
        super(ChannelAttention1D, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global average pooling along sequence length
        self.ca = nn.Sequential(
            nn.Conv1d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)  # Global average pooling
        cattn = self.ca(x_gap)  # Channel attention
        return cattn


class PixelAttention1D(nn.Module):
    """Pixel Attention for 1D data."""
    def __init__(self, dim):
        super(PixelAttention1D, self).__init__()
        self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, N, D = x.shape
        x = torch.cat([x, pattn1], dim=2)  # [B, N, 2*D]
        x = x.transpose(1, 2)  # [B, D*2, N]
        pattn2 = self.pa2(x)  # [B, D, N]
        pattn2 = self.sigmoid(pattn2)
        pattn2 = pattn2.transpose(1, 2)  # [B, N, D]
        return pattn2


class CGAFusion1D(nn.Module):
    """Cross-Modal Gated Attention Fusion for 1D data."""
    def __init__(self, dim, reduction=8):
        super(CGAFusion1D, self).__init__()
        self.sa = SpatialAttention1D()  # Sequence-level attention
        self.ca = ChannelAttention1D(dim, reduction)  # Channel-level attention
        self.pa = PixelAttention1D(dim)  # Token-level attention
        self.conv = nn.Conv1d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # Ensure input lengths match via padding (if necessary)
        if x.size(1) != y.size(1):
            max_len = max(x.size(1), y.size(1))
            x = nn.functional.pad(x, (0, 0, 0, max_len - x.size(1)))  # Pad along sequence length
            y = nn.functional.pad(y, (0, 0, 0, max_len - y.size(1)))

        initial = x + y  # Initial fusion
        cattn = self.ca(initial.transpose(1, 2)).transpose(1, 2)  # Channel attention
        sattn = self.sa(initial.transpose(1, 2)).transpose(1, 2)  # Spatial attention
        pattn1 = sattn + cattn  # Combine attentions
        pattn2 = self.sigmoid(self.pa(initial, pattn1))  # Pixel attention
        result = initial + pattn2 * x + (1 - pattn2) * y  # Weighted fusion
        result = self.conv(result.transpose(1, 2)).transpose(1, 2)  # Final output
        return result
class BasicConv1D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True, bn=True, bias=False):
        super(BasicConv1D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool1D(nn.Module):
    def forward(self, x):
        # 输入 [B, C, L]
        max_pool = torch.max(x, 1)[0].unsqueeze(1)  # 最大值池化 [B, 1, L]
        mean_pool = torch.mean(x, 1).unsqueeze(1)  # 平均值池化 [B, 1, L]
        return torch.cat((max_pool, mean_pool), dim=1)  # 拼接后 [B, 2, L]


class AttentionGate1D(nn.Module):
    def __init__(self):
        super(AttentionGate1D, self).__init__()
        kernel_size = 7
        self.compress = ZPool1D()
        self.conv = BasicConv1D(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)  # 压缩 [B, C, L] -> [B, 2, L]
        x_out = self.conv(x_compress)  # 卷积 [B, 2, L] -> [B, 1, L]
        scale = torch.sigmoid(x_out)  # Sigmoid 激活 [B, 1, L]
        return x * scale  # 注意力加权 [B, C, L]


class TripletAttention1D(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention1D, self).__init__()
        self.cw = AttentionGate1D()
        self.hc = AttentionGate1D()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate1D()

    def forward(self, x):
        # 输入 x: [B, L, C], 需要转为 [B, C, L]
        x = x.permute(0, 2, 1).contiguous()  # [B, L, C] -> [B, C, L]
        
        # 通道注意力
        x_out1 = self.cw(x)  # 通道维注意力 [B, C, L]

        # 长度维注意力
        x_perm1 = x.permute(0, 2, 1).contiguous()  # [B, C, L] -> [B, L, C]
        x_out2 = self.hc(x_perm1)  # [B, L, C]
        x_out2 = x_out2.permute(0, 2, 1).contiguous()  # [B, L, C] -> [B, C, L]

        # 如果启用空间注意力
        if not self.no_spatial:
            x_out3 = self.hw(x)  # [B, C, L]
            x_out = (x_out1 + x_out2 + x_out3 ) / 3 + x
        else:
            x_out = (x_out1 + x_out2) / 2 + x

        # 恢复原始维度顺序
        x_out = x_out.permute(0, 2, 1).contiguous()  # [B, C, L] -> [B, L, C]
        return x_out
class SimplifiedLinearAttention(nn.Module):
    """
    参数:
        dim (int): 输入特征维度。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, 可选): 如果为 True，则为查询、键和值添加一个可学习的偏置。
        qk_scale (float | None, 可选): 如果设置，则覆盖默认的 qk scale (head_dim ** -0.5)。
        attn_drop (float, 可选): 注意力权重的丢弃比率。
        proj_drop (float, 可选): 输出的丢弃比率。
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5,device="cuda"):
        super().__init__()
        self.dim = dim  # 输入的特征维度
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个注意力头的特征维度

        self.focusing_factor = focusing_factor  # 聚焦因子
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)  # 用于从 protein 生成查询
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 用于从 drug 生成键和值
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的 dropout
        self.proj = nn.Linear(dim, dim)  # 输出的线性变换
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的 dropout

        self.softmax = nn.Softmax(dim=-1)  # Softmax 计算权重

        # 深度可分离卷积 (Depthwise Convolution)，用于捕捉局部特征（改为 1D 卷积）
        self.dwc = nn.Conv1d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.length_proj = nn.Linear(275, 816)
        # self.length_proj = nn.Linear(282, 59)
        # self.length_proj = nn.Linear(287, 66)
        # self.length_proj = nn.Linear(290, 69)
        # self.length_proj = nn.Linear(285,62)
        # 打印初始化信息，便于调试
        print('Linear Attention dim={} heads={} kernel={}'.format(dim, num_heads, kernel_size))

    # 前向传播函数
    def forward(self, protein, drug, mask=None):
        """
        参数:
            protein: 用于生成查询，形状为 (B, N1, C)，其中 N1 是 protein 的序列长度。
            drug: 用于生成键和值，形状为 (B, N2, C)，其中 N2 是 drug 的序列长度。
            mask: (0/-inf) 掩码，形状为 (B, N1, N2) 或 None
        """
        B, N1, C = protein.shape  # protein 的维度信息
        _, N2, _ = drug.shape    # drug 的维度信息

        # 分别计算查询 (q) 和键值 (kv)
        q = self.q_proj(protein).reshape(B, N1, self.num_heads, C // self.num_heads)
        q = rearrange(q, "b n h c -> (b h) n c")  # (B, N1, num_heads, head_dim) -> (B*num_heads, N1, head_dim)

        kv = self.kv_proj(drug).reshape(B, N2, 2, self.num_heads, C // self.num_heads)
        k, v = kv.unbind(dim=2)  # (B, N2, num_heads, head_dim)
        k, v = (rearrange(x, "b n h c -> (b h) n c") for x in (k, v))  # 分别调整 k 和 v 的形状

        # 使用混合精度训练，保证数值精度并优化计算效率
        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            # 计算键和值的乘积：k * v
            kv_product = torch.einsum("b i c, b i d -> b c d", k, v)  # (B*num_heads, head_dim, head_dim)

            # 计算查询与键-值乘积的点积：q * (k*v)
            x = torch.einsum("b i c, b c d -> b i d", q, kv_product)  # (B*num_heads, N1, head_dim)

        # 处理深度卷积 (1D 卷积)
        feature_map = rearrange(v, "b n c -> b c n")  # 调整为 (B, C, N2)
        feature_map = rearrange(self.dwc(feature_map), "b c n -> b n c")  # 1D 卷积后重排回 (B, N2, C)
        # print(x.shape)
        # print(feature_map.shape)
        feature_map = rearrange(feature_map, "b n c -> b c n")  # (B, N2, C) -> (B, C, N2)
        feature_map = self.length_proj(feature_map)  # (B, C, N2) -> (B, C, N1)
        feature_map = rearrange(feature_map, "b c n -> b n c")  # (B, C, N1) -> (B, N1, C)
        x = x +feature_map# 将卷积结果与注意力结果相加
        # print(feature_map.shape)
        # print(x.shape)

        # 恢复多头后的形状
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)  # 线性变换
        x = self.proj_drop(x)  # 输出 dropout

        return x  # 返回最终输出

class Conv1d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv1d(in_channels, out_channels, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm1d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = nn.Conv1d(
            in_channels=w.size(1) * self.c.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class SimplifiedLinearAttention1(nn.Module):
    """
    参数:
        dim (int): 输入特征维度。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, 可选): 如果为 True，则为查询、键和值添加一个可学习的偏置。
        qk_scale (float | None, 可选): 如果设置，则覆盖默认的 qk scale (head_dim ** -0.5)。
        attn_drop (float, 可选): 注意力权重的丢弃比率。
        proj_drop (float, 可选): 输出的丢弃比率。
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5,device="cuda"):
        super().__init__()
        self.dim = dim  # 输入的特征维度
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个注意力头的特征维度

        self.focusing_factor = focusing_factor  # 聚焦因子
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)  # 用于从 protein 生成查询
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 用于从 drug 生成键和值
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的 dropout
        self.proj = nn.Linear(dim, dim)  # 输出的线性变换
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的 dropout

        self.softmax = nn.Softmax(dim=-1)  # Softmax 计算权重

        # 深度可分离卷积 (Depthwise Convolution)，用于捕捉局部特征（改为 1D 卷积）
        self.dwc = nn.Conv1d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.length_proj = nn.Linear(816, 275)
        # self.length_proj = nn.Linear(59, 282)
        # self.length_proj = nn.Linear(66, 287)
        # self.length_proj = nn.Linear(69, 290)
        # self.length_proj = nn.Linear(62, 285)
        # 打印初始化信息，便于调试
        print('Linear Attention dim={} heads={} kernel={}'.format(dim, num_heads, kernel_size))

    # 前向传播函数
    def forward(self, protein, drug, mask=None):
        """
        参数:
            protein: 用于生成查询，形状为 (B, N1, C)，其中 N1 是 protein 的序列长度。
            drug: 用于生成键和值，形状为 (B, N2, C)，其中 N2 是 drug 的序列长度。
            mask: (0/-inf) 掩码，形状为 (B, N1, N2) 或 None
        """
        B, N1, C = protein.shape  # protein 的维度信息
        _, N2, _ = drug.shape    # drug 的维度信息

        # 分别计算查询 (q) 和键值 (kv)
        q = self.q_proj(protein).reshape(B, N1, self.num_heads, C // self.num_heads)
        q = rearrange(q, "b n h c -> (b h) n c")  # (B, N1, num_heads, head_dim) -> (B*num_heads, N1, head_dim)

        kv = self.kv_proj(drug).reshape(B, N2, 2, self.num_heads, C // self.num_heads)
        k, v = kv.unbind(dim=2)  # (B, N2, num_heads, head_dim)
        k, v = (rearrange(x, "b n h c -> (b h) n c") for x in (k, v))  # 分别调整 k 和 v 的形状

        # 使用混合精度训练，保证数值精度并优化计算效率
        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            # 计算键和值的乘积：k * v
            kv_product = torch.einsum("b i c, b i d -> b c d", k, v)  # (B*num_heads, head_dim, head_dim)

            # 计算查询与键-值乘积的点积：q * (k*v)
            x = torch.einsum("b i c, b c d -> b i d", q, kv_product)  # (B*num_heads, N1, head_dim)

        # 处理深度卷积 (1D 卷积)
        feature_map = rearrange(v, "b n c -> b c n")  # 调整为 (B, C, N2)
        feature_map = rearrange(self.dwc(feature_map), "b c n -> b n c")  # 1D 卷积后重排回 (B, N2, C)
        # print(x.shape)
        # print(feature_map.shape)
        feature_map = rearrange(feature_map, "b n c -> b c n")  # (B, N2, C) -> (B, C, N2)
        feature_map = self.length_proj(feature_map)  # (B, C, N2) -> (B, C, N1)
        feature_map = rearrange(feature_map, "b c n -> b n c")  # (B, C, N1) -> (B, N1, C)
        x = x +feature_map    # 将卷积结果与注意力结果相加

        # 恢复多头后的形状
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)  # 线性变换
        x = self.proj_drop(x)  # 输出 dropout

        return x  # 返回最终输出


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out

class IntentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = SimplifiedLinearAttention(dim=dim, num_heads=num_heads, qkv_bias=kqv_bias, device=device)
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        x_t = x.permute(0, 2, 1)
        x = self.norm_layer(x)
 
        att = self.attn(x, q)#[64, 837, 128]
        # print("x_t:",x_t.shape)
        att_map = self.softmax(att)
        # print(" att_map:",att.shape)
        out = self.beta * x_t @ att_map
        return out
class IntentionBlock1(nn.Module):

    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock1, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = SimplifiedLinearAttention1(dim=dim, num_heads=num_heads, qkv_bias=kqv_bias, device=device)
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        
        x_t = x.permute(0, 2, 1)
        x = self.norm_layer(x)
 
        att = self.attn(x, q)#[64, 837, 128]
        # print("x_t:",x_t.shape)
        att_map = self.softmax(att)
        # print(" att_map:",att.shape)
        out = self.beta * x_t @ att_map
        return out

class BiSLABlock(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(BiSLABlock, self).__init__()

        self.layer = layer
        self.drug_intention = nn.ModuleList([
            IntentionBlock1(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        #self attention

        self.block = TripletAttention1D()
        self.block_ = TripletAttention1D()

  


    def forward(self, drug, protein):
        # print(drug)
        # print(protein)
        drug = self.block(drug)
        protein = self.block_(protein)

        for i in range(self.layer):
            temp_p = self.drug_intention[i](drug, protein)
            temp_d = self.protein_intention[i](protein, drug)
            drug, protein = temp_d, temp_p
        # ALL = torch.cat([drug, protein], axis=-2)
        # f = self.SEN_drug(temp_d,temp_p)
        # f = reduce(f, 'B H W -> B H', 'max')
        # ALL = ALL.unsqueeze(-1)
        # ALL = self.SEN_drug(ALL)
        # ALL = ALL.squeeze(-1)
        # f = reduce(ALL, 'B H W -> B H', 'max')
        
        # print(protein)
        v_d = reduce(drug, 'B H W -> B H', 'max')
        v_p = reduce(protein, 'B H W -> B H', 'max')

        f = torch.cat((v_d, v_p), dim=1)
        return f, drug, protein, None