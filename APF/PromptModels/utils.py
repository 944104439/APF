import math

import torch
import torch.nn as nn

# from PromptModels.pointnet2_utils import sample_and_group
from timm.models.layers import DropPath, Mlp


class PointCls(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PointCls, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channel,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256,out_channel)
        )
    def forward(self, x):
        x = self.mlp_head(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # every head_dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # 一个变成3个 一个token -- > qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, temp=1.0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q's dim is (b,heads,n,dim) , k's dim is (b,heads,dim,n)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # except matrix,there are indexs
        # softmax in last dimension, (b,heads,n,n)
        attn = (attn / temp).softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = b,heads,n,n  v = b,heads,n,dim ---- > b,heads,n,dim
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Adapter(nn.Module):
    def __init__(self, config=None, d_model=768, bottleneck=64, dropout=0.0, init_option="lora",
                 adapter_scalar="learnable_scalar",adapter_layernorm_option="in",elite = True):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        if elite:
            bottleneck = bottleneck
        else:
            bottleneck = bottleneck * 4

        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Block(nn.Module):

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # norm_layer 是一个token的layer？
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaptmlp = Adapter(dropout=0.1, d_model=dim)

    def forward(self, x,temp=1.0):
        # b,n,c
        x = x + self.drop_path(self.attn(self.norm1(x), temp=temp))
        residual = x
        # mlp and adapt_mlp
        adapt_x = self.adaptmlp(x)
        x = self.drop_path(self.mlp(self.norm2(x)))
        # parallel
        x = x + adapt_x + residual
        return x
