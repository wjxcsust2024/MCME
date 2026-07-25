import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.layers import DropPath
import functools
import math
from .ecb import SeqConv3x3
from .trans_form import dim3_4, dim4_3
from .Fusion import MDAF
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 3, padding=1),
            nn.Sigmoid())
        self.attention_1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.attention(x)
        y2 = self.attention_1(x)
        out = self.sigmoid(y1+y2)*x
        return out

class SLKA(nn.Module):
    def __init__(self, n_feats, k=21, d=3, shrink=0.5, scale=2):
        super().__init__()
        f = int(n_feats*shrink)
        self.head = nn.Conv2d(n_feats, f, 1)
        self.proj_2 = nn.Conv2d(f, f, kernel_size=1)
        self.activation = nn.GELU()
        self.LKA = nn.Sequential(
            nn.Conv2d(f, f, 1),
            conv_layer(f, f, 2 * d - 1, groups=f),
            self.activation,
            conv_layer(f, f, k // d, dilation=1, groups=f),
            self.activation,
        )
        self.conv = nn.Sequential(nn.Conv2d(f, f, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=f),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(f, f, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=f))
        self.conv2 = nn.Sequential(nn.Conv2d(f, f, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=f),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(f, f, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=f))
        self.conv3 = nn.Conv2d(f, f, kernel_size=1)
        self.ca = ChannelAttention(f, 10)
        self.VSS = SS2D(f)
        self.spatial_gating_unit = StripBlock(f)
        self.tail = nn.Conv2d(f, n_feats, 1)
        self.scale = scale

    def forward(self, x):
        b,n,c = x.shape
        h=w=int(n**0.5)
        x = x.reshape(b,h,w,c).permute(0,3,1,2)
        c1 = self.conv(self.head(x))
        c2 = F.max_pool2d(c1, kernel_size=self.scale * 2 + 1, stride=self.scale)
        c2 = self.VSS(self.LKA(c2))+self.spatial_gating_unit(c2)
        c3 = F.interpolate(c2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        a = self.tail(c3+self.ca(self.spatial_gating_unit(c1)))
        a = F.sigmoid(a)
        return (x*a).permute(0,2,3,1).reshape(b,n,c)

class StripBlock(nn.Module):    #大条带卷积Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
    def __init__(self, dim, k1=1, k2=7):   #11效果是84.4
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv0(x)
        attn = self.conv_spatial1(x1)
        attn = self.conv_spatial2(self.act(attn))
        #attn = self.conv1(attn)

        return attn

class SS2D(nn.Module):  #SCSegamba: Lightweight Structure-Aware Vision Mamba for Crack Segmentation in Structures
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv3d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv+2,
            padding=(d_conv + 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv3d(self.act(self.conv2d(x)))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out.permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MLP, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Sequential( nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,groups=hidden_features * 2, bias=bias),nn.ReLU(inplace=True))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.conv = nn.Sequential(nn.Conv2d(hidden_features, hidden_features, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2),groups=hidden_features),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0),groups=hidden_features))
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', hidden_features, hidden_features, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', hidden_features, hidden_features, -1)
        self.McConv = MultiScaleDWConv(hidden_features)

    def forward(self, x):
        x_1 = dim3_4(x)
        x = self.project_in(x_1)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x2 = self.conv1x1_sbx(x2) + self.conv1x1_sby(x2) + self.McConv(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return dim4_3(x)

class StBlock(nn.Module):    #大条带卷积Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
    def __init__(self, dim, k1=1, k2=7):   #11效果是84.4
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv0(x)
        attn = self.conv_spatial1(x1)
        attn = self.conv_spatial2(self.act(attn))
        attn = self.conv1(self.act(attn))

        return attn
class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Sequential(nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels),
                                 nn.GELU())
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, patch_size, stride, padding))    # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):  #RGB图像的编码器模块
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        #self.attn = Attention(dim, head, sr_ratio)
        self.attn = SLKA(dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim,2,False)

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1,-2)
        _, _, Nk, Ck  = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))
        
        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x 


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


cmnext_settings = {
    # 'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    # 'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 4, 3]],   #后面这一排数据是对应编码器每层的模块数量
    #'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    # 'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class CMNeXt(nn.Module):    #主网络
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in cmnext_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = cmnext_settings[model_name]
        extra_depths = depths 
        self.modals = modals[1:] if len(modals)>1 else []  
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        norm_cfg = dict(type='BN', requires_grad=True)

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2)
   
        if self.num_modals > 0:
            self.extra_downsample_layers = nn.ModuleList([
                PatchEmbedParallel(3, embed_dims[0], 7, 4, 7//2, self.num_modals),
                *[PatchEmbedParallel(embed_dims[i], embed_dims[i+1], 3, 2, 3//2, self.num_modals) for i in range(3)]
            ])
        if self.num_modals > 1:
            self.extra_score_predictor = nn.ModuleList([PredictorConv(embed_dims[i], self.num_modals) for i in range(len(depths))])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        if self.num_modals > 0:
            #self.extra_block1 = nn.ModuleList([MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[0])]) # --- MSPABlock
            self.extra_block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
            self.extra_norm1 = ConvLayerNorm(embed_dims[0])
            
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        if self.num_modals > 0:
            #self.extra_block2 = nn.ModuleList([MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[1])])
            self.extra_block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
            self.extra_norm2 = ConvLayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        if self.num_modals > 0:
            #self.extra_block3 = nn.ModuleList([MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[2])])
            self.extra_block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
            self.extra_norm3 = ConvLayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        if self.num_modals > 0:
            #self.extra_block4 = nn.ModuleList([MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[3])])
            self.extra_block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
            self.extra_norm4 = ConvLayerNorm(embed_dims[3])

        if self.num_modals > 0:
            #num_heads = [1,2,5,8]
            self.MDAFs = nn.ModuleList([
                MDAF(dim=embed_dims[0], num_heads=8,LayerNorm_type='WithBias'),
                MDAF(dim=embed_dims[1], num_heads=8,LayerNorm_type='WithBias'),
                MDAF(dim=embed_dims[2], num_heads=8,LayerNorm_type='WithBias'),
                MDAF(dim=embed_dims[3], num_heads=8,LayerNorm_type='WithBias')
            ])
    def tokenselect(self, x_ext, module):    
        x_scores = module(x_ext) 
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        x_f = functools.reduce(torch.max, x_ext)
        return x_f
     
    def forward(self, x: list) -> list:
        x_cam = x[0]        
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]
        outs = []
        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[0](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[0]) if self.num_modals > 1 else x_ext[0] 
            for blk in self.extra_block1:
                x_f = x_f.permute(0, 2, 3, 1).reshape(B, H * W, -1)
                x_f = blk(x_f, H, W)
                x_f = x_f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x1_f = self.extra_norm1(x_f)
            # x1_cam, x1_f = self.FRMs[0](x1_cam, x1_f)
            # x_fused = self.FFMs[0](x1_cam, x1_f)
            x_fused = self.MDAFs[0](x1_cam, x1_f)
            x1_cam = x_fused
            x1_f = x_fused
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x1_f for x_ in x_ext] if self.num_modals > 1 else [x1_f]
        else:
            outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[1](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[1]) if self.num_modals > 1 else x_ext[0] 
            for blk in self.extra_block2:
                x_f = x_f.permute(0, 2, 3, 1).reshape(B, H * W, -1)
                x_f = blk(x_f, H, W)
                x_f = x_f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
            x2_f = self.extra_norm2(x_f)
            # x2_cam, x2_f = self.FRMs[1](x2_cam, x2_f)
            # x_fused = self.FFMs[1](x2_cam, x2_f)
            x_fused = self.MDAFs[1](x2_cam, x2_f)
            x2_cam = x_fused
            x2_f = x_fused
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x2_f for x_ in x_ext] if self.num_modals > 1 else [x2_f]
        else:
            outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[2](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[2]) if self.num_modals > 1 else x_ext[0] 
            for blk in self.extra_block3:
                x_f = x_f.permute(0, 2, 3, 1).reshape(B, H * W, -1)
                x_f = blk(x_f, H, W)
                x_f = x_f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
            x3_f = self.extra_norm3(x_f)
            # x3_cam, x3_f = self.FRMs[2](x3_cam, x3_f)
            # x_fused = self.FFMs[2](x3_cam, x3_f)
            x_fused = self.MDAFs[2](x3_cam, x3_f)
            x3_cam = x_fused
            x3_f = x_fused
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x3_f for x_ in x_ext] if self.num_modals > 1 else [x3_f]
        else:
            outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[3](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[3]) if self.num_modals > 1 else x_ext[0] 
            for blk in self.extra_block4:
                x_f = x_f.permute(0, 2, 3, 1).reshape(B, H * W, -1)
                x_f = blk(x_f, H, W)
                x_f = x_f.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
            x4_f = self.extra_norm4(x_f)
            # x4_cam, x4_f = self.FRMs[3](x4_cam, x4_f)
            # x_fused = self.FFMs[3](x4_cam, x4_f)
            x_fused = self.MDAFs[3](x4_cam, x4_f)
            outs.append(x_fused)
        else:
            outs.append(x4_cam)

        return outs


if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    model = CMNeXt('B2', modals)
    outs = model(x)
    for y in outs:
        print(y.shape)

