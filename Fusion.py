import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F
import numbers
from typing import Sequence, Tuple, Union, List
from einops import rearrange


def _as_wavelet(wavelet):
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

def construct_2d_filt(lo, hi) -> torch.Tensor:
    """Construct two dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 2d filters of dimension
            [filt_no, 1, height, width].
            The four filters are ordered ll, lh, hl, hh.
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    # filt = filt.unsqueeze(1)
    return filt


def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
        flip (bool): If true filters are flipped.
        device (torch.device) : PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
               computation. Default: torch.float32.

    Returns:
        tuple: Tuple containing the four filter tensors
        dec_lo, dec_hi, rec_lo, rec_hi

    """
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The length of the used filter.

    Returns:
        tuple: The numbers to attach on the edges of the input.

    """
    # pad to ensure we see all filter positions and for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode.
            Supported modes are "reflect", "zero", "constant" and "periodic".
            Defaults to reflect.

    Returns:
        The padded output tensor.

    """

    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi

        # # initial dec conv
        # self.conv = torch.nn.Conv2d(c1, c2 * 4, kernel_size=dec_filt.shape[-2:], groups=c1, stride=2)
        # self.conv.weight.data = dec_filt
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        # self.convT = nn.ConvTranspose2d(c2 * 4, c1, kernel_size=weight.shape[-2:], groups=c1, stride=2)
        # self.convT.weight = torch.nn.Parameter(rec_filt)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component

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

class StripBlock(nn.Module):    #大条带卷积Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
    def __init__(self, dim, k1=1, k2=7):   #11效果是84.4
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv0(x)
        attn = self.conv_spatial1(x1)
        attn = self.conv_spatial2(self.act(attn))
        attn = self.conv1(self.act(attn)*x1)
        return attn

class CDAF(nn.Module):
    def __init__(self, dim, num_heads,wavelet='haar',initialize=True, LayerNorm_type='BiasFree', ):
        super(MDAF, self).__init__()
        self.num_heads = num_heads

        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet='haar', level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet='haar', level=1)

        self.norm1 = LayerNorm(dim* 3, LayerNorm_type)
        self.norm2 = LayerNorm(dim* 3, LayerNorm_type)
        self.project_out = nn.Conv2d(dim * 3, dim * 3, kernel_size=1)
        self.project_out2 = nn.Conv2d(dim,dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim* 3, dim* 3, (1, 7), padding=(0, 3), groups=dim*3)
        self.conv1_1_2 = nn.Conv2d(dim* 3, dim* 3, (1, 9), padding=(0, 4), groups=dim*3)
        self.conv1_1_3 = nn.Conv2d(dim* 3, dim* 3, (1, 5), padding=(0, 2), groups=dim*3)
        self.conv1_2_1 = nn.Conv2d(dim* 3, dim* 3, (7, 1), padding=(3, 0), groups=dim* 3)
        self.conv1_2_2 = nn.Conv2d(dim* 3, dim* 3, (9, 1), padding=(4, 0), groups=dim* 3)
        self.conv1_2_3 = nn.Conv2d(dim* 3, dim* 3, (5, 1), padding=(2, 0), groups=dim* 3)

        self.conv2_1_1 = nn.Conv2d(dim* 3, dim* 3, (1, 7), padding=(0, 3), groups=dim* 3)
        self.conv2_1_2 = nn.Conv2d(dim* 3, dim* 3, (1, 9), padding=(0, 4), groups=dim* 3)
        self.conv2_1_3 = nn.Conv2d(dim* 3, dim* 3, (1, 5), padding=(0, 2), groups=dim* 3)
        self.conv2_2_1 = nn.Conv2d(dim* 3, dim* 3, (7, 1), padding=(3, 0), groups=dim* 3)
        self.conv2_2_2 = nn.Conv2d(dim* 3, dim* 3, (9, 1), padding=(4, 0), groups=dim* 3)
        self.conv2_2_3 = nn.Conv2d(dim* 3, dim* 3, (5, 1), padding=(2, 0), groups=dim* 3)
        self.act = nn.GELU()
        self.spatial_gating_unit = StripBlock(dim)
        self.ca = ChannelAttention(dim, 10)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.conv1 = nn.Conv2d(dim * 3, dim * 3, 1)

    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        residual_1 = x1
        residual_2 = x2
        ya_1, (yh_1, yv_1, yd_1) = self.wavedec(x1)
        ya_2, (yh_2, yv_2, yd_2) = self.wavedec(x2)

        dec_1 = torch.cat([yh_1, yv_1, yd_1], dim=1)
        dec_2 = torch.cat([yh_2, yv_2, yd_2], dim=1)
        x1 = self.norm1(self.conv1(dec_1))
        x2 = self.norm2(self.conv1(dec_2))


        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        attn_221 = self.conv2_2_1(x2)
        attn_222 = self.conv2_2_2(x2)
        attn_223 = self.conv2_2_3(x2)

        out1 = attn_111 + attn_112 + attn_113 + attn_121 + attn_122 + attn_123
        out2 = attn_211 + attn_212 + attn_213 + attn_221 + attn_222 + attn_223
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v2) + q2
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v1) + q1
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2)
        y_1 = self.spatial_gating_unit(self.act(self.conv(ya_1)))
        y_2 = self.spatial_gating_unit(self.act(self.conv(ya_2)))
        yh_1, yv_1, yd_1 = torch.chunk(out3, 3, dim=1)
        yh_2, yv_2, yd_2 = torch.chunk(out4, 3, dim=1)
        #out = self.project_out(out3) + self.project_out(out4) + x1+x2
        y1 = self.waverec([y_1, (yh_1, yv_1, yd_1)], None)
        y2 = self.waverec([y_2, (yh_2, yv_2, yd_2)], None)
        out = self.project_out2(y1)*residual_1 + self.project_out2(y2)*residual_2


        return out
