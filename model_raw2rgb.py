import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
import random

import torch.optim as optim
import torchvision.transforms.functional as TF
from einops import rearrange
import os


 
 
class RGGBtoRGB(nn.Module):
    def __init__(self,kernel_size=3,upsample_choose = 'conv',inter_nums=4):
        super(RGGBtoRGB, self).__init__()
        self.inter_nums=inter_nums
        # 使用1x1卷积将4个通道转换为12个通道
        self.pointwise_conv = nn.Conv2d(4, 12, kernel_size=1)
        
        # 使用PixelShuffle实现分辨率提升
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        #self.bn1 = nn.BatchNorm2d(12)  # Batch Normalization after pointwise_conv
        #self.bn2 = nn.BatchNorm2d(12)  # Batch Normalization after pointwise_conv
        #self.dropout1 = nn.Dropout(0.1)  # Dropout after pointwise_conv
        # 定义不同大小的卷积核，并在每个卷积后使用PixelShuffle

        self.conv_6x6 = nn.Conv2d(3, 12, kernel_size=6, stride=2, padding=2, padding_mode='reflect')
        self.conv_10x10 = nn.Conv2d(3, 12, kernel_size=10, stride=2, padding=4, padding_mode='reflect')
        self.conv_14x14 = nn.Conv2d(3, 12, kernel_size=14, stride=2, padding=6, padding_mode='reflect')
        self.conv_18x18 = nn.Conv2d(3, 12, kernel_size=18, stride=2, padding=8, padding_mode='reflect')

    def forward(self, x):
        # 通过point-wise卷积增加通道数
        x = self.pointwise_conv(x)
        #x = self.bn1(x)        # Batch Normalization
        #x = self.dropout1(x)  # Dropout
        # 使用PixelShuffle重新排列通道并提高分辨率
        x = self.pixel_shuffle(x)
        #print(x.shape)
        # 使用不同大小的卷积核进行级联卷积操作，并在每次卷积后使用PixelShuffle
        #print(x.shape)
        x = self.conv_6x6(x)
        #x = self.bn2(x)  # Batch Normalization after pointwise_conv
        #x = self.dropout1(x)  # Dropout
        x = self.pixel_shuffle(x)

        #print(x.shape)
        x = self.conv_10x10(x)
        #x = self.dropout1(x)  # Dropout
        x = self.pixel_shuffle(x)
        # print(x.shape)
        x = self.conv_14x14(x)
        #x = self.dropout1(x)  # Dropout
        x = self.pixel_shuffle(x)
        x = self.conv_18x18(x)
        #x = self.dropout1(x)  # Dropout
        x = self.pixel_shuffle(x)
        #print(x.shape)
        return x
 



class Affine_(nn.Module):
    def __init__(self, channels=3, init_method='kaiming'):
        super(Affine_, self).__init__()
        # 初始化权重和偏置
        self.weights1 = nn.Parameter(torch.Tensor(channels))
        self.bias = nn.Parameter(torch.Tensor(channels))
        self.acti = nn.ReLU(inplace=True)
        if init_method == 'kaiming':
            nn.init.ones_(self.weights1)  # 初始化为1，表示没有缩放
            nn.init.zeros_(self.bias)
        elif init_method == 'xavier':
            nn.init.ones_(self.weights1)
            nn.init.zeros_(self.bias)
        else:
            nn.init.ones_(self.weights1)
            nn.init.zeros_(self.bias)
    def forward(self, x):
        # 使用权重和偏置调整输入
        x_transformed = x * self.weights1.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        x_transformed =self.acti(x_transformed)
        return x_transformed

class Affine(nn.Module):
    def __init__(self, num_layers=1, init_method='kaiming'):
        super(Affine, self).__init__()
        layers =[]
        for _ in range(num_layers):
            layers.append(Affine_(init_method=init_method))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)+x




class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=1):
        super(ChannelAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(channels, channels//reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction_ratio, channels, kernel_size=1)
        )
    def forward(self, x):
        return x * torch.sigmoid(self.channel_attention(x))



# 非线性变换
class NonLinear(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_layers=2):
        super(NonLinear, self).__init__()
        layers =[]
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(hidden_channels,in_channels, kernel_size=3, padding=1))
            layers.append(ChannelAttention(in_channels, reduction_ratio=1))
            #layers.append(nn.ReLU(inplace=True))
        # Convert the layers list to a nn.Sequential object
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)+x

# 多尺度
class MultiScaleProcessing(nn.Module):
    def __init__(self, scales=2, combine_method='sum', in_channels=3, hidden_channels=64):
        """
        scales: int, the number of scales to be processed
        combine_method: str, how to combine outputs from different scales ('sum' or 'concatenate')
        in_channels: int, input channels
        hidden_channels: int, hidden channels for processing at each scale
        """
        super(MultiScaleProcessing, self).__init__()
        assert scales >= 1, "There should be at least 1 scale."
        assert combine_method in ['sum', 'concatenate'], "Invalid combine_method"
        self.scales = scales
        self.combine_method = combine_method
        self.downsamples = nn.ModuleList([nn.AvgPool2d(2**i, stride=2**i) for i in range(1, scales)])
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True) for i in range(1, scales)])

        # Define processing for each scale. For simplicity, we'll just use the NonLinear module from your previous code.
        self.processings = nn.ModuleList([NonLinear(in_channels, hidden_channels) for _ in range(scales)])

    def forward(self, x):
        multi_scale_outputs = []

        # Process original scale
        x_processed = self.processings[0](x)
        multi_scale_outputs.append(x_processed)

        # Process other scales
        for i in range(1, self.scales):
            x_down = self.downsamples[i-1](x)
            x_down_processed = self.processings[i](x_down)
            x_up = self.upsamples[i-1](x_down_processed)
            multi_scale_outputs.append(x_up)

        # Combine the outputs
        if self.combine_method == 'sum':
            return sum(multi_scale_outputs)
        else:  # concatenate
            return torch.cat(multi_scale_outputs, dim=1)

# 具体亮度调整
class LocalAdjustment(nn.Module):
    def __init__(self, in_channels, window_size=5):
        super(LocalAdjustment, self).__init__()
        self.window_size = window_size
        self.weights = nn.Parameter(torch.zeros(in_channels, window_size, window_size))

    def forward(self, x):
        b, c, h, w = x.size()
        pad = self.window_size // 2
        x_padded = torch.nn.functional.pad(x, [pad, pad, pad, pad], mode='reflect')
        patches = x_padded.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        weights_expanded = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        weighted_patches = patches * weights_expanded
        y = weighted_patches.reshape(b, c, -1, h, w).contiguous().sum(dim=2)
        # 添加残差连接
        y = x + y
        return y

import torch.nn as nn

class LocalAdjustmentConv(nn.Module):
    def __init__(self, in_channels, window_size=5):
        super(LocalAdjustmentConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=window_size, padding=window_size//2, groups=in_channels, bias=False)
        # 初始化卷积权重为零
        nn.init.zeros_(self.conv.weight)
        
    def forward(self, x):
        y = self.conv(x)
        
        # 添加残差连接
        y = x + y
        return y


class LocalBrightnessAdjustment(nn.Module):
    def __init__(self, in_channels, window_size=5, num_blocks=5):
        super(LocalBrightnessAdjustment, self).__init__()
        self.blocks = nn.ModuleList([LocalAdjustmentConv(in_channels, window_size) for _ in range(num_blocks)])
    def forward(self, x):
        y =x.clone()
        for block in self.blocks:
            x = block(x)
        x=x+y
        return x

# 注意力机制模块
class Attention(nn.Module):
    def __init__(self, in_channels,kernel_size=3):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        attention_map = self.sigmoid(self.conv(x))
        return x * attention_map*2


import math
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c).contiguous()
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        #illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))
        #v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head).contiguous()
        out_c = self.proj(x).view(b, h, w, c).contiguous()
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).contiguous().permute(
            0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x  # Removed illu_fea_trans input
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=3, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea)  # bchw
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = LeWinBlcok(fea)
        
        if not x.is_contiguous():
            x = x.contiguous()
        # Mapping
        out = self.mapping(fea) + x

        return out


from torchvision.utils import save_image
class ISP_select(nn.Module):
    def __init__(self, num_layers=2, window_size=5, rggb_kernel_size=3, affine_init='kaiming', attention_kernel_size=3,scales=3,combine_method='sum',n_feat=31,level=3,num_blocks=[1,1,1],save_images =False,ablation='all',inter_nums=4):
        super(ISP_select, self).__init__()
        self.save_images =save_images
        self.ablation=ablation
        # 插值模块
        self.interpolation = RGGBtoRGB(kernel_size=rggb_kernel_size,inter_nums=inter_nums)
        # 仿射变换模块
        self.affine = Affine(init_method=affine_init)
        #STN
        #self.STN =STN()
        # 非线性变换模块
        self.non_linear = NonLinear(in_channels=3, hidden_channels=64, num_layers=num_layers)
        # 多尺度
        self.pixel_adjustment = MultiScaleProcessing(scales=scales, combine_method=combine_method)
        # 局部亮度调整模块
        self.local_brightness_adjustment = LocalBrightnessAdjustment(in_channels=3, window_size=window_size)
        # trnsformer denoiser:
        self.denoiser = Denoiser(in_dim=3,out_dim=3,dim=n_feat,level=level,num_blocks=num_blocks) 
        # 注意力机制
        self.attention = Attention(in_channels=3, kernel_size=attention_kernel_size)
    def forward(self, x):
        # 插值
        if self.ablation != 'notlinear':
            x = self.interpolation(x)
        else:
            #x = self.interpolation(x)
            #print(x.shape)
            #x = x.reshape(x.shape[0],x.shape[1]//4,x.shape[2]*2,x.shape[3]*2)
            shuffle = nn.PixelShuffle(2)
            x =shuffle(x)
            x = torch.cat((x, x, x), dim=1)
        if self.save_images:
            save_image(x, './results/interpolation.png')
        # 仿射变换
        #x =self.STN(x)
        if self.ablation == 'onlyattention':
            x = self.denoiser(x)
            return x
        if self.ablation != 'notaffinelinear':
            x = self.affine(x)
            if self.save_images:
                save_image(x, './results/affine.png')
            # 非线性变换
            x = self.non_linear(x)
            if self.save_images:
                save_image(x, './results/non_linear.png')
        if self.ablation != 'notfilter':
            # 像素调整
            x = self.pixel_adjustment(x)
            if self.save_images:
                save_image(x, './results/pixel_adjustment.png')
            # 局部亮度调整
            x = self.local_brightness_adjustment(x)
            if self.save_images:
                save_image(x, './results/local_brightness_adjustment.png')
        if self.ablation != 'notattention':
            # transform denoiser
            x = self.denoiser(x)
            # 注意力机制
        x = self.attention(x)
        if self.save_images:
            save_image(x, './results/attention.png')
            os.exit()
        return x


class ScaleProcessing(nn.Module):
    def __init__(self, in_channels, window_size=5, rggb_kernel_size=3, affine_init='kaiming', attention_kernel_size=3,scales=7,combine_method='sum'):
        super(ScaleProcessing, self).__init__()
        self.interpolation = RGGBtoRGB(kernel_size=rggb_kernel_size)
        self.affine = Affine(init_method=affine_init)
        self.non_linear = NonLinear(in_channels=in_channels, hidden_channels=64)
        self.pixel_adjustment = MultiScaleProcessing(scales=scales, combine_method=combine_method)
        self.local_brightness_adjustment = LocalBrightnessAdjustment(in_channels=in_channels, window_size=window_size)
        self.attention = Attention(in_channels=in_channels, kernel_size=attention_kernel_size)
    
    def forward(self, x):
        x = self.interpolation(x)
        x = self.affine(x)
        x = self.non_linear(x)
        x = self.local_brightness_adjustment(x)
        x = self.attention(x)
        return x

class MultiScaleISP(nn.Module):
    def __init__(self, scales=6, combine_method='concatenate', in_channels=3, window_size=5, rggb_kernel_size=3, affine_init='kaiming', attention_kernel_size=3):
        super(MultiScaleISP, self).__init__()
        assert scales >= 1, "There should be at least 1 scale."
        assert combine_method in ['sum', 'concatenate'], "Invalid combine_method"

        self.scales = scales
        self.combine_method = combine_method

        self.downsamples = nn.ModuleList([nn.AvgPool2d(2**i, stride=2**i) for i in range(1, scales)])
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True) for i in range(1, scales)])
        self.scale_processings = nn.ModuleList([ScaleProcessing(in_channels=in_channels, window_size=window_size, rggb_kernel_size=rggb_kernel_size, affine_init=affine_init, attention_kernel_size=attention_kernel_size) for _ in range(scales)])
        self.conv1x1 = nn.Conv2d(3*scales, 3, kernel_size=1)

    def forward(self, x):
        multi_scale_outputs = []
        # Process original scale
        x_processed = self.scale_processings[0](x)
        multi_scale_outputs.append(x_processed)

        # Process other scales
        for i in range(1, self.scales):
            x_down = self.downsamples[i-1](x)
            x_down_processed = self.scale_processings[i](x_down)
            x_up = self.upsamples[i-1](x_down_processed)
            multi_scale_outputs.append(x_up)

        # Combine the outputs
        if self.combine_method == 'sum':
            return sum(multi_scale_outputs)
        else:  # concatenate
            result = torch.cat(multi_scale_outputs, dim=1)
            result = self.conv1x1(result)
            return result
