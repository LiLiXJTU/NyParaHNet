from functools import partial
from torch.nn.modules.utils import _pair
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
try:
    from linerattn_RoPe import PnPNystraAttention
    from FANLayer import FAN_MY_Res
except:
    from .linerattn_RoPe import PnPNystraAttention
    from .FANLayer import FAN_MY_Res

class NyParaHNet(nn.Module):

    def __init__(self, in_chan, base_chan, num_classes=9, aux_loss=False, num_block=2, embed_dims=[32, 64, 128, 256], num_landmarks=[64, 32, 16, 16]):
        super().__init__()

        self.aux_loss = aux_loss
        self.inc = [BasicBlock(in_chan, base_chan)]
        self.inc.append(BasicBlock(base_chan, base_chan))
        self.inc = nn.Sequential(*self.inc)
        self.down1 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[0],num_landmarks=num_landmarks[0])
        self.down2 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[1],num_landmarks=num_landmarks[1])
        self.down3 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[2],num_landmarks=num_landmarks[2])
        self.down4 = down_block(8 * base_chan, 16 * base_chan, (2, 2), num_block=num_block, embed_dims=embed_dims[3],num_landmarks=num_landmarks[3])
        self.up4 = up_block(2 * base_chan, base_chan, scale=(2, 2), num_block=2)
        self.up3 = up_block(4 * base_chan, 2 * base_chan, scale=(2, 2), num_block=2)
        self.up2 = up_block(8 * base_chan, 4 * base_chan, scale=(2, 2), num_block=2)
        self.up1 = up_block(16 * base_chan, 8 * base_chan, scale=(2, 2), num_block=2)

        self.outc = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out


class TransBlock(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4., sr_ratio=1, drop=0.,
                 drop_path=0., norm_layer=nn.GroupNorm, num_landmarks=16):
        super().__init__()
        self.norm1 = norm_layer(num_groups=1, num_channels=dim)
        self.norm_cnn = norm_layer(num_groups=1, num_channels=dim)
        self.attn = PnPNystraAttention(
            num_landmarks=num_landmarks,
            iters=6,
            dim=dim,
            num_heads=num_heads,
            token_projection='conv',
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(num_groups=1, num_channels=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FAN_MY_Res(input_dim=dim, output_dim=dim,hidden_features=mlp_hidden_dim)

    def forward(self, x, cnn_feature=None):
        if cnn_feature is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), attn_kv=self.norm_cnn(cnn_feature)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Trans_CNNBlock(nn.Module):

    def __init__(self, inplanes, planes, embed_dim, stride=1, in_patch_size=7, in_stride=1, in_pad=3,num_landmarks=16):
        super().__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.conv1 = conv3x3(planes//2, planes//2, stride)
        self.bn1 = nn.BatchNorm2d(planes//2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes//2, planes//2)
        self.bn2 = nn.BatchNorm2d(planes//2)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(planes//2),
                self.relu,
                nn.Conv2d(planes//2, planes//2, kernel_size=1, stride=stride, bias=False)
            )
        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=planes//2,
                                      embed_dim=planes//2)
        self.trans_block = TransBlock(dim=planes//2,num_landmarks=num_landmarks)
        self.proj = nn.Sequential(
            nn.BatchNorm2d(planes),
            self.relu,
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes//2),

            nn.BatchNorm2d(planes),
            self.relu,
            nn.Conv2d(planes, planes//2, kernel_size=1),

            nn.BatchNorm2d(planes//2),
            self.relu,
            nn.Conv2d(planes//2, planes, kernel_size=1))

        self.conv0 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            self.relu,
            nn.Conv2d(inplanes, planes, kernel_size=1))

    def forward(self, x):
        x = self.conv0(x)
        x = torch.split(x, self.planes//2, dim=1)
        CNN_x = x[0]
        Trans_x = x[1]

        residue = CNN_x
        out = self.bn1(CNN_x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)
        _,_,H,W = out.size()
        x = self.patch_embed(Trans_x)
        x = self.trans_block(x,out)
        x = torch.cat([out, x], dim=1)
        out = self.proj(x) + x
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                self.relu,
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, embed_dims, bottleneck=False, pool=True,num_landmarks=16):
        super().__init__()

        block_list = []

        if bottleneck:
            block = 2
        else:
            block = Trans_CNNBlock

        if pool:
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block(in_ch, out_ch, embed_dims,num_landmarks=num_landmarks))
        else:
            block_list.append(block(in_ch, out_ch, embed_dims, stride=2,num_landmarks=num_landmarks))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch, embed_dims, stride=1, num_landmarks=num_landmarks))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, scale=(2, 2), bottleneck=False):
        super().__init__()
        self.scale = scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BasicBlock
        else:
            block = BasicBlock

        block_list = []
        block_list.append(block(2 * out_ch, out_ch))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                     groups=groups)

class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size=7,
                 stride=2,
                 padding=3,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans,
                      embed_dim, kernel_size=patch_size,
                      stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
    def forward(self, x):
        return self.proj(x)


def count_param(model):
    param_count = sum(p.numel() for p in model.parameters())
    return param_count


if __name__ == '__main__':
    net = NyParaHNet(in_chan=3, base_chan=32, num_classes=9).cuda()

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(net, torch.randn(1, 3, 224, 224).cuda())
    print(f"FLOPs: {flops.total()/1e9:.2f} G")

    from torchinfo import summary  # 使用更现代的库

    summary(
        net,
        input_size=(1,3, 224, 224),  # 输入维度 (C, H, W)
        col_names=["input_size", "output_size", "num_params", "kernel_size"],  # 显示更多信息
        verbose=1  # 完整输出模式
    )