import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class FAN_MY_Res(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True,hidden_features=64):
        super(FAN_MY_Res, self).__init__()
        
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        self.p_ratio = p_ratio
        p_output_dim = int(hidden_features * self.p_ratio)
        g_output_dim = hidden_features - p_output_dim * 2  # Account for cosine and sine terms
        
        self.fc1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.GELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),

        )

        self.input_linear_p = nn.Linear(hidden_features, p_output_dim, bias=use_p_bias)
        
        self.input_linear_g = nn.Linear(hidden_features, g_output_dim)
        
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else lambda x: x

    def forward(self, src):
        src = self.fc1(src)
        identity = src
        _,_,h,w = src.shape
        src = rearrange(src, 'b c h w -> b (h w) c', h=h, w=w)

        g = self.activation(self.input_linear_g(src))
        
        p = self.input_linear_p(src)
        
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        output = rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)

        output = self.fc2(output+identity)
        return output
    