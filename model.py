import math
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class SpaceTimeSteps(nn.Module):
    """
    Disc :
        Given the number of T/TotalSteps
        Instead of going through step 1,2,3 etc
        we choose N steps
    Example :
         N = 4 = number of steps
    Return :
        t : number of steps

    """
    def __init__(self, T, N):
        super().__init__()
        self.T: int = T
        self.N: int = N

    def forward(self) -> list[int]:
        assert self.T != 0
        assert self.N != 0
        jumps = (self.T - 1) / (self.N - 1)
        _c = 0.0
        t: list[int] = []
        for _ in range(self.N):
            t.append(round(_c))
            _c = _c + jumps
        return t


class TimeEmbedding(nn.Module):
    """
    Disc :
        We are looking to convert every step t
        to a vector representation
    Example :
        Step 100 -> [sin , cos , sin , ...]
    Return :
        Position of all timesteps t
    """
    def __init__(self, T, dim=64):
        super().__init__()
        self.dim: int = dim
        self.T: int = T

    def forward(self, t):
        out = []
        for step in t :
            pos = torch.zeros(self.dim)
            for i in range(self.dim // 2):
                pos[2 * i] = math.sin((step) / (10000 ** ((2 * i) / self.dim)))
                pos[2 * i + 1] = math.cos((step) / (10000 ** ((2 * i) / self.dim)))
            out.append(pos)
        return torch.stack(out)

class ResBlock(nn.Module):
    """
    Disc :
        Residual block for a diffusion U-Net. Runs the input through
        norm -> SiLU -> conv, injects the time step embedding in the
        middle, then norm -> SiLU -> conv again. The input is added
        back at the end (skip connection), so the layers only learn
        the correction to x.
    """
    def __init__(self, ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(64, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        if ch != out_ch:
            self.skip = nn.Conv2d(ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x , t_emb):
        h = self.conv1(self.norm1(F.silu(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = h.to(device)
        h = self.conv2(self.norm2(F.silu(h)))
        return self.skip(x) + h

class FakeE(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Conv2d(1, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 1, 1)
        # Encoder
        self.enc_1 = ResBlock(64, 64)
        self.enc_2 = ResBlock(64, 128)
        self.enc_3 = ResBlock(128, 256)
        self.enc_4 = ResBlock(256, 512)

        # Bottleneck
        self.bottle = ResBlock(512, 512)

        # Down
        self.down_1 = nn.Conv2d(64, 64, 4, 2, 1)
        self.down_2 = nn.Conv2d(128, 128, 4, 2, 1)
        self.down_3 = nn.Conv2d(256, 256, 4, 2, 1)

        # Up
        self.up_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up_2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up_3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        # Decoder
        self.dec_3 = ResBlock(256 + 256, 256)
        self.dec_2 = ResBlock(128 + 128, 128)
        self.dec_1 = ResBlock(64 + 64, 64)


    def forward(self, x, t_emb):
        t_emb = t_emb.to(device)
        x = self.in_conv(x)
        # Encoder
        e1 = self.enc_1(x, t_emb)       # 64x64, 64
        x = self.down_1(e1)             # 32x32, 64

        e2 = self.enc_2(x, t_emb)       # 32x32, 128
        x = self.down_2(e2)             # 16x16, 128

        e3 = self.enc_3(x, t_emb)       # 16x16, 256
        x = self.down_3(e3)             # 8x8, 256

        e4 = self.enc_4(x, t_emb)       # 8x8, 512

        # Bottleneck
        x = self.bottle(e4, t_emb)      # 8x8, 512

        # Decoder
        x = self.up_1(x)                # 16x16, 256
        x = torch.cat([x, e3], dim=1)   # 16x16, 512
        x = self.dec_3(x, t_emb)        # 16x16, 256

        x = self.up_2(x)                # 32x32, 128
        x = torch.cat([x, e2], dim=1)   # 32x32, 256
        x = self.dec_2(x, t_emb)        # 32x32, 128

        x = self.up_3(x)                # 64x64, 64
        x = torch.cat([x, e1], dim=1)   # 64x64, 128
        x = self.dec_1(x, t_emb)        # 64x64, 64
        return self.out(x)
