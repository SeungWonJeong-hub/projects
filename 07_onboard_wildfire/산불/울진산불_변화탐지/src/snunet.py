"""SNUNet-CD (Fang et al. 2021) — 샴(Siamese) NestedUNet 변화탐지.
KJRS 2025(한국 대형산불)에서 early-fusion UNet 대비 +14 F1/+19 IoU 우월 확인.
입력: pre(6ch), post(6ch) 별도 → 공유 인코더 → 밀집 스킵(UNet++) → ECAM → burn 마스크(1ch).
"""
import torch, torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(s, ci, co):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(s, x): return s.net(x)


class ChannelAttn(nn.Module):
    """ECAM 채널어텐션 (avg+max pool → 공유 MLP → sigmoid 게이트)."""
    def __init__(s, c, r=16):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True), nn.Linear(max(c // r, 4), c))
    def forward(s, x):
        b, c, _, _ = x.shape
        a = s.mlp(x.mean((2, 3))); m = s.mlp(x.amax((2, 3)))
        return x * torch.sigmoid(a + m).view(b, c, 1, 1)


class SNUNetCD(nn.Module):
    def __init__(s, in_ch=6, out_ch=1, base=32):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8, base * 16]
        s.pool = nn.MaxPool2d(2)
        s.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 공유 인코더
        s.c0_0 = ConvBlock(in_ch, f[0]); s.c1_0 = ConvBlock(f[0], f[1]); s.c2_0 = ConvBlock(f[1], f[2])
        s.c3_0 = ConvBlock(f[2], f[3]); s.c4_0 = ConvBlock(f[3], f[4])
        # 밀집 디코더 (스킵은 pre+post 두 인코더 특징을 concat)
        s.c0_1 = ConvBlock(f[0] * 2 + f[1], f[0]); s.c1_1 = ConvBlock(f[1] * 2 + f[2], f[1])
        s.c2_1 = ConvBlock(f[2] * 2 + f[3], f[2]); s.c3_1 = ConvBlock(f[3] * 2 + f[4], f[3])
        s.c0_2 = ConvBlock(f[0] * 3 + f[1], f[0]); s.c1_2 = ConvBlock(f[1] * 3 + f[2], f[1])
        s.c2_2 = ConvBlock(f[2] * 3 + f[3], f[2])
        s.c0_3 = ConvBlock(f[0] * 4 + f[1], f[0]); s.c1_3 = ConvBlock(f[1] * 4 + f[2], f[1])
        s.c0_4 = ConvBlock(f[0] * 5 + f[1], f[0])
        s.ecam = ChannelAttn(f[0] * 4)
        s.final = nn.Conv2d(f[0] * 4, out_ch, 1)

    def _enc(s, x):
        x0 = s.c0_0(x); x1 = s.c1_0(s.pool(x0)); x2 = s.c2_0(s.pool(x1))
        x3 = s.c3_0(s.pool(x2)); x4 = s.c4_0(s.pool(x3))
        return x0, x1, x2, x3, x4

    def forward(s, xA, xB):
        a0, a1, a2, a3, a4 = s._enc(xA)
        b0, b1, b2, b3, b4 = s._enc(xB)
        cat = lambda *t: torch.cat(t, 1)
        x0_1 = s.c0_1(cat(a0, b0, s.up(b1)))
        x1_1 = s.c1_1(cat(a1, b1, s.up(b2)))
        x2_1 = s.c2_1(cat(a2, b2, s.up(b3)))
        x3_1 = s.c3_1(cat(a3, b3, s.up(b4)))
        x0_2 = s.c0_2(cat(a0, b0, x0_1, s.up(x1_1)))
        x1_2 = s.c1_2(cat(a1, b1, x1_1, s.up(x2_1)))
        x2_2 = s.c2_2(cat(a2, b2, x2_1, s.up(x3_1)))
        x0_3 = s.c0_3(cat(a0, b0, x0_1, x0_2, s.up(x1_2)))
        x1_3 = s.c1_3(cat(a1, b1, x1_1, x1_2, s.up(x2_2)))
        x0_4 = s.c0_4(cat(a0, b0, x0_1, x0_2, x0_3, s.up(x1_3)))
        out = cat(x0_1, x0_2, x0_3, x0_4)
        return s.final(s.ecam(out))
