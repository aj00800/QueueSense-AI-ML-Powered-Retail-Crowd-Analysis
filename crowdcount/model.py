from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDensityCNN(nn.Module):
    """Small fully-conv encoder–decoder producing a non-negative density map.

    Uses skip connections (U-Net style) for better spatial accuracy.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels

        def conv(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = nn.Sequential(conv(3, c), conv(c, c))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), conv(c, 2 * c), conv(2 * c, 2 * c))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), conv(2 * c, 4 * c), conv(4 * c, 4 * c))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), conv(4 * c, 8 * c), conv(8 * c, 8 * c))

        # Decoder uses skip connections — input channels = upsampled + skip
        self.dec3 = nn.Sequential(conv(8 * c + 4 * c, 4 * c), conv(4 * c, 4 * c))
        self.dec2 = nn.Sequential(conv(4 * c + 2 * c, 2 * c), conv(2 * c, 2 * c))
        self.dec1 = nn.Sequential(conv(2 * c + c, c), conv(c, c))
        self.head = nn.Conv2d(c, 1, kernel_size=1)

        # Initialize head bias slightly negative so ReLU outputs near-zero initially
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)       # [B, c,   H,   W]
        x2 = self.enc2(x1)      # [B, 2c,  H/2, W/2]
        x3 = self.enc3(x2)      # [B, 4c,  H/4, W/4]
        x4 = self.enc4(x3)      # [B, 8c,  H/8, W/8]

        # Upsample and concat with skip connections
        y = F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, x3], dim=1))

        y = F.interpolate(y, size=x2.shape[2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x2], dim=1))

        y = F.interpolate(y, size=x1.shape[2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, x1], dim=1))

        y = self.head(y)

        # density must be non-negative
        return F.relu(y)
