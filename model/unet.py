# We need to change the [Unet] and make it can handle the header of the data
from typing import Optional, List, Tuple
import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(n, self.groups, c // self.groups, h, w)  # group
        x = x.transpose(1, 2).contiguous().view(n, -1, h, w)  # shuffle

        return x


class ConvBnSiLu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x) -> torch.Tensor:
        return self.module(x)


class ResidualBottleneck(nn.Module):
    """
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2
            ),
            nn.BatchNorm2d(in_channels // 2),
            ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0),
        )
        self.branch2 = nn.Sequential(
            ConvBnSiLu(in_channels // 2, in_channels // 2, 1, 1, 0),
            nn.Conv2d(
                in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2
            ),
            nn.BatchNorm2d(in_channels // 2),
            ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0),
        )
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        x = self.channel_shuffle(x)  # shuffle two branches

        return x


class ResidualDownsample(nn.Module):
    """
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0),
        )
        self.branch2 = nn.Sequential(
            ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0),
            nn.Conv2d(
                out_channels // 2, out_channels // 2, 3, 2, 1, groups=out_channels // 2
            ),
            nn.BatchNorm2d(out_channels // 2),
            ConvBnSiLu(out_channels // 2, out_channels // 2, 1, 1, 0),
        )
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = self.channel_shuffle(x)  # shuffle two branches

        return x


class TimeMLP(nn.Module):
    """
    naive introduce timestep information to feature maps with mlp and add shortcut
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb

        return self.act(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int):
        super().__init__()
        self.conv0 = nn.Sequential(
            *[ResidualBottleneck(in_channels, in_channels) for _i in range(3)],
            ResidualBottleneck(in_channels, out_channels // 2),
        )

        self.time_mlp = TimeMLP(
            embedding_dim=time_embedding_dim,
            hidden_dim=out_channels,
            out_dim=out_channels // 2,
        )
        self.conv1 = ResidualDownsample(out_channels // 2, out_channels)

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_shortcut = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x_shortcut, t)
        x = self.conv1(x)

        return x, x_shortcut


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv0 = nn.Sequential(
            *[ResidualBottleneck(in_channels, in_channels) for _i in range(3)],
            ResidualBottleneck(in_channels, in_channels // 2),
        )

        self.time_mlp = TimeMLP(
            embedding_dim=time_embedding_dim,
            hidden_dim=in_channels,
            out_dim=in_channels // 2,
        )
        self.conv1 = ResidualBottleneck(in_channels // 2, out_channels // 2)

    def forward(
        self,
        x: torch.Tensor,
        x_shortcut: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv1(x)

        return x


class Unet(nn.Module):
    """
    simple unet design without attention
    """

    def __init__(
        self,
        timesteps: int,
        time_embedding_dim: int,
        in_channels: int = 3,
        out_channels: int = 1,
        base_dim: int = 32,  # become thicker
        dim_mults: List[int] = [2, 4, 8, 16],
    ):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0

        channels = self._cal_channels(base_dim, dim_mults)

        # it is used to make channels become base_dim
        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(c[1], c[0], time_embedding_dim) for c in channels[::-1]]
        )

        self.mid_block = nn.Sequential(
            *[ResidualBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
            ResidualBottleneck(channels[-1][1], channels[-1][1] // 2),
        )

        self.final_conv = nn.Conv2d(
            in_channels=channels[0][0] // 2, out_channels=out_channels, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor, time_stamp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.init_conv(x)
        if time_stamp is not None:
            time_stamp = self.time_embedding(time_stamp)
        encoder_shortcuts: List[nn.Module] = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, time_stamp)
            encoder_shortcuts.append(x_shortcut)
        x = self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut, time_stamp)
        x = self.final_conv(x)

        return x

    def _cal_channels(
        self, base_dim: int, dim_mults: List[int]
    ) -> List[Tuple[int, int]]:
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels: List[Tuple[int, int]] = []
        for i in range(len(dims) - 1):
            channels.append((dims[i], dims[i + 1]))  # in_channel, out_channel

        return channels


if __name__ == "__main__":
    x = torch.randint(0, 100, (3, 1, 256, 224))
    t = torch.randint(0, 1000, (3,))
    model: Unet = Unet(1000, 128, in_channels=1, out_channels=1)
    y = model(x.float(), t)
    print(y.shape)
