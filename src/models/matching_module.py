import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv4d import Interpolate4d, Encoder4D

class OurModel(nn.Module):
    def __init__(
        self,
        feat_dim=[9,8],
        time_embed_dim=9,
    ):
        super().__init__()
        self.encoders = nn.ModuleList([])
        self.encoders.append(
            Encoder4D( 
                        corr_levels=(feat_dim[0], 16,  32, 16, feat_dim[1]),
                        kernel_size=(
                            (1, 1, 3, 3), 
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                        ),
                        stride=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                        ),
                        padding=(
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                        ),
                        group=(1,1,1,1),
                        residual=False
                    )
        )
        self.encoders.append(
            Encoder4D( 
                        corr_levels=(feat_dim[0], 16,  32, 16, feat_dim[1]),
                        kernel_size=(
                            (1, 1, 3, 3), 
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                        ),
                        stride=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                        ),
                        padding=(
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                        ),
                        group=(1,1,1,1),
                        residual=False
                    )
        )
        self.encoders.append(
            Encoder4D( 
                        corr_levels=(feat_dim[0], 16,  32, 16, feat_dim[1]),
                        kernel_size=(
                            (1, 1, 3, 3), 
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                        ),
                        stride=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                        ),
                        padding=(
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                        ),
                        group=(1,1,1,1),
                        residual=False
                    )
        )

    def forward(self, corr, t=None, block="down_blocks"):
        if block == "down_blocks":
            corr = self.encoders[0](corr)[0] 
        elif block == "mid_block":
            corr = self.encoders[1](corr)[0]
        elif block == "up_blocks":
            corr = self.encoders[2](corr)[0]

        return corr
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dtype = next(self.parameters()).dtype
        return self