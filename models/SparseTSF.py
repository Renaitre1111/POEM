# Adapted from lss-1138/SparseTSF (Apache-2.0), commit b8c2740.
import torch
import torch.nn as nn


class Model(nn.Module):
    """SparseTSF model from lss-1138/SparseTSF."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type

        if self.model_type not in ('linear', 'mlp'):
            raise ValueError("model_type must be 'linear' or 'mlp'")
        if self.seq_len % self.period_len or self.pred_len % self.period_len:
            raise ValueError('seq_len and pred_len must be divisible by period_len')

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode='zeros',
            bias=False,
        )

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
            -1, self.enc_in, self.seq_len
        ) + x
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        if self.model_type == 'linear':
            y = self.linear(x)
        else:
            y = self.mlp(x)

        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        return y.permute(0, 2, 1) + seq_mean
