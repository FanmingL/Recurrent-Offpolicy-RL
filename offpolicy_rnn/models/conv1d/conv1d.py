import torch
from torch import nn


class Conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_conv=4, bias=True, ff=True):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_conv = d_conv
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            kernel_size=d_conv,
            groups=in_channels,
            padding=0,
        )
        self.desired_hidden_dim = self.in_channels * (self.d_conv - 1)
        self.use_ff = ff
        if ff:
            self.ff = PositionWiseFeedForward(out_channels, 0.0)


    def conv1d_func(self, x, hidden, mask):
        (b, l, d) = x.shape
        if mask is not None:
            x = x * mask
        x_input = torch.cat((hidden, x), dim=-2)
        x = x_input.transpose(-2, -1)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(-2, -1)
        hidden = x_input[:, -(self.d_conv-1):, :]
        return x, hidden

    def forward(self, x, hidden=None, mask=None):
        batch_size = x.shape[0]
        if hidden is None:
            hidden = torch.zeros((batch_size, self.d_conv - 1, self.in_channels), device=x.device, dtype=x.dtype)
        else:
            hidden = hidden.reshape((batch_size, self.d_conv - 1, self.in_channels))

        x, hidden = self.conv1d_func(x, hidden, mask)
        hidden = hidden.reshape((batch_size, 1, -1))
        if self.use_ff:
            x = self.ff(x)
        return x, hidden




class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)


