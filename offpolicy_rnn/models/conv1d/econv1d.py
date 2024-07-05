import torch

class EConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_ensemble, d_conv=4, bias=True):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ensemble = num_ensemble
        self.d_conv = d_conv
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels * num_ensemble,
            out_channels=out_channels * num_ensemble,
            bias=bias,
            kernel_size=d_conv,
            groups=in_channels * num_ensemble,
            padding=0,
        )
        self.desired_hidden_dim = self.in_channels * (self.d_conv - 1) * num_ensemble
        self.desire_ndim = 4

    def rearange(self, x):
        # (num_ensemble, batch_size, T+d_conv-1, D)
        x = x.transpose(0, 1)
        # (batch_size, num_ensemble, T+d_conv-1, D)
        x = x.transpose(2, 3)
        # (batch_size, num_ensemble, D, T+d_conv-1)
        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        # (batch_size, num_ensemble * D, T+d_conv-1)
        return x

    def inverse_rearange(self, x, n_ensemble, d):
        # (batch_size, num_ensemble * D, T)
        x = x.reshape((x.shape[0], n_ensemble, d, x.shape[-1]))
        # (batch_size, num_ensemble, D, T)
        x = x.transpose(2, 3)
        # (batch_size, num_ensemble, T, D)
        x = x.transpose(0, 1)
        # (num_ensemble, batch_size, T, D)
        return x

    def conv1d_func(self, x, hidden, mask):
        # x (num_ensemble, batch_size, T, D)
        # hidden (num_ensemble, batch_size, d_conv-1, D)
        # mask (batch_size, T, 1)
        (n_ensemble, b, l, d) = x.shape
        # (n_ensemble, b, l, d) -> (b, n_ensemble, l, d) -> (b, n_ensemble, d, l) -> (b, n_ensemble * d, l)
        # transpose(0, 1) -> transpose(2, 3) -> reshape
        if mask is not None:
            if len(mask.shape) < len(x.shape):
                mask = mask.unsqueeze(0)
            x = x * mask
        # x: (num_ensemble, batch_size, T, D)
        # hidden: (num_ensemble, batch_size, d_conv-1, D)
        x_input = torch.cat((hidden, x), dim=-2)

        x = self.rearange(x_input)
        x = self.conv1d(x)[..., :l]
        x = self.inverse_rearange(x, n_ensemble, d)
        # x: (num_ensemble, batch_size, T, D)
        # hidden: (num_ensemble, batch_size, self.d_conv - 1, D)
        hidden = x_input[..., -(self.d_conv - 1):, :]
        return x, hidden

    def forward(self, x, hidden=None, mask=None):

        if len(x.shape) == 3 and self.desire_ndim == 4:
            x = x.unsqueeze(0).repeat_interleave(self.num_ensemble, dim=0)
        else:
            assert len(x.shape) == 4
        batch_size = x.shape[1]

        if hidden is None:
            hidden = torch.zeros((self.num_ensemble, batch_size, self.d_conv - 1, self.in_channels), device=x.device,
                                 dtype=x.dtype)
        else:
            # hidden: (batch_size, 1, num_ensemble * (self.d_conv-1) * self.in_channels)
            hidden = hidden.reshape((batch_size, self.num_ensemble, self.d_conv - 1, self.in_channels))
            hidden = hidden.transpose(0, 1)
            # hidden: (num_ensemble, batch_size, d_conv-1, D)

        x, hidden = self.conv1d_func(x, hidden, mask)
        #hidden: (num_ensemble, batch_size, self.d_conv - 1, D)
        hidden = hidden.transpose(0, 1)
        # hidden: (batch_size, num_ensemble, d_conv-1, D)
        hidden = hidden.reshape((batch_size, 1, -1))
        return x, hidden

