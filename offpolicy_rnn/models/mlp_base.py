from .rnn_base import RNNBase
from .RNNHidden import RNNHidden
class MLPBase(RNNBase):
    def __init__(self, input_size, output_size, hidden_size_list, activation):
        super().__init__(input_size, output_size, hidden_size_list, activation, ['fc'] * len(activation))
        self.empty_hidden_state = RNNHidden(0, [])

    def meta_forward(self, x, h=None, require_full_hidden=False):
        return super(MLPBase, self).meta_forward(x, self.empty_hidden_state, False)[0]

    def forward(self, x):
        return self.meta_forward(x)
