import time

import torch
import copy
import os
from .ensemble_linear_model import EnsembleLinear
from typing import Union, List, Optional, Tuple
from .RNNHidden import RNNHidden
from .lru.lru import LRULayer
from .lru.elru import EnsembleLRULayer
from .gilr.gilr import GILRLayer
from .gilr.egilr import EnsembleGILRLayer
from .gilr_lstm.egilr_lstm import EnsembleGILRLSTMLayer
from .gilr_lstm.gilr_lstm import GILRLSTMLayer
from .s6.mamba import MambaBlock, MambaResidualBlock
from .smamba.mamba import BlockList as MambaBlockList
from .conv1d.conv1d import Conv1d
from .conv1d.econv1d import EConv1d
# from .transformer.TransformerFormal import TransformerEncoderBlock
try:
    from .flash_attention.gpt import GPTLayer
except Exception as _:
    GPTLayer = None

try:
    from .flash_attention.TransformerFlashAttention import TransformerDecoder
except Exception as _:
    TransformerDecoder = None


class RNNBase(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size_list: List[int],
                 activation: List[str], layer_type: List[str]):
        """
        :param input_size: Network input size
        :param output_size: Network output size
        :param hidden_size_list: units number of hidden layers,
        :param activation: activation functions of hidden layers, e.g., tanh, relu, ...
        :param layer_type: layer types, e.g., fc, gru, efc-xx, ...
        """
        super().__init__()
        assert len(activation) - 1 == len(hidden_size_list), "number of activation should be " \
                                                             "larger by 1 than size of hidden layers."
        assert len(activation) == len(layer_type), "number of layer type should equal to the activate"
        activation_dict = {
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'sigmoid': torch.nn.Sigmoid,
            'leaky_relu': torch.nn.LeakyReLU,
            'linear': torch.nn.Identity,
            'elu': torch.nn.ELU,
            'gelu': torch.nn.GELU,
        }
        self.activation_dict = activation_dict
        layer_dict = {
            'fc': torch.nn.Linear,
            # 'efc': EnsembleLinear,
            'lstm': torch.nn.LSTM, # two output
            'gru': torch.nn.GRU,    # one output
            'lru': LRULayer,
            'gilr': GILRLayer,
            'gilr_lstm': GILRLSTMLayer,
            'mamba': MambaResidualBlock,
            'smamba': MambaBlockList,
            # 'transformer': TransformerEncoderBlock,
            'gpt': GPTLayer,
            'cgpt': TransformerDecoder
            # 'conv1d':
        }
        rnn_types = ['lstm', 'gru', 'cgru', 'lru', 'gilr', 'mamba', 'gilr_lstm', 'smamba', 'transformer', 'gpt', 'cgpt']
        rnn_headers = ['']
        rnn_type_set = set(rnn_types)
        self.check_is_rnn = lambda x: (x in rnn_type_set or (x.startswith('e') and x[1:].split('-')[0] in rnn_type_set)
                                       or x.startswith('conv1d') or x.startswith('econv1d') or x.startswith('mamba')
                                       or x.startswith('smamba') or x.startswith('transformer') or x.startswith('gpt')
                                       or x.startswith('cgpt'))
        norm_dict = {
            'ln': torch.nn.LayerNorm,
            'eln': torch.nn.LayerNorm,
            # 'bn': torch.nn.BatchNorm1d,
        }
        # def decorate(module):
        #     return torch.jit.script(module)
        def fc_decorate(module):
            return module
        def rnn_decorate(module):
            return module
        def ln_decorate(module):
            return module
        lst_nh = input_size
        self.layer_type = copy.deepcopy(layer_type)
        self.activation_type = copy.deepcopy(activation)
        self.layer_list = torch.nn.ModuleList()
        self.activation_list = torch.nn.ModuleList()
        self.rnn_hidden_state_input_size = []
        self.rnn_layer_type = []

        self.rnn_num = 0
        hidden_size_list = hidden_size_list + [output_size]
        for ind, item in enumerate(hidden_size_list):
            if self.layer_type[ind] == 'fc':
                self.layer_list.append(fc_decorate(layer_dict[self.layer_type[ind]](lst_nh, item)))
            elif self.layer_type[ind].startswith('efc'):
                self.layer_list.append(fc_decorate(EnsembleLinear(lst_nh, item, int(self.layer_type[ind].split('-')[-1]))))
            else:
                self.rnn_num += 1
                if self.layer_type[ind] in ['lru', 'gilr_lstm']:
                    self.rnn_hidden_state_input_size.append(item * 2)
                    self.layer_list.append(rnn_decorate(layer_dict[self.layer_type[ind]](lst_nh, item, batch_first=True)))
                elif self.layer_type[ind].startswith('e') and self.layer_type[ind][1:].split('-')[0] in ['lru', 'gilr_lstm']:
                    ensemble_num = int(self.layer_type[ind].split('-')[-1])
                    self.rnn_hidden_state_input_size.append(item * 2 * ensemble_num)
                    self.layer_list.append(rnn_decorate(EnsembleLRULayer(lst_nh, item, ensemble_num, batch_first=True)))
                elif self.layer_type[ind].startswith('egilr'):
                    ensemble_num = int(self.layer_type[ind].split('-')[-1])
                    self.rnn_hidden_state_input_size.append(item * ensemble_num)
                    self.layer_list.append(rnn_decorate(EnsembleGILRLayer(lst_nh, item, ensemble_num, batch_first=True)))
                elif self.layer_type[ind].startswith('mamba'):
                    d_conv = 4
                    d_state = 16
                    use_ff = True
                    if '_' in self.layer_type[ind]:
                        for config_ in self.layer_type[ind].split('_')[1:]:
                            if config_.startswith('s'):
                                d_state = int(config_[1:])
                            elif config_.startswith('c'):
                                d_conv = int(config_[1:])
                            elif config_.startswith('no'):
                                if config_[2:] == 'ff':
                                    use_ff = False
                            else:
                                raise f'Pattern {config_} has not been implemented!'
                    # print(f'MAMBA parameter: d_state: {d_state}, d_conv: {d_conv}')
                    residual_block = MambaResidualBlock(lst_nh, item, d_conv=d_conv, d_state=d_state, use_ff=use_ff)
                    self.layer_list.append(rnn_decorate(residual_block))
                    self.rnn_hidden_state_input_size.append(residual_block.mixer.desired_hidden_dim)
                elif self.layer_type[ind].startswith('smamba'):
                    d_conv = 4
                    d_state = 16
                    block_num = 2
                    rms_norm = True
                    use_ff = False
                    if '_' in self.layer_type[ind]:
                        # smamba_s32_c16_b2_nln
                        for config_ in self.layer_type[ind].split('_')[1:]:
                            if config_.startswith('s'):
                                d_state = int(config_[1:])
                            elif config_.startswith('c'):
                                d_conv = int(config_[1:])
                            elif config_.startswith('b'):
                                block_num = int(config_[1:])
                            elif config_.startswith('n'):
                                rms_norm = False if config_[1:] == 'ln' else True
                            elif config_.startswith('f'):
                                if config_[1:] == 'f':
                                    use_ff = True
                            else:
                                raise f'Pattern {config_} has not been implemented!'
                    # print(f'MAMBA parameter: d_state: {d_state}, d_conv: {d_conv}')
                    assert lst_nh == item, f'mamba_simple require input_dim == output_dim, while got {lst_nh} and {item}'
                    residual_block = MambaBlockList(block_num, lst_nh, d_conv=d_conv, d_state=d_state, rms_norm=rms_norm, use_ff=use_ff)
                    self.layer_list.append(rnn_decorate(residual_block))
                    self.rnn_hidden_state_input_size.append(residual_block.desired_hidden_dim)
                elif self.layer_type[ind].startswith('gpt'):
                    nhead = 8
                    ndim = lst_nh
                    nlayer = 4
                    pdrop = 0.1
                    maxlength = 2048
                    if '_' in self.layer_type[ind]:
                        for config_ in self.layer_type[ind].split('_')[1:]:
                            if config_.startswith('h'):
                                nhead = int(config_[1:])
                            # elif config_.startswith('d'):
                            #     ndim = int(config_[1:])
                            elif config_.startswith('l'):
                                nlayer = int(config_[1:])
                            elif config_.startswith('p'):
                                pdrop = float(config_[1:])
                            elif config_.startswith('ml'):
                                maxlength = int(config_[2:])
                            else:
                                raise f'Pattern {config_} has not been implemented!'
                    self.layer_list.append(rnn_decorate(GPTLayer(ndim=ndim, nhead=nhead, nlayer=nlayer, pdrop=pdrop)))
                    self.rnn_hidden_state_input_size.append(maxlength)
                elif self.layer_type[ind].startswith('cgpt'):
                    nhead = 8
                    ndim = lst_nh
                    nlayer = 4
                    pdrop = 0.1
                    maxlength = 1024
                    ln = True
                    if '_' in self.layer_type[ind]:
                        for config_ in self.layer_type[ind].split('_')[1:]:
                            if config_.startswith('h'):
                                nhead = int(config_[1:])
                            # elif config_.startswith('d'):
                            #     ndim = int(config_[1:])
                            elif config_.startswith('l'):
                                nlayer = int(config_[1:])
                            elif config_.startswith('p'):
                                pdrop = float(config_[1:])
                            elif config_.startswith('ml'):
                                maxlength = int(config_[2:])
                            elif config_.startswith('rms'):
                                ln = False
                            else:
                                raise f'Pattern {config_} has not been implemented!'
                    self.layer_list.append(rnn_decorate(TransformerDecoder(ndim, nhead, 4 * ndim, nlayer, pdrop, ln)))
                    self.rnn_hidden_state_input_size.append(maxlength)
                # elif self.layer_type[ind].startswith('transformer'):
                #     num_heads = 8
                #     num_layer = 6
                #     if '_' in self.layer_type[ind]:
                #         for config_ in self.layer_type[ind].split('_')[1:]:
                #             if config_.startswith('h'):
                #                 num_heads = int(config_[1:])
                #             elif config_.startswith('l'):
                #                 num_layer = int(config_[1:])
                #             else:
                #                 raise f'Pattern {config_} has not been implemented!'
                #     # print(f'MAMBA parameter: d_state: {d_state}, d_conv: {d_conv}')
                #     assert lst_nh == item, f'mamba_simple require input_dim == output_dim, while got {lst_nh} and {item}'
                #     transormer_block = TransformerEncoderBlock(lst_nh, lst_nh, num_heads=num_heads, num_layers=num_layer)
                #     self.layer_list.append(rnn_decorate(transormer_block))
                #     self.rnn_hidden_state_input_size.append(transormer_block.desired_hidden_dim)
                elif self.layer_type[ind].startswith('conv1d'):
                    if '_' in self.layer_type[ind]:
                        conv_d = int(self.layer_type[ind].split('_')[-1])
                    else:
                        conv_d = 4
                    conv1d = Conv1d(lst_nh, item, d_conv=conv_d)
                    self.layer_list.append(rnn_decorate(conv1d))
                    self.rnn_hidden_state_input_size.append(conv1d.desired_hidden_dim)
                elif self.layer_type[ind].startswith('econv1d'):
                    layer_name, num_ensemble =  self.layer_type[ind].split('-')
                    num_ensemble = int(num_ensemble)
                    if '_' in layer_name:
                        conv_d = int(layer_name.split('_')[-1])
                    else:
                        conv_d = 4
                    econv1d = EConv1d(lst_nh, item, num_ensemble=num_ensemble, d_conv=conv_d)
                    self.layer_list.append(rnn_decorate(econv1d))
                    self.rnn_hidden_state_input_size.append(econv1d.desired_hidden_dim)
                else:
                    self.rnn_hidden_state_input_size.append(item)
                    self.layer_list.append(rnn_decorate(layer_dict[self.layer_type[ind]](lst_nh, item, batch_first=True)))

                self.rnn_layer_type.append(self.layer_type[ind])
            if '+' in activation[ind]:
                norm, activation_name = activation[ind].split('+')
                if norm.startswith('eln'):
                    # norm following an activation
                    ensemble_num = int(norm.split('-')[-1])
                    norm = norm.split('-')[0]
                    self.activation_list.append(torch.nn.ModuleList([norm_dict[norm]([ensemble_num, item]), activation_dict[activation_name]()]))
                else:
                    self.activation_list.append(torch.nn.ModuleList([norm_dict[norm](item), activation_dict[activation_name]()]))
            else:
                self.activation_list.append(activation_dict[activation[ind]]())

            lst_nh = item
        self.input_size = input_size
        assert len(self.layer_list) == len(self.activation_list), "number of layer should be equal to the number of activation"
        self.xavier_initialize_weights()

    def xavier_initialize_weights(self):
        for m in self.layer_list:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, EnsembleLinear):
                for i in range(m.weight.shape[0]):
                    torch.nn.init.xavier_uniform_(m.weight[i].transpose(0, 1))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, LRULayer):
                for efc in [m.in_proj, m.middle_proj]:
                    for i in range(efc.weight.shape[0]):
                        torch.nn.init.xavier_uniform_(efc.weight[i].transpose(0, 1))
                    if hasattr(efc, 'bias') and efc.bias is not None:
                        torch.nn.init.constant_(efc.bias, 0)
            elif isinstance(m, EnsembleLRULayer):
                for efc in [m.in_proj, m.middle_proj]:
                    for i in range(efc.weight.shape[0]):
                        for j in range(efc.weight.shape[1]):
                            torch.nn.init.xavier_uniform_(efc.weight[i, j].transpose(0, 1))
                    if hasattr(efc, 'bias') and efc.bias is not None:
                        torch.nn.init.constant_(efc.bias, 0)
            elif isinstance(m, GILRLayer):
                fc = m.out_proj
                torch.nn.init.xavier_uniform_(fc.weight)
                if fc.bias is not None:
                    torch.nn.init.constant_(fc.bias, 0)
                for efc in [m.in_proj]:
                    for i in range(efc.weight.shape[0]):
                        torch.nn.init.xavier_uniform_(efc.weight[i].transpose(0, 1))
                    if hasattr(efc, 'bias') and efc.bias is not None:
                        torch.nn.init.constant_(efc.bias, 0)
            elif isinstance(m, EnsembleGILRLayer):
                efc = m.out_proj
                for i in range(efc.weight.shape[0]):
                    torch.nn.init.xavier_uniform_(efc.weight[i].transpose(0, 1))
                if hasattr(efc, 'bias') and efc.bias is not None:
                    torch.nn.init.constant_(efc.bias, 0)
                for efc in [m.in_proj]:
                    for i in range(efc.weight.shape[0]):
                        for j in range(efc.weight.shape[1]):
                            torch.nn.init.xavier_uniform_(efc.weight[i, j].transpose(0, 1))
                    if hasattr(efc, 'bias') and efc.bias is not None:
                        torch.nn.init.constant_(efc.bias, 0)
            elif isinstance(m, GILRLSTMLayer):
                fc = m.out_proj
                torch.nn.init.xavier_uniform_(fc.weight)
                if fc.bias is not None:
                    torch.nn.init.constant_(fc.bias, 0)
                for efc in [m.in_proj, m.middle_proj]:
                    for i in range(efc.weight.shape[0]):
                        torch.nn.init.xavier_uniform_(efc.weight[i].transpose(0, 1))
                    if hasattr(efc, 'bias') and efc.bias is not None:
                        torch.nn.init.constant_(efc.bias, 0)
            elif isinstance(m, EnsembleGILRLSTMLayer):
                efc = m.out_proj
                for i in range(efc.weight.shape[0]):
                    torch.nn.init.xavier_uniform_(efc.weight[i].transpose(0, 1))
                if hasattr(efc, 'bias') and efc.bias is not None:
                    torch.nn.init.constant_(efc.bias, 0)
                for efc in [m.in_proj, m.middle_proj]:
                    for i in range(efc.weight.shape[0]):
                        for j in range(efc.weight.shape[1]):
                            torch.nn.init.xavier_uniform_(efc.weight[i, j].transpose(0, 1))
                    if hasattr(efc, 'bias') and efc.bias is not None:
                        torch.nn.init.constant_(efc.bias, 0)
            elif isinstance(m, MambaResidualBlock):
                pass
            elif isinstance(m, MambaBlockList):
                pass
            elif isinstance(m, (Conv1d, EConv1d)):
                pass
            # elif isinstance(m, TransformerEncoderBlock):
            #     pass
            elif GPTLayer is not None and isinstance(m, GPTLayer):
                pass
            elif TransformerDecoder is not None and isinstance(m, TransformerDecoder):
                pass
            else:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        # 使用某种初始化方法，这里使用了xavier_uniform_作为例子
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        # 如果存在偏置，可以选择将其初始化为0
                        torch.nn.init.constant_(param.data, 0)

    def rnn_parameters(self, recursive=True):
        parameters = []
        for layer_type, layer in zip(self.layer_type, self.layer_list):
            if self.check_is_rnn(layer_type):
                if hasattr(layer, 'rnn_parameters'):
                    parameters = parameters + list(layer.rnn_parameters())
                else:
                    parameters = parameters + list(layer.parameters(recursive))
        return parameters

    def to(self, target):
        for item in self.layer_list:
            item.to(target)
        for item in self.activation_list:
            if isinstance(item, torch.nn.ModuleList):
                item[0].to(target)

    def make_init_state(self, batch_size: int, device: Union[str, torch.device]=torch.device("cpu")) -> RNNHidden:
        """
        Make initial hidden state
        :param batch_size: batch size of the hidden state
        :param device: device of the hidden state
        :return: init state
        """
        init_states = RNNHidden(self.rnn_num, self.rnn_layer_type, device)
        for input_size, rnn_type in zip(self.rnn_hidden_state_input_size, self.rnn_layer_type):
            init_states.append(init_states.init_hidden_by_type(rnn_type, batch_size, input_size, device))
        return init_states

    def make_rnd_init_state(self, batch_size: int, device: Union[str, torch.device]=torch.device("cpu")) -> RNNHidden:
        """
        Make initial hidden state randomly sampled from the space of the hidden state
        :param batch_size: batch size of the hidden state
        :param device: device of the hidden state
        :return: init random hidden state
        """
        init_states = RNNHidden(self.rnn_num, self.rnn_layer_type, device)
        for input_size, rnn_type in zip(self.rnn_hidden_state_input_size, self.rnn_layer_type):
            init_states.append(init_states.init_random_hidden_by_type(rnn_type, batch_size, input_size, device))
        return init_states

    def meta_forward(self, x: torch.Tensor, hidden_state: Optional[RNNHidden]=None, require_full_hidden: bool=False) -> Tuple[torch.Tensor, RNNHidden, Optional[RNNHidden]]:
        """
        network forward
        :param x: network input (seq_len, dim), (batch_size, seq_len, dim), (..., batch_size, seq_len, dim)
        :param hidden_state: hidden state
        :param require_full_hidden: if True, return the full output of each RNN layer (output for every timestep will be returned)
        :return: network output, rnn hidden state, output rnn
        """
        assert x.shape[-1] == self.input_size, f"inputting size does not match!!!! input is {x.shape[-1]}, expected: {self.input_size}"
        if hidden_state is None:
            hidden_state = self.make_init_state(x.shape[0], x.device)
        assert len(hidden_state) == self.rnn_num, f"rnn num does not match, input is {len(hidden_state)}, expected: {self.rnn_num}"
        x_dim = len(x.shape)
        assert x_dim >= 2, f"dim of input is {x_dim}, which < 1"
        if x_dim == 2 and self.rnn_num > 0:
            x = torch.unsqueeze(x, 0)
        rnn_count = 0
        # the meaning of each dimension of each hidden state in output_hidden_state: 1, batch_size, dim
        output_hidden_state = RNNHidden(self.rnn_num, self.rnn_layer_type, device=x.device, batch_first=False)
        # the meaning of each dimension of each hidden state in output_rnn: batch_size, timestep, dim
        if require_full_hidden:
            output_rnn = RNNHidden(self.rnn_num, self.rnn_layer_type, device=x.device, batch_first=True)
        else:
            output_rnn = None
        for ind, layer in enumerate(self.layer_list):
            activation = self.activation_list[ind]
            layer_type = self.layer_type[ind]
            if self.check_is_rnn(layer_type):
                if 'gilr' in layer_type or 'cgru' in layer_type:
                    x, h = layer(x, hidden_state[rnn_count], hidden_state.rnn_start)
                elif 'lru' in layer_type:
                    x, h = layer(x, hidden_state[rnn_count], hidden_state.rnn_start, hidden_state.grad_detach)
                elif layer_type.startswith('mamba'):
                    x, h = layer(x, hidden_state[rnn_count], hidden_state.rnn_start, hidden_state.mask, hidden_state.grad_detach)
                elif layer_type.startswith('smamba'):
                    x, h = layer(x, hidden_state[rnn_count], hidden_state.rnn_start, hidden_state.mask)
                elif layer_type.startswith('transformer'):
                    x, h = layer(x, hidden_state[rnn_count])
                elif 'conv1d' in layer_type:
                    x, h = layer(x, hidden_state[rnn_count], hidden_state.mask)
                elif 'gpt' in layer_type:
                    if len(x.shape) == 3 and x.shape[-2] > 1:
                        hidden_item_ = None
                        attention_mask = hidden_state.attention_concat_mask
                    else:
                        hidden_item_ = hidden_state[rnn_count]
                        attention_mask = None
                    if layer_type.startswith('gpt'):
                        x = layer(x, inference_params=hidden_item_, attention_mask_in_length=attention_mask)
                    elif layer_type.startswith('cgpt'):
                        x = layer(x, inference_params=hidden_item_, seqlens=attention_mask)
                    else:
                        raise ValueError(f"unsupported layer type: {layer_type}")
                    h = hidden_item_ if hidden_item_ is not None else hidden_state[rnn_count]
                    if hidden_item_ is not None:
                        hidden_item_.seqlen_offset += x.shape[-2]
                else:
                    x, h = layer(x, hidden_state[rnn_count])
                rnn_count += 1
                output_hidden_state.append(h)
                if require_full_hidden:
                    output_rnn.append(x)
            else:
                x = layer(x)
            if isinstance(activation, torch.nn.ModuleList):
                if '+' in self.activation_type[ind] and self.activation_type[ind].startswith('eln'):
                    x = activation[0](x.transpose(-2, 0))
                    x = x.transpose(-2, 0)
                else:
                    x = activation[0](x)
                x = activation[1](x)
            else:
                x = activation(x)
        if x_dim == 2 and self.rnn_num > 0:
            x = torch.squeeze(x, 0)
        return x, output_hidden_state, output_rnn if require_full_hidden else None

    @staticmethod
    def _copy_weight_from(dst_net: "RNNBase", src_net: "RNNBase", tau: float):
        """
        copy weight of a source network to a destination network
        :param dst_net: target network (destination network)
        :param src_net: source network
        :param tau: soft copy weight, if tau <- 0.0, src will be completely copied to dst. if tau <- 1.0, dst will not change.
        :return: None
        """
        with torch.no_grad():
            if tau == 0.0:
                dst_net.load_state_dict(src_net.state_dict())
                return
            elif tau == 1.0:
                return
            assert len([*src_net.parameters(True)]) == len([*dst_net.parameters(True)]), f"parameter number show be equal!"
            for param, target_param in zip(src_net.parameters(True), dst_net.parameters(True)):
                target_param.data.copy_(target_param.data * tau + (1-tau) * param.data)

    def copy_weight_from(self, src_net: "RNNBase", tau: float):
        """
        copy weight from a source network
        :param src_net: source network
        :param tau: soft copy weight, if tau <- 0.0, self will be completely copied to dst. if tau <- 1.0, self will not change.
        :return: None
        """
        RNNBase._copy_weight_from(self, src_net, tau)

    def info(self, info):
        """
        print
        """
        print(info)

    def save(self, path):
        """
        save state dict to path
        :param path: weight save path
        :return: None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.info(f'saving model to {path}..')
        torch.save(self.state_dict(), path)

    def load(self, path, **kwargs):
        """
        load state dict from path
        :param path: data path
        :param kwargs: other parameters
        :return: None
        """
        self.info(f'loading from {path}..')
        map_location = None
        if 'map_location' in kwargs:
            map_location = kwargs['map_location']
        self.load_state_dict(torch.load(path, map_location=map_location))

    def l2_norm_square(self) -> torch.Tensor:
        return sum([torch.sum(param ** 2) for param in self.parameters(True)])
