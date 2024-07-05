import os
from .rnn_base import RNNBase
import torch
from typing import List, Union, Tuple, Dict, Optional, Callable
from .RNNHidden import RNNHidden
from .torch_utility import fixed_length_forward
from .mlp_base import MLPBase

class ContextualModel:
    def __init__(self, embedding_input_size: int, embedding_size: int, embedding_hidden: List[int], embedding_activations: List[str],
                 embedding_layer_type: List[str], uni_model_input_size: int, uni_model_output_size: int, uni_model_hidden: List[int],
                 uni_model_activations: List[str],
                 uni_model_layer_type: List[str], fix_rnn_length: int, name: str, uni_model_input_mapping_dim: int=0, uni_model_input_mapping_activation: str='linear'):
        self.name = name
        self._fix_rnn_length = fix_rnn_length
        self.fix_rnn_length = fix_rnn_length
        self.embedding_size = embedding_size
        self.uni_model_input_mapping_dim = uni_model_input_mapping_dim
        self.embedding_network = RNNBase(embedding_input_size, embedding_size, embedding_hidden,
                                         embedding_activations, embedding_layer_type)
        uni_network_input = uni_model_input_size if self.uni_model_input_mapping_dim == 0 else self.uni_model_input_mapping_dim
        self.uni_network = RNNBase(embedding_size + uni_network_input, uni_model_output_size, uni_model_hidden,
                                   uni_model_activations, uni_model_layer_type)
        self.contextual_modules: Dict[str, RNNBase] = dict()
        # if self.input_layer_norm:
        #     self.embedding_network_input_layer_norm = torch.nn.LayerNorm(embedding_input_size)
        #     self.uni_network_layer_norm = torch.nn.LayerNorm(uni_model_input_size)
        #     self.contextual_register_rnn_base_module(self.embedding_network_input_layer_norm, 'embedding_network_input_layer_norm')
        #     self.contextual_register_rnn_base_module(self.uni_network_layer_norm, 'uni_network_layer_norm')
        self.contextual_register_rnn_base_module(self.embedding_network, 'embedding_model')
        self.contextual_register_rnn_base_module(self.uni_network, 'universal_model')
        if self.uni_model_input_mapping_dim > 0:
            self.uni_input_mapping_network = MLPBase(uni_model_input_size, uni_model_input_mapping_dim, [], [uni_model_input_mapping_activation])
            self.contextual_register_rnn_base_module(self.uni_input_mapping_network, 'uni_input_mapping_network')
        else:
            self.uni_input_mapping_network = torch.nn.Identity()
        self.rnn_num = self.embedding_network.rnn_num + self.uni_network.rnn_num
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def contextual_register_rnn_base_module(self, module: RNNBase, module_name: str):
        self.contextual_modules[module_name] = module

    def parameters(self, recursive=True) -> List[torch.Tensor]:
        parameter = []
        for k, v in self.contextual_modules.items():
            parameter += list(v.parameters(recursive))
        return parameter

    def rnn_parameters(self, recursive=True):
        parameter = []
        for k, v in self.contextual_modules.items():
            if hasattr(v, 'rnn_parameters'):
                parameter += list(v.rnn_parameters(recursive))
        return parameter

    def meta_forward(self, embedding_input: torch.Tensor, uni_model_input: torch.Tensor, rnn_memory=None, detach_embedding=False) -> Tuple[
        torch.Tensor, RNNHidden, torch.Tensor, RNNHidden
    ]:
        """
        :param embedding_input: (seq_len, dim) or (batch_size, seq_len, dim)
        :param uni_model_input: (seq_len, dim) or (batch_size, seq_len, dim)
        :param rnn_memory: RNNHidden
        :return:
        """
        if rnn_memory is None:
            rnn_memory = self.make_init_state(1 if len(embedding_input.shape) == 2 else embedding_input.shape[0], embedding_input.device)
        embedding_output, embedding_rnn_memory, embedding_full_memory = self._meta_forward_embedding(embedding_input,
                                                                                                     rnn_memory)
        if detach_embedding:
            embedding_output = embedding_output.detach()
        uni_model_output, uni_model_rnn_memory, uni_model_full_memory = self._meta_forward_uni_model(uni_model_input,
                                                                                                     embedding_output,
                                                                                                     rnn_memory)

        full_rnn_memory = embedding_full_memory + uni_model_full_memory if self.fix_rnn_length <= 0 else None
        rnn_memory = embedding_rnn_memory + uni_model_rnn_memory
        return uni_model_output, rnn_memory, embedding_output, full_rnn_memory

    def _meta_forward_embedding(self, embedding_input: torch.Tensor, rnn_memory: Optional[RNNHidden]) -> Tuple[
        torch.Tensor, RNNHidden, RNNHidden
    ]:
        # if self.input_layer_norm:
        #     embedding_input = self.embedding_network_input_layer_norm(embedding_input)
        rnn_memory_embedding = rnn_memory[:self.embedding_network.rnn_num] if rnn_memory is not None and len(rnn_memory) > 0 else None
        if self.fix_rnn_length > 0 and self.embedding_network.rnn_num > 0:
            # if fix_rnn_length is not 0, use fixed length_forward function
            output, rnn_memory_embedding = fixed_length_forward(embedding_input, self.embedding_network, rnn_memory_embedding, fixed_length=self.fix_rnn_length)
            full_hidden_embedding = None
        else:
            # otherwise, use the default forward function
            output, rnn_memory_embedding, full_hidden_embedding = self.embedding_network.meta_forward(embedding_input,
                                                                                                      rnn_memory_embedding,
                                                                                                      require_full_hidden=True)
        return output, rnn_memory_embedding, full_hidden_embedding

    def _meta_forward_uni_model(self, uni_model_input: torch.Tensor, embedding: torch.Tensor, rnn_memory: Optional[RNNHidden])-> Tuple[
        torch.Tensor, RNNHidden, RNNHidden
    ]:
        # if self.input_layer_norm:
        #     uni_model_input = self.uni_network_layer_norm(uni_model_input)
        uni_model_input = self.uni_input_mapping_network(uni_model_input)
        rnn_memory_uni_model = rnn_memory[self.embedding_network.rnn_num:] if rnn_memory is not None and len(rnn_memory) > 0 else None
        if len(embedding.shape) - len(uni_model_input.shape) == 1:
            uni_model_input = uni_model_input.unsqueeze(0).repeat_interleave(repeats=embedding.shape[0], dim=0)
        uni_model_input = torch.cat((uni_model_input, embedding), dim=-1)
        if self.fix_rnn_length > 0 and self.uni_network.rnn_num > 0:
            # if fix_rnn_length is not 0, use fixed length_forward function
            output, rnn_memory_uni_model = fixed_length_forward(uni_model_input, self.uni_network, rnn_memory_uni_model, fixed_length=self.fix_rnn_length)
            full_hidden_embedding = None
        else:
            # otherwise, use the default forward function
            output, rnn_memory_uni_model, full_hidden_embedding = self.uni_network.meta_forward(uni_model_input,
                                                                                                   rnn_memory_uni_model,
                                                                                                   require_full_hidden=True)
        return output, rnn_memory_uni_model, full_hidden_embedding

    def to(self, device: torch.device=None, dtype: torch.dtype=None) -> None:
        if device is not None and not self.device == device:
            self.device = device
            for k, v in self.contextual_modules.items():
                v.to(device)
        if dtype is not None and not self.dtype == dtype:
            self.dtype = dtype
            for k, v in self.contextual_modules.items():
                v.to(dtype)

    def get_embedding(self, x, rnn_memory) -> Tuple[
        torch.Tensor, RNNHidden, RNNHidden
    ]:
        embedding_output, embedding_rnn_memory, embedding_full_memory = self._meta_forward_embedding(x,
                                                                                                     rnn_memory)
        return embedding_output, embedding_rnn_memory, embedding_full_memory

    def save(self, path: str, index: int=0) -> None:
        for k, v in self.contextual_modules.items():
            name = f'{self.name}-{index}-{k}.pt'
            full_path = os.path.join(path, name)
            if hasattr(v, 'save'):
                v.save(full_path)
            else:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                torch.save(v.state_dict(), full_path)

    def load(self, path: str, index: int=0, **kwargs) -> None:
        for k, v in self.contextual_modules.items():
            name = f'{self.name}-{index}-{k}.pt'
            full_path = os.path.join(path, name)
            if hasattr(v, 'load'):
                v.load(full_path, **kwargs)
            else:
                item = torch.load(full_path, **kwargs)
                v.load_state_dict(item)


    def copy_weight_from(self, src: "ContextualModel", tau: float) -> None:
        """I am target net, tau ~~ 1
            if tau = 0, self <--- src_net
            if tau = 1, self <--- self
        """
        for k, v in self.contextual_modules.items():
            RNNBase._copy_weight_from(v, src.contextual_modules[k], tau)
            # v.copy_weight_from(src.contextual_modules[k], tau)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = dict()
        for k, v in self.contextual_modules.items():
            state_dict[k] = v.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in self.contextual_modules.items():
            v.load_state_dict(state_dict[k])

    def make_init_state(self, batch_size: int, device: torch.device) -> RNNHidden:
        if self.fix_rnn_length > 1:
            # if fix rnn length > 0, padding zeros before data
            if self.embedding_network.rnn_num > 0:
                output, embedding_hidden = fixed_length_forward(torch.zeros((batch_size, self.fix_rnn_length - 1, self.embedding_network.input_size), device=device), self.embedding_network, None, fixed_length=self.fix_rnn_length)
            else:
                embedding_hidden = self.embedding_network.make_init_state(batch_size, device)
            if self.uni_network.rnn_num > 0:
                _, uni_model_hidden = fixed_length_forward(torch.cat((torch.zeros((batch_size, self.fix_rnn_length - 1, self.uni_network.input_size - output.shape[-1]), device=device), output), dim=-1), self.uni_network, None, fixed_length=self.fix_rnn_length)
            else:
                uni_model_hidden = self.uni_network.make_init_state(batch_size, device)
        else:
            embedding_hidden = self.embedding_network.make_init_state(batch_size, device)
            uni_model_hidden = self.uni_network.make_init_state(batch_size, device)
        hidden = embedding_hidden + uni_model_hidden
        return hidden

    def make_rnd_init_state(self, batch_size, device):
        embedding_hidden = self.embedding_network.make_rnd_init_state(batch_size, device)
        uni_model_hidden = self.uni_network.make_rnd_init_state(batch_size, device)
        hidden = embedding_hidden + uni_model_hidden
        return hidden

    def generate_hidden_state(self, embedding_input: torch.Tensor, uni_model_input: torch.Tensor, slice_num: int,
                              initial_hidden_process_func: Callable=None, hidden_process_func: Callable=None):
        with torch.no_grad():
            batch_size = embedding_input.shape[0]
            device = embedding_input.device
            hidden_state_now = self.make_init_state(batch_size, device)

            _, _, _, full_hidden = self.meta_forward(embedding_input, uni_model_input, hidden_state_now)
            if hidden_process_func is not None and full_hidden is not None:
                hidden_process_func(full_hidden)
                initial_hidden_process_func(hidden_state_now)
            full_hidden.full_rnn_insert_init_hidden_(hidden_state_now, pop_final_hidden=True)
            hidden_states_res = full_hidden.reshape_full_rnn_output_to_hidden(slice_num)
        return hidden_states_res

    def train(self, mode=True):
        for k, v in self.contextual_modules.items():
            v.train(mode)

    def eval(self):
        for k, v in self.contextual_modules.items():
            v.eval()

    def set_fix_length(self, enable: bool):
        if enable:
            self.fix_rnn_length = self._fix_rnn_length
        else:
            self.fix_rnn_length = 0

    def l2_norm_square(self) -> torch.Tensor:
        return sum([module.l2_norm_square() for k, module in self.contextual_modules.items() if hasattr(module, 'l2_norm_square')])