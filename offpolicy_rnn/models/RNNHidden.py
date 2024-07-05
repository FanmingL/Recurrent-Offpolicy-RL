import copy

import numpy as np
import torch
from typing import List, Union, Tuple, Dict
try:
    from .flash_attention.TransformerFlashAttention import InferenceParams
except Exception as _:
    InferenceParams = None


class RNNHidden:
    SUPPORTED_RNN_TYPES = ['gru', 'lru', 'gilr', 'cgru', 'gilr_lstm', 'mamba', 'conv1d', 'smamba', 'transformer']
    def __init__(self, rnn_num: int, rnn_types: List[str], device: torch.device=torch.device('cpu'), batch_first=False):
        """
        :param rnn_num:
        :param rnn_types:
        :param device:
        :param batch_first: if true, the shape of the item in self._data is (batch_size, timestep, dim), if false, the shape is (1, batch_size, dim)
        Note: batch_first is False by default. Only when RNN hidden is stored with full RNN output, batch_first is True. When batch_first is True, can only call reshape_full_rnn_output_to_hidden
        """
        assert len(rnn_types) == rnn_num, f'number of rnn layers should be equal to the rnn types'
        if batch_first:
            for rnn_type in rnn_types:
                assert not rnn_type == 'lstm', f'It is not supported to store full RNN output of LSTM!!!'
        self._rnn_types: List[str] = copy.deepcopy(rnn_types)
        self._rnn_num: int = rnn_num
        self._data: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = []
        self._device = device
        self._batch_first = batch_first
        self._rnn_start = None
        self._attention_concat_mask = None
        self._mask = None
        self._grad_detach = None

    def set_rnn_start(self, rnn_start):
        self._rnn_start = rnn_start

    def set_attention_concat_mask(self, attention_concat_mask):
        self._attention_concat_mask = attention_concat_mask

    def set_mask(self, mask):
        self._mask = mask

    def set_grad_detach(self, grad_detach):
        self._grad_detach = grad_detach

    @property
    def rnn_start(self):
        return self._rnn_start

    @property
    def attention_concat_mask(self):
        return self._attention_concat_mask

    @property
    def mask(self):
        return self._mask

    @property
    def grad_detach(self):
        return self._grad_detach

    @property
    def size(self) -> int:
        return len(self._data)
    @property
    def device(self) -> torch.device:
        return self._device
    @property
    def capacity(self) -> int:
        return self._rnn_num

    def append(self, hidden_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], rnn_type=None) -> None:
        assert len(self._data) < self.capacity, f'hidden num exceeds the number of RNN layers'
        # assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        if isinstance(hidden_state, tuple):
            assert self._device == hidden_state[0].device, f'data device does not match the RNNHidden device, expected {self._device} got {hidden_state[0].device}'
        elif InferenceParams is not None and isinstance(hidden_state, InferenceParams):
            pass
        else:
            assert self._device == hidden_state.device, f'data device does not match the RNNHidden device, expected {self._device} got {hidden_state.device}'
        if rnn_type is not None:
            assert rnn_type == self._rnn_types[self.size], f'current RNN type does not match the desired type, expected {self._rnn_types[self.size]}, got {rnn_type}'

        self._data.append(hidden_state)

    def _check_is_rnn(self, rnn_type):
        return (rnn_type in self.SUPPORTED_RNN_TYPES or (rnn_type.startswith('e') and rnn_type[1:].split('-')[0] in self.SUPPORTED_RNN_TYPES)
                or rnn_type.startswith('conv1d') or rnn_type.startswith('econv1d') or rnn_type.startswith('mamba') or rnn_type.startswith('smamba') or rnn_type.startswith('transformer'))

    @torch.no_grad()
    def init_hidden_by_type(self, rnn_type: str, batch_size: int, unit_num: int, device: Union[str, torch.device]) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if rnn_type == 'lstm':
            return (torch.zeros((1, batch_size, unit_num), device=device),
                                torch.zeros((1, batch_size, unit_num), device=device))
        elif rnn_type.startswith('gpt') or rnn_type.startswith('cgpt'):
            return InferenceParams(max_seqlen=unit_num, max_batch_size=batch_size)
        elif self._check_is_rnn(rnn_type):
            return torch.zeros((1, batch_size, unit_num), device=device)
        else:
            raise NotImplementedError(f'rnn type: {rnn_type} has not been implemented!!')

    @torch.no_grad()
    def init_random_hidden_by_type(self, rnn_type: str, batch_size: int, unit_num: int, device: Union[str, torch.device]) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if rnn_type == 'lstm':
            return (torch.rand((1, batch_size, unit_num), device=device) * 2 - 1,
                                torch.rand((1, batch_size, unit_num), device=device) * 2 - 1)
        elif rnn_type.startswith('gpt') or rnn_type.startswith('cgpt'):
            return InferenceParams(max_seqlen=unit_num, max_batch_size=batch_size)
        elif self._check_is_rnn(rnn_type):
            return torch.rand((1, batch_size, unit_num), device=device) * 2 - 1
        else:
            raise NotImplementedError(f'rnn type: {rnn_type} has not been implemented!!')


    def __getitem__(self, key) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], "RNNHidden"]:
        if isinstance(key, slice):
            rnn_types = self._rnn_types[key]
            data = self._data[key]
            rnn_num = len(data)
            result = RNNHidden(rnn_num, rnn_types, self._device, self._batch_first)
            result._data = data
            result._rnn_start = self._rnn_start
            result._attention_concat_mask = self._attention_concat_mask
            result._mask = self._mask
            result._grad_detach = self._grad_detach
            return result
        else:
            return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __len__(self) -> int:
        return self.size

    def elementwise_append(self, data: "RNNHidden") -> None:
        assert data._rnn_num == self._rnn_num, f'rnn num should be equal! {data._rnn_num} != {self._rnn_num}'
        assert data.size == self.size, f'hidden num should be equal! {data.size} != {self.size}'
        assert data._device == self._device, f'device should be equal!  {data._device} != {self._device}'
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'

        for i in range(self.size):
            cur_data = self._data[i]
            apd_data = data._data[i]
            if cur_data is None:
                self._data[i] = apd_data
            elif isinstance(cur_data, tuple):
                self._data[i] = (
                    torch.cat((cur_data[0], apd_data[0]), dim=1),
                    torch.cat((cur_data[1], apd_data[1]), dim=1),
                )
            elif InferenceParams is not None and isinstance(cur_data, InferenceParams):
                raise NotImplementedError(f'GPT does not support element wise append!')
            else:
                self._data[i] = torch.cat((cur_data, apd_data), dim=1)

    def elementwise_pop(self, pop_num: int=1) -> None:
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'

        for i in range(self.size):
            cur_data = self._data[i]
            if cur_data is not None:
                if isinstance(cur_data, tuple):
                    if cur_data[0].shape[1] <= pop_num:
                        self._data[i] = None
                    else:
                        self._data[i] = (
                            (cur_data[0][:, pop_num:, :],
                             cur_data[1][:, pop_num:, :])
                        )
                elif InferenceParams is not None and isinstance(cur_data, InferenceParams):
                    raise NotImplementedError(f'GPT does not support elementwise_pop!')
                else:
                    if cur_data.shape[1] <= pop_num:
                        self._data[i] = None
                    else:
                        self._data[i] = cur_data[:, pop_num:, :]

    @property
    def hidden_batch_size(self) -> int:
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        if self.size == 0:
            return 0
        if self._data[0] is not None:
            if isinstance(self._data[0], tuple):
                return self._data[0][0].shape[1]
            elif InferenceParams is not None and isinstance(self._data[0], InferenceParams):
                raise NotImplementedError(f'GPT does not support hidden_batch_size!')
            else:
                return self._data[0].shape[1]
        return 0

    @staticmethod
    def append_hidden_state(hidden_state: "RNNHidden", data: "RNNHidden") -> "RNNHidden":
        hidden_state.elementwise_append(data)
        return hidden_state

    @staticmethod
    def pop_hidden_state(hidden_state: "RNNHidden", pop_num: int=1) -> "RNNHidden":
        hidden_state.elementwise_pop(pop_num)
        return hidden_state

    @staticmethod
    def get_hidden_length(hidden_state: "RNNHidden") -> int:
        return hidden_state.hidden_batch_size

    def __copy__(self) -> "RNNHidden":
        data = RNNHidden(self._rnn_num, self._rnn_types, self._device, self._batch_first)
        data._data = self._data
        data._rnn_start = self._rnn_start
        data._attention_concat_mask = self._attention_concat_mask
        data._mask = self._mask
        data._grad_detach = self._grad_detach
        return data

    def __deepcopy__(self, memo: Dict) -> "RNNHidden":
        data = RNNHidden(self._rnn_num, self._rnn_types, self._device, self._batch_first)
        data_list = []
        for item in self._data:
            if isinstance(item, tuple):
                data_list.append((item[0].clone(), item[1].clone()))
            elif InferenceParams is not None and isinstance(self._data[0], InferenceParams):
                data_list.append(copy.deepcopy(item))
            else:
                data_list.append(item.clone())

        data._data = data_list
        if self._rnn_start is not None:
            data._rnn_start = self._rnn_start.clone()
        if self._attention_concat_mask is not None:
            data._attention_concat_mask = self._attention_concat_mask.clone()
        if self._mask is not None:
            data._mask = self._mask.clone()
        if self._grad_detach is not None:
            data._grad_detach = self._grad_detach.clone()
        return data

    def to_device(self, device: torch.device) -> None:
        if not self._device == device:
            self._device = device
            for i in range(self.size):
                if isinstance(self._data[i], tuple):
                    self._data[i] = (self._data[i][0].to(device), self._data[i][1].to(device))
                elif InferenceParams is not None and isinstance(self._data[0], InferenceParams):
                    pass
                else:
                    self._data[i] = self._data[i].to(device)

    def reshape_full_rnn_output_to_hidden_(self, target_traj_len: int) -> "RNNHidden":
        assert self._batch_first == True, f'only full_rnn_output variable can call this function!'
        if self.size > 0:
            assert isinstance(self._data[0], torch.Tensor), f'only GRU supports '
        traj_len = 0
        if self.size > 0:
            traj_len = self._data[0].shape[1]
        idx = [i * target_traj_len for i in range(traj_len // target_traj_len)]
        for i in range(self.size):
            self._data[i] = self._data[i][:, idx, :].transpose(0, 1)
            it_shape = self._data[i].shape
            self._data[i] = self._data[i].reshape((1, it_shape[0] * it_shape[1], it_shape[2]))
        self._batch_first = False
        return self

    def reshape_full_rnn_output_to_hidden(self, target_traj_len: int) -> "RNNHidden":
        assert self._batch_first == True, f'only full_rnn_output variable can call this function!'

        if self.size > 0:
            assert isinstance(self._data[0], torch.Tensor), f'only GRU supports '
        result = copy.deepcopy(self)
        return result.reshape_full_rnn_output_to_hidden_(target_traj_len)

    def full_rnn_insert_init_hidden_(self, init_hidden: "RNNHidden"=None, pop_final_hidden=False) -> "RNNHidden":
        if init_hidden is None:
            init_hidden = RNNHidden(self._rnn_num, self._rnn_types, self._device, batch_first=False)
            for i in range(self.size):
                batch_size = self._data[i].shape[0]
                unit_num = self._data[i].shape[-1]
                init_hidden.append(self.init_hidden_by_type(self._rnn_types[i], batch_size, unit_num, self._device))
        for i in range(self.size):
            if not pop_final_hidden:
                self._data[i] = torch.cat((init_hidden[i].squeeze(0).unsqueeze(1), self._data[i]), dim=1)
            else:
                self._data[i] = torch.cat((init_hidden[i].squeeze(0).unsqueeze(1), self._data[i][..., :-1, :]), dim=1)

        return self

    def sample_full_rnn_output_to_hidden_(self, traj_idxes: List[int], time_idxes: List[int]) -> "RNNHidden":
        assert self._batch_first == True, f'only full_rnn_output variable can call this function!'
        if self.size > 0:
            assert isinstance(self._data[0], torch.Tensor), f'only GRU supports '
        for i in range(self.size):
            self._data[i] = self._data[i][traj_idxes, time_idxes, :].unsqueeze(0)
        self._batch_first = False
        return self

    def sample_full_rnn_output_to_hidden(self, traj_idxes: List[int], time_idxes: List[int]) -> "RNNHidden":
        assert self._batch_first == True, f'only full_rnn_output variable can call this function!'
        if self.size > 0:
            assert isinstance(self._data[0], torch.Tensor), f'only GRU supports '
        result = RNNHidden(self._rnn_num, self._rnn_types, self._device, False)
        data_list = []
        for item in self._data:
            data_list.append(item[traj_idxes, time_idxes, :].unsqueeze(0).clone())
        result._data = data_list
        return result

    def hidden_state_mask_(self, masks: Union[np.ndarray, torch.LongTensor]) -> None:
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        for i in range(self.size):
            if isinstance(self._data[i], tuple):
                self._data[i] = (self._data[i][0].squeeze(0)[masks].unsqueeze(0),
                                 self._data[i][1].squeeze(0)[masks].unsqueeze(0))
            elif InferenceParams is not None and isinstance(self._data[i], InferenceParams):
                raise NotImplementedError(f'GPT does not support hidden_state_mask_!')
            else:
                self._data[i] = self._data[i].squeeze(0)[masks].unsqueeze(0)

    def hidden_state_mask(self, masks: Union[np.ndarray, torch.LongTensor]) -> "RNNHidden":
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        result = copy.deepcopy(self)
        result.hidden_state_mask_(masks)
        return result

    def hidden_state_sample_(self, idxes: Union[List[int], np.ndarray]) -> None:
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        for i in range(self.size):
            if isinstance(self._data[i], tuple):
                self._data[i] = (self._data[i][0][:, idxes],
                                   self._data[i][1][:, idxes])
            elif InferenceParams is not None and isinstance(self._data[i], InferenceParams):
                raise NotImplementedError(f'GPT does not support hidden_state_sample_!')
            else:
                self._data[i] = self._data[i][:, idxes]

    def hidden_state_sample(self, idxes: Union[List[int], np.ndarray]) -> "RNNHidden":
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        result = copy.deepcopy(self)
        result.hidden_state_sample_(idxes)
        return result

    def hidden_detach_(self) -> None:
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        for i in range(self.size):
            if isinstance(self._data[i], tuple):
                self._data[i] = (self._data[i][0].detach(),
                                 self._data[i][1].detach())
            elif InferenceParams is not None and isinstance(self._data[i], InferenceParams):
                raise NotImplementedError(f'GPT does not support hidden_detach_!')
            else:
                self._data[i] = self._data[i].detach()

    def hidden_detach(self) -> "RNNHidden":
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        result = copy.deepcopy(self)
        result.hidden_detach_()
        return result

    def hidden_state_slice_(self, start: int, end: int) -> None:
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        for i in range(self.size):
            if isinstance(self._data[i], tuple):
                self._data[i] = (self._data[i][0][:, start: end],
                                 self._data[i][1][:, start: end])
            elif InferenceParams is not None and isinstance(self._data[i], InferenceParams):
                raise NotImplementedError(f'GPT does not support hidden_state_slice_!')
            else:
                self._data[i] = self._data[i][:, start: end]
    def hidden_state_slice(self, start: int, end: int) -> "RNNHidden":
        assert self._batch_first == False, f'only when batch_first is False, this function is callable'
        result = copy.deepcopy(self)
        result.hidden_state_slice_(start, end)
        return result

    def __add__(self, other: "RNNHidden") -> "RNNHidden":
        if isinstance(other, RNNHidden):
            assert self._device == other._device, f'devices mismatch!! got {self._device} and {other._device}'
            res = RNNHidden(rnn_num=self._rnn_num+other._rnn_num, rnn_types=self._rnn_types+other._rnn_types, device=self._device, batch_first=self._batch_first)
            res._data = self._data + other._data
            return res
        elif other is None:
            return self
        else:
            return NotImplemented

    def __str__(self):
        res = ''
        for i, data in enumerate(self._data):
            if isinstance(data, torch.Tensor):
                res += f'RNN hidden {i+1}/{len(self._data)} {data.shape}: {data}\n'
            else:
                res += f'RNN hidden {i+1}/{len(self._data)}: {data}\n'

        if len(res) > 0 and res[-1] == '\n':
            res = res[:-1]
        return res